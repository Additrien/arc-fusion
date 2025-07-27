"""
Corpus Retrieval Agent - Advanced RAG pipeline with HyDE and hybrid search.

This agent implements the full RAG pipeline:
1. HyDE (Hypothetical Document Embeddings) for query expansion
2. Hybrid search (vector + BM25) using Weaviate
3. Parent document retrieval for context
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from weaviate.classes.query import MetadataQuery
from app.agents.state import GraphState
from app.agents.registry import AgentRegistry
from app.core.vector_store import VectorStore
from app.core.embeddings.embedding_service import EmbeddingService
from app.core.config.services import VectorStoreConfig, EmbeddingConfig
from app.utils.logger import get_logger
from app.utils.cache import embedding_cache, hyde_cache
from app.utils.performance import time_async_block
from app import config
from app.prompts import (
    HYDE_PROMPT_HEADER, HYDE_CONVERSATION_CONTEXT_SECTION, HYDE_QUERY_SECTION,
    format_conversation_history
)

logger = get_logger('arc_fusion.agents.corpus_retrieval')


class CorpusRetrievalService:
    """
    Service for retrieving documents from the corpus.
    
    This service handles:
    1. Query embedding generation via Gemini API
    2. Hypothetical Document Embedding (HyDE) generation
    3. Hybrid search (vector + BM25) against Weaviate
    4. Parent chunk reconstruction
    """
    
    def __init__(self):
        # Use models from central config with new API
        self.hyde_model = config.PRIMARY_MODEL
        
        # Load configurations
        self.vector_config = VectorStoreConfig()
        self.embedding_config = EmbeddingConfig()
        
        # Core services
        self.vector_store = VectorStore(config=self.vector_config)
        self.embedding_service = EmbeddingService(config=self.embedding_config)
        
        # Configure Gemini with new API
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
        logger.info("Corpus retrieval service initialized without reranking")

    @time_async_block("corpus_retrieval.retrieve")
    async def retrieve(self, state: GraphState) -> GraphState:
        """
        Execute the full RAG pipeline: HyDE -> Hybrid Search -> Parent Retrieval.
        
        Args:
            state: Current graph state containing the query and session info
            
        Returns:
            Updated state with retrieved and reranked content
        """
        query = state.get("query", "")
        session_id = state.get("session_id", "unknown")
        
        logger.info(f"Starting corpus retrieval for session {session_id}")
        
        try:
            async with time_async_block("corpus_retrieval.execute_pipeline"):
                # Get conversation history for context and ensure it's in the right format
                conversation_history = state.get("conversation_messages", [])
                
                # Defensive handling: convert any LangGraph message objects to plain dicts
                safe_conversation_history = []
                for msg in conversation_history:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):  # LangGraph message object
                        # Convert LangGraph message to plain dict
                        safe_conversation_history.append({
                            "role": getattr(msg, 'type', 'unknown'),
                            "content": str(msg.content),
                            "timestamp": getattr(msg, 'timestamp', None)
                        })
                    elif isinstance(msg, dict):  # Already a plain dict
                        safe_conversation_history.append(msg)
                    else:
                        # Skip unknown message types
                        continue
                
                conversation_history = safe_conversation_history
                
                # Step 1&2: Parallelize HyDE and base embedding generation
                hyde_task = asyncio.create_task(self._generate_hypothetical_document(query, conversation_history))
                base_embedding_task = asyncio.create_task(self._generate_query_embedding(query))
                
                # Wait for both to complete
                expanded_query, base_embedding = await asyncio.gather(hyde_task, base_embedding_task)
                
                # Step 3: Generate final embedding from expanded query (or use base if HyDE failed)
                if expanded_query != query:  # HyDE succeeded
                    query_embedding = await self._generate_query_embedding(expanded_query)
                else:  # HyDE failed, use base embedding
                    query_embedding = base_embedding
                
                # Step 4: Perform hybrid search using Weaviate
                child_results = await self.vector_store.hybrid_search(query, query_embedding, limit=config.INITIAL_RETRIEVAL_K)
                
                if not child_results:
                    logger.warning("No results found from hybrid search")
                    return self._create_empty_result(state)
                
                logger.info(f"Retrieved {len(child_results)} child chunks from hybrid search")
                
                # Step 4: Get parent chunks for better context
                parent_chunks, sources, hybrid_scores = await self._get_parent_chunks(child_results)
                
                if not parent_chunks:
                    logger.warning("No parent chunks found")
                    return self._create_empty_result(state)
                
                logger.info(f"Retrieved {len(parent_chunks)} unique parent chunks.")
                
                # Debug: Log the actual scores we're getting
                if hybrid_scores:
                    logger.info(f"Hybrid scores range: min={min(hybrid_scores):.4f}, max={max(hybrid_scores):.4f}, avg={sum(hybrid_scores)/len(hybrid_scores):.4f}")
                    logger.info(f"First 5 scores: {[f'{score:.4f}' for score in hybrid_scores[:5]]}")
                
                # Debug: Also log some sample child results to see distance vs score
                if child_results:
                    logger.info(f"Sample child result - distance: {child_results[0].get('distance', 'missing')}, score: {child_results[0].get('score', 'missing')}")
                    logger.info(f"Child result keys: {list(child_results[0].keys())}")
                
                # Filter out chunks with very low scores
                scored_tuples = list(zip(hybrid_scores, parent_chunks, sources))

                # Apply minimum score threshold
                logger.info(f"Using min_chunk_score threshold: {self.vector_config.min_chunk_score}")
                filtered_tuples = [t for t in scored_tuples if t[0] >= self.vector_config.min_chunk_score]
                removed_count = len(scored_tuples) - len(filtered_tuples)
                
                if removed_count > 0:
                    logger.info(f"Filtered out {removed_count} chunks with scores below {self.vector_config.min_chunk_score}")
                
                if not filtered_tuples:
                    logger.warning("No chunks remain after score filtering")
                    return self._create_empty_result(state)

            # Use top chunks based on hybrid scores up to the limit
            filtered_tuples.sort(key=lambda x: x[0], reverse=True)
            final_pairs = filtered_tuples[:config.FINAL_SELECTION_K]
            final_context = [pair[1] for pair in final_pairs]
            retrieval_scores = [pair[0] for pair in final_pairs]  # Use actual hybrid scores
            
            # Add hybrid scores to sources
            final_sources = []
            for i, (score, chunk, source) in enumerate(final_pairs):
                source_with_score = source.copy()
                source_with_score['score'] = float(score)  # Add hybrid score
                final_sources.append(source_with_score)
            
            best_retrieval_score = max(retrieval_scores) if retrieval_scores else 0.0
            
            logger.info(f"Selected {len(final_context)} final chunks. Best score: {best_retrieval_score:.3f}")
            
            # ReAct Enhancement: Generate observations and evidence for the planner
            observation = self._generate_observation(
                query, final_context, final_sources, retrieval_scores, best_retrieval_score
            )
            
            # Gather evidence for ReAct reasoning
            evidence = self._gather_evidence(final_context, final_sources, retrieval_scores)
            
            # Determine if replanning is needed based on retrieval quality
            needs_replanning = self._should_trigger_replanning(
                best_retrieval_score, len(final_context), query
            )
            
            # Update state for the orchestrator with ReAct enhancements
            updated_state = state.copy()
            updated_state.update({
                "retrieved_context": final_context,
                "document_sources": final_sources,
                "retrieval_scores": retrieval_scores,
                "best_retrieval_score": best_retrieval_score,
                "step_count": state.get("step_count", 0) + 1,
                "tasks_completed": state.get("tasks_completed", []) + ["corpus_retrieval"],
                # ReAct enhancements
                "observations": state.get("observations", []) + [observation],
                "gathered_evidence": state.get("gathered_evidence", []) + evidence,
                "needs_replanning": needs_replanning,
                "current_focus": self._determine_focus(query, best_retrieval_score, len(final_context))
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in corpus retrieval: {str(e)}")
            return self._create_error_result(state, str(e))
    
    @time_async_block("corpus_retrieval.hyde_expansion")
    async def _generate_hypothetical_document(self, query: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        # Include conversation context in cache key for follow-up questions
        history_context = ""
        if conversation_history:
            # Use last 2 exchanges for context in cache key
            recent_messages = conversation_history[-4:]
            # Apply defensive handling for cache key generation
            safe_msgs = []
            for msg in recent_messages:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):  # LangGraph message object
                    safe_msgs.append(f"{getattr(msg, 'type', '')}: {str(msg.content)[:50]}")
                elif isinstance(msg, dict):  # Plain dict
                    safe_msgs.append(f"{msg.get('role', '')}:{msg.get('content', '')[:50]}")
            history_context = "|".join(safe_msgs)
        
        cache_key = f"hyde:{query}|{history_context}"
        cached_result = hyde_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Build HyDE prompt using centralized prompts
        prompt_parts = [HYDE_PROMPT_HEADER]
        
        # Add conversation context if available
        if conversation_history:
            formatted_history = format_conversation_history(conversation_history)
            prompt_parts.append(HYDE_CONVERSATION_CONTEXT_SECTION.format(
                conversation_history=formatted_history
            ))
        
        # Add query section
        prompt_parts.append(HYDE_QUERY_SECTION.format(query=query))
        
        hyde_prompt = "\n".join(prompt_parts)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.hyde_model, 
                contents=hyde_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking for speed
                )
            )
            hypothetical_doc = response.text.strip()
            expanded_query = f"{query} {hypothetical_doc}"
            hyde_cache.set(cache_key, expanded_query)
            return expanded_query
        except Exception as e:
            logger.warning(f"HyDE generation failed, using original query: {str(e)}")
            return query

    @time_async_block("corpus_retrieval.embedding_generation")
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate query embedding using EmbeddingService."""
        cache_key = f"embedding:{query}"
        cached_result = embedding_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Use EmbeddingService for consistency and unified configuration
            embeddings = await self.embedding_service.generate_embeddings([query])
            embedding = embeddings[0]
            embedding_cache.set(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed, using empty embedding: {str(e)}")
            return [0.0] * config.FALLBACK_EMBEDDING_DIM

    @time_async_block("corpus_retrieval.parent_retrieval")
    async def _get_parent_chunks(self, child_results: List[dict]) -> Tuple[List[str], List[dict], List[float]]:
        """Get parent chunks from child search results, preserving hybrid scores."""
        parent_chunks = []
        sources = []
        hybrid_scores = []
        seen_parent_ids = set()
        
        for result in child_results:
            parent_id = result.get('parent_id')
            if parent_id and parent_id not in seen_parent_ids:
                parent_chunk = await self.vector_store.get_parent_chunk_by_id(parent_id)
                if parent_chunk:  # parent_chunk is already the content string
                    parent_chunks.append(parent_chunk)
                    sources.append({
                        'parent_id': parent_id,
                        'document_id': result.get('document_id'),  # Get from original result
                        'filename': result.get('filename', 'Unknown'),
                        'chunk_index': result.get('parent_index', 0)
                    })
                    # Preserve the hybrid score from Weaviate
                    hybrid_scores.append(result.get('score', 0.0))
                    seen_parent_ids.add(parent_id)
        
        return parent_chunks, sources, hybrid_scores

    def _create_empty_result(self, state: GraphState) -> GraphState:
        """Create an empty result state."""
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": [],
            "document_sources": [],
            "retrieval_scores": [],
            "best_retrieval_score": 0.0,
            "step_count": state.get("step_count", 0) + 1,
            "tasks_completed": state.get("tasks_completed", []) + ["corpus_retrieval"]
        })
        return updated_state

    def _create_error_result(self, state: GraphState, error_message: str) -> GraphState:
        """Create an error result state."""
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": [],
            "document_sources": [],
            "retrieval_scores": [],
            "best_retrieval_score": 0.0,
            "errors": state.get("errors", {}).copy(),
            "step_count": state.get("step_count", 0) + 1,
            "tasks_completed": state.get("tasks_completed", []) + ["corpus_retrieval"]
        })
        updated_state["errors"]["retrieval_error"] = error_message
        return updated_state
    
    def _generate_observation(self, query: str, context: List[str], sources: List[Dict[str, Any]], 
                             scores: List[float], best_score: float) -> Dict[str, Any]:
        """Generate ReAct observation from retrieval results."""
        
        # Analyze retrieval quality
        if best_score >= config.RETRIEVAL_HIGH_QUALITY_THRESHOLD:
            quality_assessment = "high"
        elif best_score >= config.RETRIEVAL_MEDIUM_QUALITY_THRESHOLD:
            quality_assessment = "medium"
        else:
            quality_assessment = "low"
        
        # Determine status and details
        if not context:
            status = "no_results_found"
            details = "No relevant documents found in corpus"
        elif best_score < config.RETRIEVAL_LOW_QUALITY_THRESHOLD:
            status = "quality_too_low"
            details = f"Best relevance score {best_score:.3f} is too low for reliable answers"
        elif len(context) < config.RETRIEVAL_SUFFICIENT_CONTEXT_CHUNKS:
            status = "context_insufficient"
            details = f"Only {len(context)} relevant chunk found, may need additional sources"
        else:
            status = "retrieval_successful"
            details = f"Found {len(context)} relevant chunks with best score {best_score:.3f}"
        
        return {
            "agent": "corpus_retrieval",
            "status": status,
            "details": details,
            "quality_assessment": quality_assessment,
            "chunks_found": len(context),
            "best_score": best_score,
            "quality_too_low": best_score < config.RETRIEVAL_LOW_QUALITY_THRESHOLD,
            "context_insufficient": len(context) < config.RETRIEVAL_SUFFICIENT_CONTEXT_CHUNKS,
            "needs_web_fallback": best_score < config.RETRIEVAL_MEDIUM_QUALITY_THRESHOLD,
            "timestamp": 0  # Will be updated by caller
        }
    
    def _gather_evidence(self, context: List[str], sources: List[Dict[str, Any]], 
                        scores: List[float]) -> List[Dict[str, Any]]:
        """Gather evidence from retrieval results for ReAct reasoning."""
        evidence = []
        
        for i, (chunk, source, score) in enumerate(zip(context, sources, scores)):
            # Extract key information from chunk
            summary = chunk[:config.EVIDENCE_SUMMARY_LENGTH] + "..." if len(chunk) > config.EVIDENCE_SUMMARY_LENGTH else chunk
            
            evidence.append({
                "source": source.get("filename", "Unknown"),
                "summary": summary,
                "quality_score": float(score),
                "chunk_index": i + 1,
                "evidence_type": "document_chunk"
            })
        
        return evidence
    
    def _should_trigger_replanning(self, best_score: float, chunk_count: int, query: str) -> bool:
        """Determine if replanning should be triggered based on retrieval quality."""
        
        # Trigger replanning if:
        triggers = [
            best_score < config.REPLANNING_LOW_QUALITY_SCORE,
            chunk_count == 0,
            chunk_count == 1 and "compare" in query.lower(),
            best_score < config.REPLANNING_INSUFFICIENT_CONTEXT_SCORE and any(term in query.lower() for term in ["recent", "latest", "current", "new"])
        ]
        
        should_replan = any(triggers)
        
        if should_replan:
            logger.info(f"ReAct: Triggering replanning due to retrieval quality issues. Best score: {best_score:.3f}, chunks: {chunk_count}")
        
        return should_replan
    
    def _determine_focus(self, query: str, best_score: float, chunk_count: int) -> str:
        """Determine what the system should focus on next."""
        
        if best_score < config.RETRIEVAL_LOW_QUALITY_THRESHOLD:
            return "web_search_fallback"
        elif chunk_count < config.RETRIEVAL_SUFFICIENT_CONTEXT_CHUNKS and any(word in query.lower() for word in ["compare", "versus", "vs"]):
            return "comparison_needs_more_sources"
        elif best_score < config.RETRIEVAL_MEDIUM_QUALITY_THRESHOLD:
            return "supplementary_web_search"
        else:
            return "synthesis_ready"


# Create the service instance
corpus_retrieval_service = CorpusRetrievalService()

@AgentRegistry.register(
    name="corpus_retrieval",
    capabilities=["document_search", "rag", "hybrid_search"],
    dependencies=["routing"]
)
async def corpus_retrieval_agent(state: GraphState) -> GraphState:
    """
    Advanced corpus retrieval agent using HyDE, hybrid search, and reranking.
    
    This agent implements a sophisticated RAG pipeline that combines:
    - HyDE for query expansion
    - Hybrid search (vector + BM25) for recall
    - Cross-encoder reranking for precision
    """
    return await corpus_retrieval_service.retrieve(state)
