"""
Corpus Retrieval Agent - Advanced RAG pipeline with HyDE and hybrid search.

This agent implements the full RAG pipeline:
1. HyDE (Hypothetical Document Embeddings) for query expansion
2. Hybrid search (vector + BM25) using Weaviate
3. Parent document retrieval for context
4. Re-ranking and context selection
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
from google import genai
from google.genai import types
from pydantic import BaseModel

from .registry import AgentRegistry
from .state import GraphState
from ..core.vector_store import VectorStore
from ..core.document_processor import DocumentProcessor
from ..utils.logger import get_logger
from .. import config

logger = get_logger('arc_fusion.agents.corpus_retrieval')

# Pydantic models for structured output
class AssessmentResult(BaseModel):
    chunk_index: int
    score: float
    explanation: str
    is_relevant: bool

class JudgeResults(BaseModel):
    assessments: List[AssessmentResult]

# Remove global client initialization - will be done in the service class


class CorpusRetrievalService:
    """Service for advanced corpus retrieval using HyDE and hybrid search."""
    
    def __init__(self):
        # Use models from central config with new API
        self.hyde_model = config.PRIMARY_MODEL
        self.llm_judge_model = config.PRIMARY_MODEL
        logger.info(f"Initialized LLM as a Judge model: {config.PRIMARY_MODEL}")
        
        # Embedding model for queries - using new google-genai SDK
        self.embedding_model = config.EMBEDDING_MODEL
        
        # Core services
        self.vector_store = VectorStore()
        # Initialize DocumentProcessor to access parent chunks
        self.document_processor = DocumentProcessor()
        
        # Configure Gemini with new API
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
    async def retrieve_and_rerank(self, state: GraphState) -> GraphState:
        """
        Execute the full RAG pipeline: HyDE -> Hybrid Search -> Parent Retrieval -> LLM Judging.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with retrieved context and sources
        """
        query = state["query"]
        session_id = state["session_id"]
        
        logger.info(f"Starting corpus retrieval for session {session_id}")
        logger.info(f"Query: {query}")
        
        try:
            # Add timeout for the entire retrieval process
            from app.config import CORPUS_RETRIEVAL_TIMEOUT
            
            # Wrap the entire process in a timeout
            return await asyncio.wait_for(
                self._execute_retrieval_pipeline(state),
                timeout=CORPUS_RETRIEVAL_TIMEOUT
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Corpus retrieval timed out after {CORPUS_RETRIEVAL_TIMEOUT}s")
            return self._create_timeout_state(state)
        except Exception as e:
            logger.error(f"Error in corpus retrieval: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_state(state, e)
    
    async def _execute_retrieval_pipeline(self, state: GraphState) -> GraphState:
        """Execute the core retrieval pipeline with all steps."""
        query = state["query"]
        session_id = state["session_id"]
        
        # Step 1: HyDE Query Expansion
        logger.info("Performing HyDE query expansion")
        expanded_query = await self._generate_hypothetical_document(query)
        logger.info(f"HyDE expanded query length: {len(expanded_query)} chars")
        
        # Step 2: Generate query embedding
        logger.info("Generating query embedding")
        query_embedding = await self._generate_query_embedding(expanded_query)
        logger.info(f"Query embedding generated: {len(query_embedding)} dimensions")
        
        # Step 3: Hybrid search on child chunks
        logger.info(f"Performing hybrid search with limit={config.MAX_CHILD_CHUNKS_RETRIEVAL}")
        child_results = await self.vector_store.hybrid_search(
            query=expanded_query,
            query_embedding=query_embedding,
            limit=config.MAX_CHILD_CHUNKS_RETRIEVAL
        )
        
        logger.info(f"Hybrid search returned {len(child_results)} child results")
        
        if not child_results:
            logger.warning("No results found in corpus - checking database stats")
            # Add database diagnostics
            try:
                stats = await self.vector_store.get_database_stats()
                logger.warning(f"Database contains {stats['child_chunks']} child chunks and {stats['parent_chunks']} parent chunks across {stats['unique_documents']} documents")
                
                # List available documents
                documents = await self.vector_store.get_all_documents()
                logger.warning(f"Available documents: {[doc['filename'] for doc in documents[:5]]}")
                
            except Exception as e:
                logger.error(f"Failed to get database stats: {str(e)}")
            
            return self._create_empty_result_state(state)
        
        # Log sample results for debugging
        logger.info("Sample hybrid search results:")
        for i, result in enumerate(child_results[:3]):  # Log first 3 results
            score = result.get('score', 0)
            distance = result.get('distance', 0)
            
            # Handle cases where score/distance might be None
            score_str = f"{score:.3f}" if score is not None else "N/A"
            distance_str = f"{distance:.3f}" if distance is not None else "N/A"
            
            logger.info(f"  Result {i+1}: score={score_str}, "
                      f"distance={distance_str}, "
                      f"filename={result.get('filename', 'N/A')}, "
                      f"content_preview={result.get('content', '')[:100]}...")
        
        # Step 4: Group by parent chunks and retrieve context
        logger.info("Retrieving parent chunks for context")
        parent_chunks, sources = await self._retrieve_parent_context(child_results)
        logger.info(f"Retrieved {len(parent_chunks)} parent chunks from {len(sources)} sources")
        
        if not parent_chunks:
            logger.warning("No parent chunks found - this indicates parent-child relationship issues")
            # Log some child results for debugging
            logger.warning("Sample child results that failed to resolve to parents:")
            for i, result in enumerate(child_results[:3]):
                logger.warning(f"  Child {i+1}: parent_id={result.get('parent_id')}, "
                             f"document_id={result.get('document_id')}, "
                             f"filename={result.get('filename')}")
            return self._create_empty_result_state(state)
        
        # Log parent chunk info
        logger.info("Retrieved parent chunks:")
        for i, (chunk, source) in enumerate(zip(parent_chunks[:3], sources[:3])):  # Log first 3
            score = source.get('score', 0)
            score_str = f"{score:.3f}" if score is not None else "N/A"
            
            logger.info(f"  Parent {i+1}: score={score_str}, "
                      f"filename={source.get('filename', 'N/A')}, "
                      f"length={len(chunk)} chars, "
                      f"preview={chunk[:100]}...")
        
        # Step 5: Re-rank and select top chunks with LLM as a Judge
        logger.info(f"Re-ranking and selecting top context with LLM as a Judge (judging {len(parent_chunks)} chunks)")
        final_context, final_sources, judge_assessments = await self._select_top_context_with_llm_judge(
            parent_chunks, sources, query
        )
        
        logger.info(f"LLM Judge results: {len(final_context)} chunks selected from {len(parent_chunks)} candidates")
        
        if final_context:
            # Log final selected chunks
            logger.info("Final selected chunks:")
            for i, (chunk, source) in enumerate(zip(final_context, final_sources)):
                llm_score = source.get('llm_judge_score', 'N/A')
                logger.info(f"  Final {i+1}: llm_score={llm_score}, "
                          f"filename={source.get('filename', 'N/A')}, "
                          f"length={len(chunk)} chars")
        else:
            logger.warning("LLM Judge filtered out all chunks - checking scores")
            if judge_assessments:
                scores = [assessment.get('llm_judge_score', 0) for assessment in judge_assessments]
                logger.warning(f"LLM Judge scores: min={min(scores):.1f}, "
                             f"max={max(scores):.1f}, "
                             f"avg={sum(scores)/len(scores):.1f}, "
                             f"threshold={config.RELEVANCE_THRESHOLD}")
        
        logger.info(f"Retrieved {len(final_context)} context chunks from {len(set(s['filename'] for s in final_sources))} documents")
        
        # Extract LLM judge scores for routing decision
        llm_judge_scores = [assessment.get("llm_judge_score", 0.0) for assessment in final_sources]
        best_llm_judge_score = max(llm_judge_scores) if llm_judge_scores else 0.0
        
        logger.info(f"LLM Judge assessment: best_score={best_llm_judge_score:.1f}/10, threshold={config.RELEVANCE_THRESHOLD}")
        
        # Update state - let conditional routing decide next step based on quality
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": final_context,
            "document_sources": final_sources,
            "retrieval_scores": [s.get("score", 0.0) for s in final_sources], # Hybrid search scores
            "llm_judge_assessments": judge_assessments,
            "best_llm_judge_score": best_llm_judge_score,
            "step_count": state.get("step_count", 0) + 1
        })
        
        return updated_state
    
    async def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document using HyDE technique.
        
        HyDE works by generating a document that might contain the answer,
        which often produces better embeddings for retrieval than the raw query.
        """
        
        hyde_prompt = f"""
You are an expert academic researcher. Given a research question, write a detailed paragraph that would likely appear in an academic paper answering this question.

Write as if you are extracting from a real research paper. Include specific methodologies, results, and technical details that would typically appear in academic literature.

Research Question: {query}

Write a comprehensive paragraph (100-200 words) that would contain the answer:
"""
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.hyde_model, 
                contents=hyde_prompt
            )
            hypothetical_doc = response.text.strip()
            
            # Combine original query with hypothetical document for better retrieval
            expanded_query = f"{query} {hypothetical_doc}"
            
            logger.debug(f"HyDE expansion generated {len(hypothetical_doc)} characters")
            return expanded_query
            
        except Exception as e:
            logger.warning(f"HyDE generation failed, using original query: {str(e)}")
            return query
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the query using the new google-genai SDK."""
        try:
            # Use the new google-genai SDK for embeddings
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768  # Fix dimension for consistency
                )
            )
            
            # Extract embedding values - the structure should be response.embeddings[0].values
            if hasattr(response, 'embeddings') and response.embeddings:
                embedding = list(response.embeddings[0].values)
            elif hasattr(response, 'embedding'):
                embedding = list(response.embedding.values)
            else:
                raise ValueError("Unexpected embedding response structure")
                
            logger.debug(f"Generated query embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise
    
    async def _retrieve_parent_context(self, child_results: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Retrieve parent chunks for the top child results.
        
        Args:
            child_results: Results from hybrid search on child chunks
            
        Returns:
            Tuple of (parent_chunks, source_metadata)
        """
        parent_chunks = []
        sources = []
        seen_parents = set()
        
        # Group child results by parent_id and take best score per parent
        parent_groups = {}
        for result in child_results:
            parent_id = result["parent_id"]
            if parent_id not in parent_groups or result["score"] > parent_groups[parent_id]["score"]:
                parent_groups[parent_id] = result
        
        # Sort by score and take top parents
        top_parents = sorted(parent_groups.values(), key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Attempting to retrieve {len(top_parents[:config.MAX_PARENT_CHUNKS_FOR_JUDGING])} parent chunks")
        
        for result in top_parents[:config.MAX_PARENT_CHUNKS_FOR_JUDGING]:  # Get extra for judging
            parent_id = result["parent_id"]
            
            if parent_id not in seen_parents:
                # Use VectorStore to get parent chunk from Weaviate instead of in-memory store
                try:
                    parent_chunk = await self.vector_store.get_parent_chunk_by_id(parent_id)
                    
                    if parent_chunk:
                        parent_chunks.append(parent_chunk)
                        sources.append({
                            "parent_id": parent_id,
                            "document_id": result["document_id"],
                            "filename": result["filename"],
                            "parent_index": result["parent_index"],
                            "score": result["score"],
                            "distance": result.get("distance", 0.0)
                        })
                        seen_parents.add(parent_id)
                        logger.debug(f"Retrieved parent chunk for parent_id: {parent_id}")
                    else:
                        logger.warning(f"Parent chunk not found in Weaviate for parent_id: {parent_id}")
                        
                except Exception as e:
                    logger.error(f"Error retrieving parent chunk {parent_id}: {str(e)}")
        
        logger.info(f"Successfully retrieved {len(parent_chunks)} parent chunks from Weaviate")
        return parent_chunks, sources
    
    async def _select_top_context_with_llm_judge(self, parent_chunks: List[str], sources: List[Dict[str, Any]], 
                                                 query: str) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Selects the top context chunks using an LLM as a Judge model for re-ranking.
        """
        if not parent_chunks:
            return [], [], []

        logger.info(f"Judging {len(parent_chunks)} parent chunks with LLM Judge.")

        prompt = self._build_llm_judge_prompt(query, parent_chunks)

        try:
            # Call the LLM Judge with structured output
            response = await self.client.aio.models.generate_content(
                model=self.llm_judge_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JudgeResults,
                }
            )
            
            # Use the parsed structured response directly
            judge_results: JudgeResults = response.parsed
            assessments = judge_results.assessments

            # Combine chunks, sources, and assessments
            scored_items = []
            for i, (chunk, source) in enumerate(zip(parent_chunks, sources)):
                # Find assessment for this chunk
                assessment = next((a for a in assessments if a.chunk_index == i), None)
                if assessment:
                    source.update({
                        "llm_judge_score": assessment.score,
                        "llm_judge_explanation": assessment.explanation,
                        "llm_judge_is_relevant": assessment.is_relevant,
                        "reranked": True
                    })
                    scored_items.append((chunk, source, assessment.score))

            # Sort by LLM judge score
            scored_items.sort(key=lambda x: x[2], reverse=True)
            
            # Filter for relevant chunks and take top N
            relevant_items = [item for item in scored_items if item[1].get("llm_judge_score", 0) >= config.RELEVANCE_THRESHOLD]
            top_items = relevant_items[:config.MAX_FINAL_CHUNKS_FOR_SYNTHESIS]

            final_chunks = [chunk for chunk, _, _ in top_items]
            final_sources = [source for _, source, _ in top_items]
            
            best_score = max([item[2] for item in top_items]) if top_items else 0.0
            logger.info(f"LLM Judge selected {len(final_chunks)} chunks. Best score: {best_score:.1f}/10")

            return final_chunks, final_sources, final_sources # Return sources which contain the assessments

        except Exception as e:
            logger.error(f"LLM Judge re-ranking failed: {str(e)}")
            logger.warning("Falling back to original hybrid search ranking.")
            # Fallback to hybrid search scores if LLM judge fails
            paired = list(zip(parent_chunks, sources))
            paired.sort(key=lambda x: x[1].get("score", 0), reverse=True)
            
            top_chunks = [chunk for chunk, _ in paired[:config.MAX_FINAL_CHUNKS_FOR_SYNTHESIS]]
            top_sources = [source for _, source in paired[:config.MAX_FINAL_CHUNKS_FOR_SYNTHESIS]]
            
            return top_chunks, top_sources, []

    def _build_llm_judge_prompt(self, query: str, chunks: List[str]) -> str:
        """Constructs the prompt for the LLM as a Judge."""

        chunk_text = ""
        for i, chunk in enumerate(chunks):
            chunk_text += f'<document index="{i}">\n{chunk}\n</document>\n\n'

        prompt = f"""
You are a meticulous and impartial "LLM Judge". Your role is to evaluate a set of retrieved documents based on their relevance to a user's query.

**User Query:**
"{query}"

**Documents to Evaluate:**
{chunk_text}

**Your Task:**
For each document, provide a detailed assessment with the following fields:
- chunk_index: The integer index of the document (0, 1, 2, etc.)
- score: A relevance score from 1.0 (not relevant at all) to 10.0 (perfectly relevant)
- explanation: A brief (1-2 sentences) justification for your score
- is_relevant: true if the document is substantially relevant to answering the query, false otherwise

Evaluate all documents systematically and provide your assessment.
"""
        return prompt

    def _create_empty_result_state(self, state: GraphState) -> GraphState:
        """
        Create state when no results are found.
        
        The conditional routing will automatically detect empty retrieved_context
        and route to web search - no manual intent setting needed.
        """
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": [],
            "document_sources": [],
            "retrieval_scores": [],
            "step_count": state.get("step_count", 0) + 1
            # Note: No manual intent - conditional routing handles fallback to web search
        })
        return updated_state
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """
        Create state when an error occurs.
        
        With empty retrieved_context, conditional routing will attempt web search fallback.
        """
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": [],
            "document_sources": [],
            "retrieval_scores": [],
            "step_count": state.get("step_count", 0) + 1,
            "error_info": {
                "retrieval_error": str(error)
            }
            # Note: No manual intent - let conditional routing decide (likely web search fallback)
        })
        return updated_state

    def _create_timeout_state(self, state: GraphState) -> GraphState:
        """Create state when corpus retrieval times out."""
        updated_state = state.copy()
        updated_state.update({
            "retrieved_context": [],
            "document_sources": [],
            "retrieval_scores": [],
            "step_count": state.get("step_count", 0) + 1,
            "error_info": {
                "retrieval_error": "Corpus retrieval timed out"
            }
        })
        return updated_state


# Create service instance
corpus_retrieval_service = CorpusRetrievalService()


@AgentRegistry.register(
    name="corpus_retrieval",
    capabilities=["document_search", "rag", "hybrid_search"],
    priority=10
)
def corpus_retrieval_agent(state: GraphState) -> GraphState:
    """
    Advanced corpus retrieval agent implementing HyDE + Hybrid Search + Re-ranking.
    
    This agent provides the core RAG functionality with state-of-the-art retrieval methods.
    """
    # Note: LangGraph expects sync functions, so we need to handle async
    import asyncio
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, use run_coroutine_threadsafe
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            corpus_retrieval_service.retrieve_and_rerank(state),
            loop
        )
        return future.result()
    except RuntimeError:
        # No running loop, safe to use run_until_complete
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(corpus_retrieval_service.retrieve_and_rerank(state)) 