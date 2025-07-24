"""
Corpus Retrieval Agent - Advanced RAG pipeline with HyDE and hybrid search.

This agent implements the full RAG pipeline:
1. HyDE (Hypothetical Document Embeddings) for query expansion
2. Hybrid search (vector + BM25) using Weaviate
3. Parent document retrieval for context
4. Re-ranking with Qwen3-Reranker using direct transformers approach
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
import torch
from google import genai
from google.genai import types
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Any, Optional
import asyncio
from app.agents.state import GraphState
from app.agents.registry import AgentRegistry
from app.core.vector_store import VectorStore
from app.utils.logger import get_logger
from app.utils.cache import embedding_cache, hyde_cache
from app.utils.performance import time_async_block
from app import config

logger = get_logger('arc_fusion.agents.corpus_retrieval')


class CorpusRetrievalService:
    """
    Service for retrieving and reranking documents from the corpus.
    
    This service handles:
    1. Query embedding generation via Gemini API
    2. Hypothetical Document Embedding (HyDE) generation
    3. Hybrid search (vector + BM25) against Weaviate
    4. Document reranking using Qwen3-Reranker model
    5. Parent chunk reconstruction
    """
    
    def __init__(self):
        # Use models from central config with new API
        self.hyde_model = config.PRIMARY_MODEL
        
        # Embedding model for queries - using new google-genai SDK
        self.embedding_model = config.EMBEDDING_MODEL
        
        # Core services
        self.vector_store = VectorStore()
        
        # Configure Gemini with new API
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

        # Device detection for reranker model
        if torch.cuda.is_available() and config.DEVICE in ["auto", "cuda"]:
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu" 
            logger.info("Using CPU for local models")
        
        # Initialize reranker components
        self.reranker_model = None
        self.tokenizer = None
        self.token_true_id = None
        self.token_false_id = None
        self.prefix_tokens = None
        self.suffix_tokens = None
        self.max_length = 8192
        
        if config.ENABLE_RERANKING:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL, padding_side='left')
                
                # Configure model loading parameters
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                }
                
                if self.device == "cuda" and config.ENABLE_MODEL_QUANTIZATION:
                    # Use new BitsAndBytesConfig approach - keep model on GPU
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0  # Only offload outliers above threshold
                        # Remove llm_int8_enable_fp32_cpu_offload=True to keep model on GPU
                    )
                    model_kwargs.update({
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                    })
                    logger.info("Enabled 8-bit quantization with GPU placement for optimal performance")
                elif self.device == "cpu" and config.ENABLE_MODEL_QUANTIZATION:
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True
                    })
                    logger.info("Enabled FP16 precision for CPU model")
                
                self.reranker_model = AutoModelForCausalLM.from_pretrained(
                    config.RERANKER_MODEL, 
                    **model_kwargs
                ).eval()
                
                if self.device == "cuda" and "device_map" not in model_kwargs:
                    self.reranker_model = self.reranker_model.cuda()
                
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                
                # Prepare prefix and suffix tokens
                prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
                self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
                
                logger.info("Reranker model loaded successfully with BitsAndBytesConfig approach.")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {str(e)}")

    def _get_device(self) -> str:
        """Determines the optimal device for Torch models based on config and availability."""
        device = config.DEVICE
        if device == "auto":
            if torch.cuda.is_available():
                selected_device = "cuda"
                logger.info("Auto-detected CUDA-enabled GPU. Using 'cuda' device.")
            else:
                selected_device = "cpu"
                logger.info("No CUDA-enabled GPU found. Using 'cpu' device for local models.")
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("Device was configured to 'cuda' but no GPU is available. Falling back to 'cpu'.")
                selected_device = "cpu"
            else:
                selected_device = "cuda"
        else:
            selected_device = "cpu"
        
        return selected_device

    def format_instruction(self, instruction: str, query: str, doc: str) -> str:
        """Format instruction according to Qwen3-Reranker format."""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs: List[str]) -> dict:
        """Process input pairs for the Qwen3-Reranker model."""
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.reranker_model.device)
            
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs: dict) -> List[float]:
        """Compute reranking scores using the Qwen3-Reranker approach."""
        batch_scores = self.reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    @time_async_block("corpus_retrieval.retrieve_and_rerank")
    async def retrieve_and_rerank(self, state: GraphState) -> GraphState:
        """
        Execute the full RAG pipeline: HyDE -> Hybrid Search -> Parent Retrieval -> Reranking.
        
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
                # Step 1: Generate hypothetical document for query expansion (HyDE)
                expanded_query = await self._generate_hypothetical_document(query)
                
                # Step 2: Generate embedding for the expanded query
                query_embedding = await self._generate_query_embedding(expanded_query)
                
                # Step 3: Perform hybrid search using Weaviate
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
                
                # Filter out chunks with very low scores
                scored_tuples = list(zip(hybrid_scores, parent_chunks, sources))

                # Apply minimum score threshold
                filtered_tuples = [t for t in scored_tuples if t[0] >= config.MIN_CHUNK_SCORE]
                removed_count = len(scored_tuples) - len(filtered_tuples)
                
                if removed_count > 0:
                    logger.info(f"Filtered out {removed_count} chunks with scores below {config.MIN_CHUNK_SCORE}")
                
                if not filtered_tuples:
                    logger.warning("No chunks remain after score filtering")
                    return self._create_empty_result(state)

            # Step 5: Pre-filter chunks for reranking based on hybrid scores
            if config.ENABLE_RERANKING and self.reranker_model and filtered_tuples and self.tokenizer:
                # Sort by hybrid scores and select top PRE_RERANKER_TOP_K chunks
                filtered_tuples.sort(key=lambda x: x[0], reverse=True)
                
                # Limit to PRE_RERANKER_TOP_K for reranking
                pre_rerank_tuples = filtered_tuples[:config.PRE_RERANKER_TOP_K]
                pre_rerank_chunks = [t[1] for t in pre_rerank_tuples]
                pre_rerank_sources = [t[2] for t in pre_rerank_tuples]
                
                logger.info(f"Pre-filtered to {len(pre_rerank_chunks)} chunks for reranking (from {len(parent_chunks)} total). Top hybrid scores: {[f'{t[0]:.3f}' for t in pre_rerank_tuples[:3]]}")
                
                logger.info(f"Reranking {len(pre_rerank_chunks)} chunks with model {config.RERANKER_MODEL}...")
                
                # Format pairs according to Qwen3-Reranker format
                task_instruction = 'Given a question about academic research, determine if the passage directly addresses the question and contains relevant information to answer it'
                formatted_pairs = [
                    self.format_instruction(task_instruction, query, chunk) 
                    for chunk in pre_rerank_chunks
                ]
                
                try:
                    # Process in small batches to avoid CUDA out of memory
                    batch_size = 2 if self.device == "cuda" else 4  # Single items for GPU to avoid OOM
                    rerank_scores = []
                    
                    for i in range(0, len(formatted_pairs), batch_size):
                        batch = formatted_pairs[i:i + batch_size]
                        logger.info(f"Processing reranking batch {i//batch_size + 1}/{(len(formatted_pairs) + batch_size - 1)//batch_size}")
                        
                        # Process batch
                        batch_inputs = self.process_inputs(batch)
                        batch_scores = self.compute_logits(batch_inputs)
                        rerank_scores.extend(batch_scores)
                        
                        # Debug: Log actual reranker scores for this batch
                        logger.info(f"Batch {i//batch_size + 1} reranker scores: {[f'{s:.4f}' for s in batch_scores]}")
                    
                    logger.info(f"Successfully reranked {len(rerank_scores)} chunks. All reranker scores: {[f'{s:.4f}' for s in rerank_scores]}")
                    
                    # Verify reranker produced valid scores
                    if not rerank_scores or all(s == 0.5 for s in rerank_scores):
                        logger.warning("Reranker produced default/empty scores, falling back to hybrid scores")
                        # Use hybrid scores instead of reranker scores
                        rerank_scores = [t[0] for t in pre_rerank_tuples]  # Extract hybrid scores
                        logger.info(f"Using hybrid scores instead: {[f'{s:.4f}' for s in rerank_scores]}")
                    
                except Exception as e:
                    logger.error(f"Reranking failed: {e}, using hybrid scores as fallback")
                    # Use hybrid scores instead of default uniform scores
                    rerank_scores = [t[0] for t in pre_rerank_tuples]  # Extract hybrid scores
                    logger.info(f"Fallback hybrid scores: {[f'{s:.4f}' for s in rerank_scores]}")
                
                # Combine scores with pre-filtered chunks and sources
                scored_parent_pairs = list(zip(rerank_scores, pre_rerank_chunks, pre_rerank_sources))
                
                # Sort by the new reranker score in descending order
                scored_parent_pairs.sort(key=lambda x: x[0], reverse=True)
                
                logger.info(f"Top 3 reranked scores: {[f'{s[0]:.3f}' for s in scored_parent_pairs[:3]]}")
                
                # Select the top K results after reranking
                final_pairs = scored_parent_pairs[:config.RERANKER_TOP_K]
                final_context = [pair[1] for pair in final_pairs]
                retrieval_scores = [pair[0] for pair in final_pairs]
                
                # Add scores to the sources for citation display
                final_sources = []
                for i, (score, chunk, source) in enumerate(final_pairs):
                    source_with_score = source.copy()
                    source_with_score['score'] = float(score)  # Add reranker score
                    final_sources.append(source_with_score)
                    
            else:
                # No reranking - use top chunks based on hybrid scores up to the limit
                # filtered_tuples is already sorted and filtered by minimum score
                final_pairs = filtered_tuples[:config.RERANKER_TOP_K]
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
            
            # Update state for the orchestrator
            updated_state = state.copy()
            updated_state.update({
                "retrieved_context": final_context,
                "document_sources": final_sources,
                "retrieval_scores": retrieval_scores,
                "best_retrieval_score": best_retrieval_score,
                "step_count": state.get("step_count", 0) + 1,
                "tasks_completed": state.get("tasks_completed", []) + ["corpus_retrieval"]
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in corpus retrieval: {str(e)}")
            return self._create_error_result(state, str(e))
    
    @time_async_block("corpus_retrieval.hyde_expansion")
    async def _generate_hypothetical_document(self, query: str) -> str:
        cache_key = f"hyde:{query}"
        cached_result = hyde_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        hyde_prompt = f"You are an expert academic researcher. Given a research question, write a detailed paragraph that would likely appear in an academic paper answering this question.\n\nResearch Question: {query}\n\nWrite a comprehensive paragraph (100-200 words) that would contain the answer:"
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
        cache_key = f"embedding:{query}"
        cached_result = embedding_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        try:
            response = await self.client.aio.models.embed_content(
                model=self.embedding_model,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)
            )
            embedding = response.embeddings[0].values
            embedding_cache.set(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed, using empty embedding: {str(e)}")
            return [0.0] * 768  # Return zero embedding as fallback

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
    return await corpus_retrieval_service.retrieve_and_rerank(state)