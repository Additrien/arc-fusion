"""
Centralized configuration for the Arc-Fusion application.

This module consolidates all tunable parameters, model names, and thresholds
to allow for easy adjustments without modifying the core application logic.
"""

import os

# --- LLM Models ---
# The main model for routing and judging. Fast, cost-effective, and powerful.
# See: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-lite
PRIMARY_MODEL = "gemini-2.5-flash-lite"

# The model for high-quality synthesis.
# Using Flash for speed while maintaining good quality
SYNTHESIS_MODEL = "models/gemini-2.5-flash"

# Evaluation model for RAG quality assessment
EVALUATION_MODEL = "models/gemini-2.5-flash"

# Dataset generation model
DATASET_GENERATION_MODEL = "models/gemini-2.5-flash"

# The embedding model for retrieval. Using the latest generation model.
EMBEDDING_MODEL = "gemini-embedding-001"


# --- Reranking & Context Selection ---
# Enables the two-stage retrieval process with a Cross-Encoder reranker.
# Can be disabled via ENABLE_RERANKING=false environment variable for faster responses
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# The Cross-Encoder model to use for reranking.
# See: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Enable model quantization for memory optimization
# GPU: 8-bit quantization with automatic device mapping (~50% memory reduction)
# CPU: FP16 precision with low memory usage (~40% smaller, 15-400% faster)
ENABLE_MODEL_QUANTIZATION = os.getenv("ENABLE_MODEL_QUANTIZATION", "true").lower() == "true"

# --- Hardware Configuration ---
# The device to run local models on.
# Options: 'auto', 'cuda', 'cpu'
# 'auto' will use a CUDA-enabled GPU if available, otherwise fallback to CPU.
DEVICE = os.getenv("DEVICE", "auto")

# The number of initial candidates to retrieve from the vector store.
# Reduced for speed - still good coverage with hybrid search
INITIAL_RETRIEVAL_K = 20

# The number of top-scoring chunks from hybrid search to send to the reranker.
# This reduces GPU load by filtering out lower-quality chunks before reranking.
PRE_RERANKER_TOP_K = 8

# The final number of top-ranked documents to pass to the synthesis agent.
RERANKER_TOP_K = 4


# --- Hybrid Search Relevance ---
# On a scale of 0-1, the minimum hybrid score for a chunk to be considered "relevant".
# This is used in the framework for routing to synthesis vs. web search.
RELEVANCE_THRESHOLD = 0.85

# Minimum score threshold for individual chunks (filters out very low quality chunks)
# Chunks with scores below this threshold are discarded before reranking or synthesis
MIN_CHUNK_SCORE = 0.7


# --- Routing Agent ---
# The model used for initial intent classification. Should be fast.
ROUTING_MODEL = PRIMARY_MODEL


# --- Web Search Agent ---
# The model used for optimizing search queries.
WEB_SEARCH_QUERY_MODEL = PRIMARY_MODEL


# --- Document Processing & Storage ---
# Batch size for storing chunks in Weaviate (smaller batches reduce connection timeouts)
WEAVIATE_BATCH_SIZE = 50

# Delay between batches to avoid overwhelming connections (seconds)
BATCH_DELAY_SECONDS = 0.1

# Maximum retries for embedding generation with exponential backoff
EMBEDDING_MAX_RETRIES = 5

# Base delay for embedding retry in seconds (will be exponentially increased)
EMBEDDING_RETRY_BASE_DELAY = 2.0

# Document processing chunk sizes (optimized for reranker context window)
PARENT_CHUNK_SIZE = 3000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 1000  # Stays within the 512 token limit of most rerankers
CHILD_CHUNK_OVERLAP = 100


# --- API Rate Limiting ---
# These settings help manage Gemini free tier limitations
# Gemini free tier limits (as of 2024):
# - 15 requests per minute for embedding model
# - 15 requests per minute for text generation
# - 1 million tokens per day

# Delay between embedding requests to stay under rate limits (seconds)
EMBEDDING_REQUEST_DELAY = 4.0  # 15 requests/minute = 1 request per 4 seconds

# Maximum concurrent embedding requests
MAX_CONCURRENT_EMBEDDINGS = 1  # Free tier is quite restrictive

# Disable rate limiting for speed (use with caution on production)
ENABLE_RATE_LIMITING = False


# --- API Timeouts & Performance ---
# Timeout for corpus retrieval operations (seconds)
CORPUS_RETRIEVAL_TIMEOUT = 30

# Timeout for web search operations (seconds)
WEB_SEARCH_TIMEOUT = 15

# Timeout for LLM generation (seconds)
LLM_GENERATION_TIMEOUT = 30

# Enable debug mode for detailed logging
DEBUG_MODE = True

# Maximum processing time before fallback to web search (seconds)
MAX_PROCESSING_TIME = 45