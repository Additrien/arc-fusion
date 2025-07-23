"""
Centralized configuration for the Arc-Fusion application.

This module consolidates all tunable parameters, model names, and thresholds
to allow for easy adjustments without modifying the core application logic.
"""

# --- LLM Models ---
# The main model for routing and judging. Fast, cost-effective, and powerful.
# See: https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-lite
PRIMARY_MODEL = "gemini-2.5-flash-lite"

# The model for high-quality synthesis.
SYNTHESIS_MODEL = "gemini-2.5-pro"

# The embedding model for retrieval. Using the latest generation model.
EMBEDDING_MODEL = "gemini-embedding-001"


# --- Corpus Retrieval Agent ---
# Number of initial child chunks to retrieve from the vector store.
MAX_CHILD_CHUNKS_RETRIEVAL = 30

# Number of parent chunks to pass to the LLM Judge for evaluation.
MAX_PARENT_CHUNKS_FOR_JUDGING = 10

# Number of final, top-ranked chunks to pass to the synthesis agent.
MAX_FINAL_CHUNKS_FOR_SYNTHESIS = 5


# --- Hybrid Search Relevance ---
# On a scale of 0-1, the minimum hybrid score for a chunk to be considered "relevant".
# This is used in the framework for routing to synthesis vs. web search.
RELEVANCE_THRESHOLD = 0.7


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

# Document processing chunk sizes (optimized for fewer API calls)
PARENT_CHUNK_SIZE = 3000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 1000
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

# Enable rate limiting protection
ENABLE_RATE_LIMITING = True


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