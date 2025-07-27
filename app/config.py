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
EVALUATION_MODEL = "models/gemini-2.5-pro"

# Dataset generation model
DATASET_GENERATION_MODEL = "models/gemini-2.5-pro"

# The embedding model for retrieval. Using the latest generation model.
EMBEDDING_MODEL = "gemini-embedding-001"


# The number of initial candidates to retrieve from the vector store.
# Increased for better recall and parallel processing efficiency
INITIAL_RETRIEVAL_K = 50

# The number of top-scoring chunks from hybrid search to select
# Increased to provide more context while maintaining quality
INITIAL_SELECTION_K = 15

# The final number of top-ranked documents to pass to the synthesis agent.
FINAL_SELECTION_K = 4


# --- Hybrid Search Relevance ---
# On a scale of 0-1, the minimum hybrid score for a chunk to be considered "relevant".
# This is used in the framework for routing to synthesis vs. web search.
RELEVANCE_THRESHOLD = 0.85

# Minimum score threshold for individual chunks (filters out very low quality chunks)
# Chunks with scores below this threshold are discarded before synthesis
MIN_CHUNK_SCORE = 0.6


# --- Routing Agent ---
# The model used for initial intent classification. Should be fast.
ROUTING_MODEL = PRIMARY_MODEL


# --- Web Search Agent ---
# The model used for optimizing search queries.
WEB_SEARCH_QUERY_MODEL = PRIMARY_MODEL


# --- Document Processing & Storage ---
# Batch size for storing chunks in Weaviate (larger batches for better performance)
WEAVIATE_BATCH_SIZE = 200

# Delay between batches to avoid overwhelming connections (seconds)
BATCH_DELAY_SECONDS = 0.0  # Removed artificial delay for performance

# Maximum retries for embedding generation with exponential backoff
EMBEDDING_MAX_RETRIES = 5

# Base delay for embedding retry in seconds (will be exponentially increased)
EMBEDDING_RETRY_BASE_DELAY = 2.0

# Document processing chunk sizes (optimized for context and performance)
PARENT_CHUNK_SIZE = 3000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 1000  # Optimal size for hybrid search and retrieval
CHILD_CHUNK_OVERLAP = 100


# --- API Rate Limiting ---
# These settings help manage Gemini free tier limitations
# Gemini free tier limits (as of 2024):
# - 15 requests per minute for embedding model
# - 15 requests per minute for text generation
# - 1 million tokens per day

# Delay between embedding requests to stay under rate limits (seconds)
EMBEDDING_REQUEST_DELAY = 0.1  # Optimized for performance - only small delay for API stability

# Maximum concurrent embedding requests
MAX_CONCURRENT_EMBEDDINGS = 3  # Increased for better throughput

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


# --- Cache Settings ---
# Time-to-live (TTL) for embedding cache in seconds (1 hour)
EMBEDDING_CACHE_TTL = 3600

# Time-to-live (TTL) for HyDE cache in seconds (30 minutes)
HYDE_CACHE_TTL = 1800


# --- Dataset Generation ---
# Default path for the golden dataset
GOLDEN_DATASET_PATH = "data/golden_dataset.jsonl"

# Default number of Q&A pairs to generate for the golden dataset
DEFAULT_QA_PAIRS = 50

# Number of sample questions to retrieve from the golden dataset
DATASET_SAMPLE_QUESTIONS = 5


# --- Clarification Agent ---
# Generation config for clarification agent
CLARIFICATION_TEMP = 0.2
CLARIFICATION_TOP_P = 0.9
CLARIFICATION_MAX_TOKENS = 500

# Number of recent messages to analyze for conversation patterns
CLARIFICATION_CONTEXT_MESSAGES = 3

# Lists of words for ambiguity detection
VAGUE_QUANTIFIERS = ["many", "enough", "few", "some", "several", "most"]
UNDEFINED_REFERENTS = ["it", "this", "that", "they", "them", "these", "those"]
UNCLEAR_COMPARISONS = ["better", "best", "worse", "optimal", "superior"]

# Number of available papers to show in the clarification prompt
CLARIFICATION_MAX_PAPERS_TO_SHOW = 3

# Maximum number of clarification options to return
CLARIFICATION_MAX_OPTIONS = 3

# Default error message for clarification agent
CLARIFICATION_ERROR_MESSAGE = "I'm sorry, but I'm having trouble understanding your question. Could you please provide more specific details about what you're looking for?"

# Default confidence score for clarification agent error
CLARIFICATION_ERROR_CONFIDENCE = 0.3

# Default confidence score for clarification agent response
CLARIFICATION_RESPONSE_CONFIDENCE = 0.9


# --- Corpus Retrieval Agent ---
# Embedding dimension for fallback
FALLBACK_EMBEDDING_DIM = 768

# Quality assessment thresholds
RETRIEVAL_HIGH_QUALITY_THRESHOLD = 0.8
RETRIEVAL_MEDIUM_QUALITY_THRESHOLD = 0.5
RETRIEVAL_LOW_QUALITY_THRESHOLD = 0.3

# Minimum number of chunks for sufficient context
RETRIEVAL_SUFFICIENT_CONTEXT_CHUNKS = 2

# Length of the summary in the evidence
EVIDENCE_SUMMARY_LENGTH = 200

# Replanning triggers
REPLANNING_LOW_QUALITY_SCORE = 0.3
REPLANNING_INSUFFICIENT_CONTEXT_SCORE = 0.5


# --- Synthesis Agent ---
# Generation config for synthesis agent
SYNTHESIS_TEMP = 0.1
SYNTHESIS_TOP_P = 0.8
SYNTHESIS_MAX_TOKENS = 1024

# Confidence assessment parameters
SYNTHESIS_BASE_CONFIDENCE = 0.5
SYNTHESIS_DOC_CONTEXT_BONUS = 0.2
SYNTHESIS_WEB_CONTEXT_BONUS = 0.15
SYNTHESIS_DOC_COUNT_BONUS = 0.1
SYNTHESIS_WEB_COUNT_BONUS = 0.05
SYNTHESIS_SHORT_ANSWER_PENALTY = 0.1
SYNTHESIS_UNCERTAINTY_PENALTY = 0.2
SYNTHESIS_MAX_DOC_BONUS = 0.2
SYNTHESIS_MAX_WEB_BONUS = 0.15
SYNTHESIS_SHORT_ANSWER_LENGTH = 100
SYNTHESIS_UNCERTAINTY_PHRASES = ["not available", "insufficient information", "cannot determine", "unclear"]

# Default error message for synthesis agent
SYNTHESIS_ERROR_MESSAGE = "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or try again."


# --- Web Search Agent ---
# Tavily API base URL
TAVILY_API_URL = "https://api.tavily.com"

# Maximum number of results to fetch from Tavily
TAVILY_MAX_RESULTS = 8

# Timeout for Tavily API requests in seconds
TAVILY_TIMEOUT = 30.0

# Default error message for web search agent when disabled
WEB_SEARCH_DISABLED_MESSAGE = "Web search is currently unavailable (API key not configured)."

# Default error message for web search agent on error
WEB_SEARCH_ERROR_MESSAGE = "An error occurred while searching the web. Please try again."
