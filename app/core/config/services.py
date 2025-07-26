"""
Service-specific configuration classes for Arc-Fusion core services.

These Pydantic models provide centralized, typed, and validated configuration
for each service in the application.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from app import config


class DocumentProcessingConfig(BaseSettings):
    """Configuration for document processing service."""
    
    # Document chunking parameters - imported from main config
    parent_chunk_size: int = config.PARENT_CHUNK_SIZE
    parent_chunk_overlap: int = config.PARENT_CHUNK_OVERLAP
    child_chunk_size: int = config.CHILD_CHUNK_SIZE
    child_chunk_overlap: int = config.CHILD_CHUNK_OVERLAP
    
    # PDF processing parameters
    pdf_extraction_dpi: int = 200
    pdf_extraction_format: str = "text"
    
    class Config:
        env_prefix = "DOC_PROC_"


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding service."""
    
    # Model configuration - imported from main config
    model: str = config.EMBEDDING_MODEL
    
    # Rate limiting parameters - imported from main config
    max_retries: int = config.EMBEDDING_MAX_RETRIES
    base_delay: float = config.EMBEDDING_RETRY_BASE_DELAY
    request_delay: float = config.EMBEDDING_REQUEST_DELAY
    max_concurrent_requests: int = config.MAX_CONCURRENT_EMBEDDINGS
    enable_rate_limiting: bool = config.ENABLE_RATE_LIMITING
    
    class Config:
        env_prefix = "EMBEDDING_"


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store service."""
    
    # Connection parameters
    url: str = "http://localhost:8080"
    collection_name: str = "DocumentChunks"
    parent_collection_name: str = "ParentChunks"
    
    # Batch processing parameters - imported from main config
    batch_size: int = config.WEAVIATE_BATCH_SIZE
    batch_delay: float = config.BATCH_DELAY_SECONDS
    
    # Chunk filtering parameters - imported from main config
    min_chunk_score: float = config.MIN_CHUNK_SCORE
    
    class Config:
        env_prefix = "VECTOR_"


class SessionConfig(BaseSettings):
    """Configuration for session management."""
    
    # Session parameters
    max_history_length: int = 10
    session_timeout: int = 3600  # 1 hour in seconds
    
    class Config:
        env_prefix = "SESSION_"


class RerankerConfig(BaseSettings):
    """Configuration for reranker service."""
    
    # Model parameters - imported from main config
    model_name: str = config.RERANKER_MODEL
    enable_reranking: bool = config.ENABLE_RERANKING
    enable_quantization: bool = config.ENABLE_MODEL_QUANTIZATION
    
    # Processing parameters
    max_length: int = 8192
    batch_size_cpu: int = 8
    batch_size_gpu: int = 2
    
    class Config:
        env_prefix = "RERANKER_"
