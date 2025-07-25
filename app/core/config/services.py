"""
Service-specific configuration classes for Arc-Fusion core services.

These Pydantic models provide centralized, typed, and validated configuration
for each service in the application.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class DocumentProcessingConfig(BaseSettings):
    """Configuration for document processing service."""
    
    # Document chunking parameters
    parent_chunk_size: int = 3000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 1000
    child_chunk_overlap: int = 100
    
    # PDF processing parameters
    pdf_extraction_dpi: int = 200
    pdf_extraction_format: str = "text"
    
    class Config:
        env_prefix = "DOC_PROC_"


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding service."""
    
    # Model configuration
    model: str = "gemini-embedding-001"
    
    # Rate limiting parameters
    max_retries: int = 5
    base_delay: float = 2.0
    request_delay: float = 4.0
    max_concurrent_requests: int = 1
    enable_rate_limiting: bool = False
    
    class Config:
        env_prefix = "EMBEDDING_"


class VectorStoreConfig(BaseSettings):
    """Configuration for vector store service."""
    
    # Connection parameters
    url: str = "http://localhost:8080"
    collection_name: str = "DocumentChunks"
    parent_collection_name: str = "ParentChunks"
    
    # Batch processing parameters
    batch_size: int = 50
    batch_delay: float = 0.1
    
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
    
    # Model parameters
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    enable_reranking: bool = True
    enable_quantization: bool = True
    
    # Processing parameters
    max_length: int = 8192
    batch_size_cpu: int = 8
    batch_size_gpu: int = 2
    
    class Config:
        env_prefix = "RERANKER_"
