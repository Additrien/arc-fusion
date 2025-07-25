"""
Service factory for Arc-Fusion.

This module provides a centralized factory for creating and managing
service instances with proper dependency injection.
"""

import os
from functools import lru_cache
from typing import Optional
from app.core.config.services import (
    DocumentProcessingConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    SessionConfig,
    RerankerConfig
)
from app.core.document.pdf_extractor import PDFExtractor
from app.core.document.chunking_service import ChunkingService
from app.core.embeddings.embedding_service import EmbeddingService
from app.core.session.session_manager import SessionManager
from app.core.vector_store import VectorStore
from app.core.document_processor import DocumentProcessor
from app.core.agent_service import AgentService
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.core.factories')


class ServiceFactory:
    """Factory for creating and managing service instances."""
    
    def __init__(self):
        """Initialize the service factory with configurations."""
        logger.info("Initializing service factory")
        
        # Load all configurations
        self.doc_config = DocumentProcessingConfig()
        self.embedding_config = EmbeddingConfig()
        self.vector_config = VectorStoreConfig()
        self.session_config = SessionConfig()
        self.reranker_config = RerankerConfig()
        
        logger.info("Service factory initialized with configurations")
    
    def create_pdf_extractor(self) -> PDFExtractor:
        """
        Create a PDF extractor instance.
        
        Returns:
            PDFExtractor instance
        """
        logger.debug("Creating PDF extractor")
        return PDFExtractor(format=self.doc_config.pdf_extraction_format)
    
    def create_chunking_service(self) -> ChunkingService:
        """
        Create a chunking service instance.
        
        Returns:
            ChunkingService instance
        """
        logger.debug("Creating chunking service")
        return ChunkingService(config=self.doc_config)
    
    def create_embedding_service(self) -> EmbeddingService:
        """
        Create an embedding service instance.
        
        Returns:
            EmbeddingService instance
        """
        logger.debug("Creating embedding service")
        return EmbeddingService(config=self.embedding_config)
    
    def create_vector_store(self) -> VectorStore:
        """
        Create a vector store instance.
        
        Returns:
            VectorStore instance
        """
        logger.debug("Creating vector store")
        return VectorStore()
    
    def create_session_manager(self) -> SessionManager:
        """
        Create a session manager instance.
        
        Returns:
            SessionManager instance
        """
        logger.debug("Creating session manager")
        return SessionManager(config=self.session_config)
    
    def create_document_processor(self) -> DocumentProcessor:
        """
        Create a document processor instance with injected dependencies.
        
        Returns:
            DocumentProcessor instance
        """
        logger.debug("Creating document processor with dependencies")
        
        pdf_extractor = self.create_pdf_extractor()
        chunking_service = self.create_chunking_service()
        embedding_service = self.create_embedding_service()
        vector_store = self.create_vector_store()
        
        return DocumentProcessor(
            pdf_extractor=pdf_extractor,
            chunking_service=chunking_service,
            embedding_service=embedding_service,
            vector_store=vector_store,
            config=self.doc_config
        )
    
    def create_agent_service(self) -> AgentService:
        """
        Create an agent service instance with injected dependencies.
        
        Returns:
            AgentService instance
        """
        logger.debug("Creating agent service with dependencies")
        
        session_manager = self.create_session_manager()
        return AgentService(session_manager=session_manager)


@lru_cache()
def get_service_factory() -> ServiceFactory:
    """
    Get a cached instance of the service factory.
    
    Returns:
        ServiceFactory instance
    """
    logger.info("Creating cached service factory instance")
    return ServiceFactory()


# FastAPI dependency functions
def get_document_processor(
    factory: ServiceFactory = None
) -> DocumentProcessor:
    """
    FastAPI dependency for document processor.
    
    Args:
        factory: Service factory instance (injected by FastAPI)
        
    Returns:
        DocumentProcessor instance
    """
    if factory is None:
        factory = get_service_factory()
    return factory.create_document_processor()


def get_vector_store(
    factory: ServiceFactory = None
) -> VectorStore:
    """
    FastAPI dependency for vector store.
    
    Args:
        factory: Service factory instance (injected by FastAPI)
        
    Returns:
        VectorStore instance
    """
    if factory is None:
        factory = get_service_factory()
    return factory.create_vector_store()


def get_agent_service(
    factory: ServiceFactory = None
) -> AgentService:
    """
    FastAPI dependency for agent service.
    
    Args:
        factory: Service factory instance (injected by FastAPI)
        
    Returns:
        AgentService instance
    """
    if factory is None:
        factory = get_service_factory()
    return factory.create_agent_service()
