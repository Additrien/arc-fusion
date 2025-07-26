"""
Handles PDF processing with parent-child chunking strategy.
"""
import os
import uuid
from typing import Dict, List, Any, Optional
import io

from PyPDF2 import PdfReader
from google import genai
from google.genai import types
import asyncio
import time

from app.core.document.pdf_extractor import PDFExtractor
from app.core.document.chunking_service import ChunkingService, ChunkingResult
from app.core.embeddings.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.config.services import DocumentProcessingConfig
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.document_processor')


class DocumentProcessor:
    """Handles PDF processing with parent-child chunking strategy."""
    
    def __init__(self, 
                 pdf_extractor: PDFExtractor,
                 chunking_service: ChunkingService,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStore,
                 config: DocumentProcessingConfig):
        """
        Initialize the document processor with injected dependencies.
        
        Args:
            pdf_extractor: Service for extracting text from PDFs
            chunking_service: Service for chunking text content
            embedding_service: Service for generating embeddings
            vector_store: Service for storing chunks in vector database
            config: Document processing configuration
        """
        self.pdf_extractor = pdf_extractor
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.config = config
        
        # Parent chunk store (in-memory for rapid context retrieval)
        self.parent_store: Dict[str, str] = {}
    
    async def process_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF document with parent-child chunking strategy."""
        document_id = str(uuid.uuid4())
        logger.info("Starting document processing", extra={
            "document_filename": filename,
            "document_id": document_id,
            "file_size": len(content)
        })
        
        # Extract text from PDF using the PDFExtractor service
        text = self.pdf_extractor.extract_text(content)
        logger.info("PDF text extracted", extra={
            "text_length": len(text),
            "document_id": document_id
        })
        
        # Create chunks using the ChunkingService
        chunking_result = self.chunking_service.create_chunks(text, document_id, filename)
        logger.info("Chunks created", extra={
            "parent_chunk_count": len(chunking_result.parent_chunks),
            "child_chunk_count": len(chunking_result.child_chunks),
            "document_id": document_id
        })
        
        # Generate embeddings for child chunks using the EmbeddingService
        child_texts = [chunk["content"] for chunk in chunking_result.child_chunks]
        embeddings = await self.embedding_service.generate_embeddings(child_texts)
        
        # Add embeddings to child chunks
        child_chunks_data = []
        for i, chunk in enumerate(chunking_result.child_chunks):
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding["embedding"] = embeddings[i]
            child_chunks_data.append(chunk_with_embedding)
            
            # Store parent chunk in memory
            parent_id = chunk["parent_id"]
            if parent_id not in self.parent_store:
                # Find the parent chunk
                for parent_chunk in chunking_result.parent_chunks:
                    if parent_chunk["id"] == parent_id:
                        self.parent_store[parent_id] = parent_chunk["content"]
                        break
        
        logger.info("Document processing completed", extra={
            "document_id": document_id,
            "document_filename": filename,
            "parent_chunks": len(chunking_result.parent_chunks),
            "child_chunks": len(child_chunks_data),
            "total_embedding_calls": len(child_chunks_data)
        })
        
        return {
            "document_id": document_id,
            "filename": filename,
            "parent_chunk_count": len(chunking_result.parent_chunks),
            "child_chunk_count": len(child_chunks_data),
            "child_chunks": child_chunks_data,
            "parent_chunks": chunking_result.parent_chunks
        }

    
    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        """Retrieve parent chunk by ID."""
        return self.parent_store.get(parent_id)
    
    async def clear_parent_store(self):
        """Clear all parent chunks from memory."""
        self.parent_store.clear()
