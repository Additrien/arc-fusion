"""
Chunking service for Arc-Fusion.

This module handles the chunking of text content into parent and child chunks,
separating this responsibility from the main document processor.
"""

import uuid
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config.services import DocumentProcessingConfig
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.document.chunking_service')


class ChunkingResult:
    """Container for chunking results."""
    
    def __init__(self, parent_chunks: List[Dict[str, Any]], child_chunks: List[Dict[str, Any]]):
        self.parent_chunks = parent_chunks
        self.child_chunks = child_chunks


class ChunkingService:
    """Service for chunking text content into parent and child chunks."""
    
    def __init__(self, config: DocumentProcessingConfig):
        """
        Initialize the chunking service.
        
        Args:
            config: Document processing configuration
        """
        self.config = config
        
        # Initialize text splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.child_chunk_size,
            chunk_overlap=config.child_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def create_chunks(self, text: str, document_id: str, filename: str) -> ChunkingResult:
        """
        Create parent and child chunks from text content.
        
        Args:
            text: Text content to chunk
            document_id: Unique identifier for the document
            filename: Original filename
            
        Returns:
            ChunkingResult containing parent and child chunks
        """
        logger.info(f"Creating chunks for document {document_id}")
        
        # Create parent chunks
        parent_texts = self.parent_splitter.split_text(text)
        logger.info(f"Created {len(parent_texts)} parent chunks")
        
        # Create parent chunk objects
        parent_chunks = []
        for i, parent_text in enumerate(parent_texts):
            parent_chunk = {
                "id": str(uuid.uuid4()),
                "content": parent_text,
                "document_id": document_id,
                "filename": filename,
                "parent_index": i
            }
            parent_chunks.append(parent_chunk)
        
        # Create child chunks from parent chunks
        child_chunks = []
        for parent_chunk in parent_chunks:
            parent_id = parent_chunk["id"]
            parent_text = parent_chunk["content"]
            parent_index = parent_chunk["parent_index"]
            
            # Split parent into child chunks
            child_texts = self.child_splitter.split_text(parent_text)
            
            for j, child_text in enumerate(child_texts):
                child_chunk = {
                    "id": str(uuid.uuid4()),
                    "content": child_text,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "filename": filename,
                    "parent_index": parent_index,
                    "child_index": j
                }
                child_chunks.append(child_chunk)
        
        logger.info(f"Created {len(child_chunks)} child chunks from {len(parent_chunks)} parents")
        
        return ChunkingResult(parent_chunks=parent_chunks, child_chunks=child_chunks)
    
    def get_chunk_stats(self, text: str) -> Dict[str, int]:
        """
        Get statistics about how text would be chunked.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Dictionary with chunking statistics
        """
        parent_texts = self.parent_splitter.split_text(text)
        total_child_chunks = 0
        
        for parent_text in parent_texts:
            child_texts = self.child_splitter.split_text(parent_text)
            total_child_chunks += len(child_texts)
        
        return {
            "parent_chunks": len(parent_texts),
            "child_chunks": total_child_chunks,
            "avg_parent_length": len(text) // len(parent_texts) if parent_texts else 0,
            "avg_child_length": len(text) // total_child_chunks if total_child_chunks > 0 else 0
        }
