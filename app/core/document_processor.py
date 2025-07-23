import io
import uuid
import asyncio
import time
from typing import Dict, List, Any, Optional
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from google import genai
from google.genai import types

from app.utils.logger import get_logger

logger = get_logger('arc_fusion.document_processor')

class DocumentProcessor:
    """Handles PDF processing with parent-child chunking strategy."""
    
    def __init__(self):
        # Use config values for chunk sizes
        from app.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
        self.parent_chunk_size = PARENT_CHUNK_SIZE
        self.parent_overlap = PARENT_CHUNK_OVERLAP
        self.child_chunk_size = CHILD_CHUNK_SIZE
        self.child_overlap = CHILD_CHUNK_OVERLAP
        
        # Parent chunk store (in-memory for rapid context retrieval)
        self.parent_store: Dict[str, str] = {}
        
        # Initialize Gemini client for embeddings
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        # Use the same embedding model as defined in config for consistency
        from app.config import EMBEDDING_MODEL
        self.embedding_model = EMBEDDING_MODEL
        
        # Rate limiting for API calls
        from app.config import ENABLE_RATE_LIMITING, EMBEDDING_REQUEST_DELAY
        self.rate_limiting_enabled = ENABLE_RATE_LIMITING
        self.request_delay = EMBEDDING_REQUEST_DELAY
        self.last_request_time = 0
        
        # Text splitters
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    async def process_document(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF document with parent-child chunking strategy."""
        document_id = str(uuid.uuid4())
        logger.info("Starting document processing", extra={
            "document_filename": filename,
            "document_id": document_id,
            "file_size": len(content)
        })
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(content)
        logger.info("PDF text extracted", extra={
            "text_length": len(text),
            "document_id": document_id
        })
        
        # Create parent chunks
        parent_chunks = self.parent_splitter.split_text(text)
        logger.info("Parent chunks created", extra={
            "parent_chunk_count": len(parent_chunks),
            "document_id": document_id
        })
        
        # Process each parent chunk
        child_chunks_data = []
        total_child_chunks = 0
        
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            parent_id = str(uuid.uuid4())
            
            # Store parent chunk in memory
            self.parent_store[parent_id] = parent_chunk
            
            # Create child chunks from parent
            child_chunks = self.child_splitter.split_text(parent_chunk)
            total_child_chunks += len(child_chunks)
            
            logger.info("Processing parent chunk", extra={
                "parent_index": parent_idx,
                "child_chunks_count": len(child_chunks),
                "parent_chunk_length": len(parent_chunk),
                "document_id": document_id
            })
            
            for child_idx, child_chunk in enumerate(child_chunks):
                child_id = str(uuid.uuid4())
                
                logger.debug("Generating embedding for child chunk", extra={
                    "parent_index": parent_idx,
                    "child_index": child_idx,
                    "chunk_length": len(child_chunk),
                    "document_id": document_id
                })
                
                # Generate embedding for child chunk
                embedding = await self._generate_embedding_with_retry(child_chunk)
                
                child_chunks_data.append({
                    "id": child_id,
                    "parent_id": parent_id,
                    "content": child_chunk,
                    "embedding": embedding,
                    "document_id": document_id,
                    "filename": filename,
                    "parent_index": parent_idx,
                    "child_index": child_idx
                })
        
        logger.info("Document processing completed", extra={
            "document_id": document_id,
            "document_filename": filename,
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks_data),
            "total_embedding_calls": len(child_chunks_data)
        })
        
        # Collect parent chunks data for storage
        parent_chunks_data = []
        for parent_idx, parent_chunk in enumerate(parent_chunks):
            # Find the parent_id for this parent chunk by looking at the first child chunk
            parent_id = None
            for child_data in child_chunks_data:
                if child_data["parent_index"] == parent_idx:
                    parent_id = child_data["parent_id"]
                    break
            
            if parent_id:
                parent_chunks_data.append({
                    "id": parent_id,
                    "content": parent_chunk,
                    "document_id": document_id,
                    "filename": filename,
                    "parent_index": parent_idx
                })
        
        return {
            "document_id": document_id,
            "filename": filename,
            "parent_chunk_count": len(parent_chunks),
            "child_chunk_count": len(child_chunks_data),
            "child_chunks": child_chunks_data,
            "parent_chunks": parent_chunks_data  # Add parent chunks data
        }
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text content from PDF bytes."""
        pdf_file = io.BytesIO(content)
        pdf_reader = PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
    
    async def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """Generate embedding with retry logic and rate limiting."""
        from app.config import EMBEDDING_MAX_RETRIES, EMBEDDING_RETRY_BASE_DELAY
        max_retries = EMBEDDING_MAX_RETRIES
        base_delay = EMBEDDING_RETRY_BASE_DELAY
        
        for attempt in range(max_retries):
            try:
                # Rate limiting: ensure we don't exceed API limits
                if self.rate_limiting_enabled:
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    
                    if time_since_last < self.request_delay:
                        sleep_time = self.request_delay - time_since_last
                        logger.info(f"Rate limiting: sleeping {sleep_time:.1f}s before embedding request")
                        await asyncio.sleep(sleep_time)
                    
                    self.last_request_time = time.time()
                
                logger.debug(f"Generating embedding (attempt {attempt + 1}/{max_retries})")
                
                response = await self.client.aio.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=768  # Fix dimension for consistency
                    )
                )
                
                # Extract embedding values
                if hasattr(response, 'embeddings') and response.embeddings:
                    embedding = list(response.embeddings[0].values)
                elif hasattr(response, 'embedding'):
                    embedding = list(response.embedding.values)
                else:
                    raise ValueError("Unexpected embedding response structure")
                
                logger.debug("Embedding generated successfully", extra={
                    "embedding_dimensions": len(embedding),
                    "attempt": attempt + 1
                })
                return embedding
                
            except Exception as e:
                error_msg = str(e).lower()
                
                logger.warning("Embedding generation failed", extra={
                    "attempt": attempt + 1,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
                # Check if it's a rate limit or quota error
                if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                    if attempt < max_retries - 1:
                        # Exponential backoff with longer delays for quota issues
                        delay = base_delay * (2 ** attempt)
                        if "quota" in error_msg:
                            delay = max(delay, 60)  # At least 1 minute for quota issues
                        
                        logger.warning("API rate/quota limit hit, will retry", extra={
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "retry_delay": delay,
                            "error": str(e)
                        })
                        await asyncio.sleep(delay)
                        continue
                
                # For other errors or max retries reached, raise
                logger.error("Failed to generate embedding", extra={
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "final_error": str(e)
                })
                raise Exception(f"Failed to generate embedding after {max_retries} attempts: {str(e)}")
        
        logger.error("Max retries exceeded for embedding generation")
        raise Exception("Max retries exceeded for embedding generation")
    
    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        """Retrieve parent chunk by ID."""
        return self.parent_store.get(parent_id)
    
    async def clear_parent_store(self):
        """Clear all parent chunks from memory."""
        self.parent_store.clear() 