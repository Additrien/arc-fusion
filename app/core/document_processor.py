import io
import uuid
import asyncio
import time
from typing import Dict, List, Any, Optional
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

from app.utils.logger import get_logger

logger = get_logger('arc_fusion.document_processor')

class DocumentProcessor:
    """Handles PDF processing with parent-child chunking strategy."""
    
    def __init__(self):
        self.parent_chunk_size = 3000  # Plus gros pour moins de chunks
        self.parent_overlap = 200
        self.child_chunk_size = 1000   # Plus gros pour moins d'appels API
        self.child_overlap = 100
        
        # Parent chunk store (in-memory for rapid context retrieval)
        self.parent_store: Dict[str, str] = {}
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
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
                embedding = await self._generate_embedding(child_chunk)
                
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
        
        return {
            "document_id": document_id,
            "filename": filename,
            "parent_chunk_count": len(parent_chunks),
            "child_chunk_count": len(child_chunks_data),
            "child_chunks": child_chunks_data
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
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini embedding model with rate limiting."""
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        logger.debug("Starting embedding generation", extra={
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "text_length": len(text)
        })
        
        for attempt in range(max_retries):
            try:
                # Add delay between requests to respect rate limits (100 RPM = 0.6s between requests)
                if attempt > 0:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.info("Retrying with exponential backoff", extra={
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "max_retries": max_retries
                    })
                    await asyncio.sleep(delay)
                else:
                    logger.debug("Applying base rate limiting delay (0.6s)")
                    await asyncio.sleep(0.6)  # Base delay for rate limiting
                
                embedding = await self.embeddings.aembed_query(text)
                
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
                
                # Check if it's a rate limit error
                if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning("Rate limit hit, will retry", extra={
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
                raise Exception(f"Failed to generate embedding: {str(e)}")
        
        logger.error("Max retries exceeded for embedding generation")
        raise Exception("Max retries exceeded for embedding generation")
    
    def get_parent_chunk(self, parent_id: str) -> Optional[str]:
        """Retrieve parent chunk by ID."""
        return self.parent_store.get(parent_id)
    
    async def clear_parent_store(self):
        """Clear all parent chunks from memory."""
        self.parent_store.clear() 