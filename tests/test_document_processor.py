"""
Tests for DocumentProcessor - PDF processing, chunking, and embeddings.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.document_processor import DocumentProcessor

@pytest.mark.unit
class TestDocumentProcessor:
    """Test DocumentProcessor functionality."""

    def test_init(self, document_processor):
        """Test DocumentProcessor initialization."""
        assert document_processor.parent_chunk_size == 3000
        assert document_processor.parent_overlap == 200
        assert document_processor.child_chunk_size == 1000
        assert document_processor.child_overlap == 100
        assert document_processor.parent_store == {}

    def test_extract_text_from_pdf(self, document_processor, sample_pdf_bytes):
        """Test PDF text extraction."""
        text = document_processor._extract_text_from_pdf(sample_pdf_bytes)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Arc-Fusion Test Document" in text
        assert "Advanced Prompting Techniques" in text
        assert "text-to-SQL" in text

    @pytest.mark.asyncio
    async def test_process_document_success(self, document_processor, sample_pdf_bytes):
        """Test successful document processing."""
        filename = "test_document.pdf"
        
        result = await document_processor.process_document(sample_pdf_bytes, filename)
        
        # Validate result structure
        assert "document_id" in result
        assert "filename" in result
        assert "parent_chunk_count" in result
        assert "child_chunk_count" in result
        assert "child_chunks" in result
        
        # Validate values
        assert result["filename"] == filename
        assert result["parent_chunk_count"] > 0
        assert result["child_chunk_count"] > 0
        assert len(result["child_chunks"]) == result["child_chunk_count"]
        
        # Validate chunk structure
        for chunk in result["child_chunks"]:
            assert "id" in chunk
            assert "parent_id" in chunk
            assert "content" in chunk
            assert "embedding" in chunk
            assert "document_id" in chunk
            assert "filename" in chunk
            assert len(chunk["embedding"]) == 768  # Gemini embedding size

    @pytest.mark.asyncio
    async def test_process_document_chunking(self, document_processor, sample_pdf_bytes):
        """Test that document is properly chunked."""
        result = await document_processor.process_document(sample_pdf_bytes, "test.pdf")
        
        # Should create multiple chunks for our sample PDF
        assert result["parent_chunk_count"] >= 1
        assert result["child_chunk_count"] >= 2
        
        # Verify parent chunks are stored in memory
        assert len(document_processor.parent_store) == result["parent_chunk_count"]
        
        # Verify all child chunks have valid parent IDs
        parent_ids = set(document_processor.parent_store.keys())
        for chunk in result["child_chunks"]:
            assert chunk["parent_id"] in parent_ids

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, document_processor):
        """Test successful embedding generation."""
        test_text = "This is a test text for embedding."
        
        embedding = await document_processor._generate_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_rate_limit_retry(self, slow_gemini_embeddings):
        """Test rate limit handling with retry logic."""
        from app.core.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        test_text = "Test text for rate limiting."
        
        # Should succeed after retry despite initial rate limit error
        embedding = await processor._generate_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_generate_embedding_failure(self, document_processor):
        """Test embedding generation failure handling."""
        with patch.object(document_processor.embeddings, 'aembed_query', 
                         side_effect=Exception("API Error")) as mock_embed:
            
            test_text = "This should fail."
            
            with pytest.raises(Exception) as exc_info:
                await document_processor._generate_embedding(test_text)
            
            assert "Failed to generate embedding" in str(exc_info.value)

    def test_get_parent_chunk(self, document_processor):
        """Test retrieval of parent chunk by ID."""
        # Store a test parent chunk
        parent_id = "test-parent-123"
        parent_content = "This is a test parent chunk content."
        document_processor.parent_store[parent_id] = parent_content
        
        # Retrieve it
        retrieved = document_processor.get_parent_chunk(parent_id)
        assert retrieved == parent_content
        
        # Test non-existent parent
        non_existent = document_processor.get_parent_chunk("non-existent")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_process_empty_pdf(self, document_processor):
        """Test processing of empty or invalid PDF."""
        empty_bytes = b""
        
        with pytest.raises(Exception):
            await document_processor.process_document(empty_bytes, "empty.pdf")

    @pytest.mark.asyncio
    async def test_process_document_logging(self, document_processor, sample_pdf_bytes, mock_logger):
        """Test that document processing logs appropriately."""
        with patch("app.core.document_processor.logger", mock_logger):
            await document_processor.process_document(sample_pdf_bytes, "test.pdf")
        
        # Verify logging calls were made
        assert mock_logger.info.called
        assert mock_logger.debug.called
        
        # Check specific log messages
        log_calls = [call.args for call in mock_logger.info.call_args_list]
        log_messages = [call[0] if call else "" for call in log_calls]
        
        assert any("Starting document processing" in msg for msg in log_messages)
        assert any("PDF text extracted" in msg for msg in log_messages) 
        assert any("Parent chunks created" in msg for msg in log_messages)
        assert any("Document processing completed" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_process_large_document(self, mock_gemini_embeddings):
        """Test processing of a document that creates many chunks."""
        # Create large text content
        large_text = "Large text content. " * 1000  # Create ~20KB of text
        
        with patch("app.core.document_processor.DocumentProcessor._extract_text_from_pdf", 
                   return_value=large_text):
            processor = DocumentProcessor()
            result = await processor.process_document(b"fake_pdf_bytes", "large.pdf")
            
            # Should create multiple parent and child chunks
            assert result["parent_chunk_count"] > 1
            assert result["child_chunk_count"] > result["parent_chunk_count"]
            
            # Verify all embeddings were generated (one per child chunk)
            assert mock_gemini_embeddings.aembed_query.call_count == result["child_chunk_count"]

    def test_chunk_size_configuration(self, mock_gemini_embeddings):
        """Test that chunk sizes can be configured."""
        # Create processor with custom chunk sizes
        processor = DocumentProcessor()
        processor.parent_chunk_size = 500
        processor.child_chunk_size = 200
        
        assert processor.parent_chunk_size == 500
        assert processor.child_chunk_size == 200 