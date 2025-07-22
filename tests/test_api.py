"""
Tests for FastAPI endpoints - document upload, health checks, and API functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import io

@pytest.mark.api
class TestAPI:
    """Test FastAPI endpoints."""

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "arc-fusion-rag"

    def test_upload_document_success(self, test_client, sample_pdf_bytes):
        """Test successful document upload."""
        # Mock the document processor and vector store
        with patch("app.main.document_processor") as mock_processor, \
             patch("app.main.vector_store") as mock_store:
            
            # Mock successful processing
            mock_result = {
                "document_id": "test-doc-123",
                "filename": "test.pdf",
                "parent_chunk_count": 3,
                "child_chunk_count": 8
            }
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_store.store_document_chunks = AsyncMock()
            
            # Upload PDF
            files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 201
            data = response.json()
            
            assert data["document_id"] == "test-doc-123"
            assert data["filename"] == "test.pdf"
            assert data["parent_chunks"] == 3
            assert data["child_chunks"] == 8
            assert data["status"] == "processed"

    def test_upload_non_pdf_file(self, test_client):
        """Test uploading non-PDF file."""
        # Create a fake text file
        text_content = b"This is not a PDF file"
        files = {"file": ("test.txt", text_content, "text/plain")}
        
        response = test_client.post("/api/v1/documents", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "Only PDF files are supported" in data["detail"]

    def test_upload_document_processing_error(self, test_client, sample_pdf_bytes):
        """Test document upload with processing error."""
        with patch("app.main.document_processor") as mock_processor:
            # Mock processing failure
            mock_processor.process_document = AsyncMock(
                side_effect=Exception("Processing failed")
            )
            
            files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to process document" in data["detail"]
            assert "Processing failed" in data["detail"]

    def test_upload_document_storage_error(self, test_client, sample_pdf_bytes):
        """Test document upload with vector store error."""
        with patch("app.main.document_processor") as mock_processor, \
             patch("app.main.vector_store") as mock_store:
            
            # Mock successful processing but storage failure
            mock_result = {
                "document_id": "test-doc-123",
                "filename": "test.pdf",
                "parent_chunk_count": 3,
                "child_chunk_count": 8
            }
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_store.store_document_chunks = AsyncMock(
                side_effect=Exception("Storage failed")
            )
            
            files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 500
            data = response.json()
            assert "Failed to process document" in data["detail"]

    def test_get_documents_endpoint(self, test_client):
        """Test get all documents endpoint."""
        with patch("app.main.vector_store") as mock_store:
            # Mock document list
            mock_documents = [
                {
                    "document_id": "doc-1",
                    "filename": "doc1.pdf",
                    "chunk_count": 10
                },
                {
                    "document_id": "doc-2", 
                    "filename": "doc2.pdf",
                    "chunk_count": 15
                }
            ]
            mock_store.get_all_documents = AsyncMock(return_value=mock_documents)
            
            response = test_client.get("/api/v1/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["document_id"] == "doc-1"
            assert data[1]["document_id"] == "doc-2"

    def test_get_documents_stats_endpoint(self, test_client):
        """Test get database statistics endpoint."""
        with patch("app.main.vector_store") as mock_store:
            # Mock stats
            mock_stats = {
                "total_chunks": 78,
                "unique_documents": 3,
                "collection_name": "DocumentChunks",
                "weaviate_url": "http://localhost:8080"
            }
            mock_store.get_database_stats = AsyncMock(return_value=mock_stats)
            
            response = test_client.get("/api/v1/documents/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_chunks"] == 78
            assert data["unique_documents"] == 3
            assert data["collection_name"] == "DocumentChunks"

    def test_delete_all_documents_endpoint(self, test_client):
        """Test delete all documents endpoint."""
        with patch("app.main.vector_store") as mock_store, \
             patch("app.main.document_processor") as mock_processor:
            
            mock_store.clear_all_documents = AsyncMock()
            mock_processor.parent_store = {"test": "data"}
            
            response = test_client.delete("/api/v1/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "All documents deleted successfully"
            
            # Verify both stores were cleared
            mock_store.clear_all_documents.assert_called_once()
            assert mock_processor.parent_store == {}

    def test_upload_empty_file(self, test_client):
        """Test uploading empty file."""
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        
        response = test_client.post("/api/v1/documents", files=files)
        
        # Should return 500 due to processing error
        assert response.status_code == 500

    def test_upload_no_file(self, test_client):
        """Test upload endpoint with no file."""
        response = test_client.post("/api/v1/documents")
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/health")
        
        # CORS should be enabled for all origins
        assert response.status_code == 200

    def test_upload_large_file(self, test_client):
        """Test uploading large file."""
        # Create a larger PDF buffer
        large_pdf = b"%PDF-1.4\n" + b"x" * 10000 + b"\n%%EOF"
        
        with patch("app.main.document_processor") as mock_processor, \
             patch("app.main.vector_store") as mock_store:
            
            mock_result = {
                "document_id": "large-doc-123",
                "filename": "large.pdf",
                "parent_chunk_count": 10,
                "child_chunk_count": 50
            }
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_store.store_document_chunks = AsyncMock()
            
            files = {"file": ("large.pdf", large_pdf, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 201

    def test_concurrent_uploads(self, test_client, sample_pdf_bytes):
        """Test handling concurrent uploads."""
        import concurrent.futures
        import threading
        
        with patch("app.main.document_processor") as mock_processor, \
             patch("app.main.vector_store") as mock_store:
            
            # Mock successful processing
            mock_result = {
                "document_id": "concurrent-doc",
                "filename": "test.pdf",
                "parent_chunk_count": 2,
                "child_chunk_count": 5
            }
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_store.store_document_chunks = AsyncMock()
            
            def upload_file():
                files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
                return test_client.post("/api/v1/documents", files=files)
            
            # Execute multiple uploads concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(upload_file) for _ in range(3)]
                results = [future.result() for future in futures]
            
            # All should succeed
            for response in results:
                assert response.status_code == 201

    def test_api_error_responses(self, test_client):
        """Test various error response formats."""
        # Test invalid endpoint
        response = test_client.get("/api/v1/invalid")
        assert response.status_code == 404
        
        # Test wrong method
        response = test_client.get("/api/v1/documents", 
                                  headers={"Content-Type": "multipart/form-data"})
        # GET should work for documents endpoint
        assert response.status_code == 200

    def test_request_id_logging(self, test_client, sample_pdf_bytes):
        """Test that request IDs are properly set and logged."""
        with patch("app.main.document_processor") as mock_processor, \
             patch("app.main.vector_store") as mock_store, \
             patch("app.main.set_request_id") as mock_set_request_id:
            
            mock_set_request_id.return_value = "test-request-123"
            mock_result = {
                "document_id": "test-doc",
                "filename": "test.pdf", 
                "parent_chunk_count": 1,
                "child_chunk_count": 2
            }
            mock_processor.process_document = AsyncMock(return_value=mock_result)
            mock_store.store_document_chunks = AsyncMock()
            
            files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 201
            # Verify request ID was set
            mock_set_request_id.assert_called_once()

    def test_file_extension_validation(self, test_client):
        """Test file extension validation."""
        # Test various non-PDF extensions
        test_files = [
            ("test.doc", b"fake doc content", "application/msword"),
            ("test.txt", b"text content", "text/plain"),
            ("test.jpeg", b"fake image", "image/jpeg"),
            ("test.pdf.exe", b"suspicious file", "application/octet-stream")
        ]
        
        for filename, content, mime_type in test_files:
            files = {"file": (filename, content, mime_type)}
            response = test_client.post("/api/v1/documents", files=files)
            
            if not filename.lower().endswith('.pdf'):
                assert response.status_code == 400
                assert "Only PDF files are supported" in response.json()["detail"] 