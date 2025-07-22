"""
Integration tests for the complete document processing pipeline.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio

@pytest.mark.integration  
class TestIntegration:
    """Test complete document processing pipeline."""

    @pytest.mark.asyncio
    async def test_complete_document_processing_pipeline(
        self, 
        document_processor, 
        vector_store, 
        sample_pdf_bytes,
        mock_logger
    ):
        """Test the complete pipeline from PDF to Weaviate storage."""
        # Process document
        result = await document_processor.process_document(
            sample_pdf_bytes, 
            "integration_test.pdf"
        )
        
        # Verify processing results
        assert result["filename"] == "integration_test.pdf"
        assert result["parent_chunk_count"] > 0
        assert result["child_chunk_count"] > 0
        assert len(result["child_chunks"]) == result["child_chunk_count"]
        
        # Store in vector database
        await vector_store.store_document_chunks(result)
        
        # Verify storage was attempted
        assert vector_store.client.collections.get.called
        collection_mock = vector_store.client.collections.get.return_value
        assert collection_mock.data.insert_many.called

    @pytest.mark.asyncio
    async def test_end_to_end_api_processing(self, test_client, sample_pdf_bytes):
        """Test complete API processing flow with real components."""
        with patch("app.core.document_processor.GoogleGenerativeAIEmbeddings") as mock_embeddings, \
             patch("app.core.vector_store.weaviate") as mock_weaviate:
            
            # Setup mocks
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embeddings.return_value = mock_embedding_instance
            
            mock_client = MagicMock()
            mock_collections = MagicMock()
            mock_collection = MagicMock()
            
            mock_collection.data.insert_many = MagicMock()
            mock_collections.get = MagicMock(return_value=mock_collection)
            mock_collections.create = MagicMock()
            mock_collections.list_all = MagicMock(return_value=[])
            
            mock_client.collections = mock_collections
            mock_client.is_ready = MagicMock(return_value=True)
            mock_weaviate.connect_to_local = MagicMock(return_value=mock_client)
            
            # Test upload
            files = {"file": ("integration.pdf", sample_pdf_bytes, "application/pdf")}
            response = test_client.post("/api/v1/documents", files=files)
            
            assert response.status_code == 201
            data = response.json()
            
            # Verify response structure
            assert "document_id" in data
            assert "filename" in data
            assert "parent_chunks" in data
            assert "child_chunks" in data
            assert data["status"] == "processed"
            
            # Verify embeddings were generated
            assert mock_embedding_instance.aembed_query.called
            
            # Verify data was stored
            assert mock_collection.data.insert_many.called

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, document_processor, vector_store):
        """Test error handling throughout the pipeline."""
        # Test with invalid PDF content
        with pytest.raises(Exception):
            await document_processor.process_document(b"invalid pdf", "bad.pdf")

    @pytest.mark.asyncio
    async def test_pipeline_with_rate_limiting(self, sample_pdf_bytes):
        """Test pipeline behavior with rate limiting."""
        with patch("app.core.document_processor.GoogleGenerativeAIEmbeddings") as mock_embeddings, \
             patch("app.core.vector_store.weaviate") as mock_weaviate:
            
            # Setup rate limiting behavior
            call_count = 0
            async def rate_limited_embedding(text):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:  # First 2 calls fail
                    raise Exception("429 Rate limit exceeded")
                return [0.1] * 768
            
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.aembed_query = rate_limited_embedding
            mock_embeddings.return_value = mock_embedding_instance
            
            # Setup Weaviate mock
            mock_client = MagicMock()
            mock_collections = MagicMock()
            mock_collection = MagicMock()
            
            mock_collection.data.insert_many = MagicMock()
            mock_collections.get = MagicMock(return_value=mock_collection)
            mock_collections.create = MagicMock()
            mock_collections.list_all = MagicMock(return_value=[])
            
            mock_client.collections = mock_collections
            mock_weaviate.connect_to_local = MagicMock(return_value=mock_client)
            
            from app.core.document_processor import DocumentProcessor
            from app.core.vector_store import VectorStore
            
            processor = DocumentProcessor()
            store = VectorStore()
            
            # Process should succeed despite rate limiting
            result = await processor.process_document(sample_pdf_bytes, "rate_test.pdf")
            await store.store_document_chunks(result)
            
            # Verify processing completed
            assert result["child_chunk_count"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, sample_pdf_bytes):
        """Test processing multiple documents concurrently."""
        with patch("app.core.document_processor.GoogleGenerativeAIEmbeddings") as mock_embeddings, \
             patch("app.core.vector_store.weaviate") as mock_weaviate:
            
            # Setup mocks
            mock_embedding_instance = AsyncMock()
            mock_embedding_instance.aembed_query = AsyncMock(return_value=[0.1] * 768)
            mock_embeddings.return_value = mock_embedding_instance
            
            mock_client = MagicMock()
            mock_collections = MagicMock()
            mock_collection = MagicMock()
            
            mock_collection.data.insert_many = MagicMock()
            mock_collections.get = MagicMock(return_value=mock_collection)
            mock_collections.create = MagicMock()
            mock_collections.list_all = MagicMock(return_value=[])
            
            mock_client.collections = mock_collections
            mock_weaviate.connect_to_local = MagicMock(return_value=mock_client)
            
            from app.core.document_processor import DocumentProcessor
            from app.core.vector_store import VectorStore
            
            # Create multiple processors and stores
            processors = [DocumentProcessor() for _ in range(3)]
            stores = [VectorStore() for _ in range(3)]
            
            # Process multiple documents concurrently
            async def process_doc(processor, store, doc_name):
                result = await processor.process_document(sample_pdf_bytes, doc_name)
                await store.store_document_chunks(result)
                return result
            
            tasks = [
                process_doc(processors[i], stores[i], f"doc_{i}.pdf")
                for i in range(3)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == 3
            for result in results:
                assert result["child_chunk_count"] > 0

    @pytest.mark.asyncio
    async def test_search_functionality(self, vector_store):
        """Test document search after storage."""
        query = "test search query"
        query_embedding = [0.1] * 768
        
        # Mock search results
        mock_result = MagicMock()
        mock_result.objects = [
            MagicMock(
                properties={
                    "content": "This is a test chunk about search functionality.",
                    "document_id": "search-test-doc",
                    "parent_id": "parent-123"
                },
                uuid="chunk-456"
            )
        ]
        
        collection_mock = vector_store.client.collections.get.return_value
        collection_mock.query.hybrid.return_value = mock_result
        
        # Perform search
        results = await vector_store.hybrid_search(query, query_embedding)
        
        # Verify search results
        assert isinstance(results, list)
        assert len(results) >= 0
        
        # Verify search was performed
        collection_mock.query.hybrid.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_management(self, document_processor, sample_pdf_bytes):
        """Test memory management during processing."""
        # Process a document
        result = await document_processor.process_document(sample_pdf_bytes, "memory_test.pdf")
        
        # Verify parent chunks are stored in memory
        assert len(document_processor.parent_store) == result["parent_chunk_count"]
        
        # Verify parent chunks can be retrieved
        for chunk in result["child_chunks"]:
            parent_content = document_processor.get_parent_chunk(chunk["parent_id"])
            assert parent_content is not None
            assert isinstance(parent_content, str)

    def test_logging_integration(self, mock_logger):
        """Test that logging works throughout the pipeline."""
        with patch("app.utils.logger.get_logger", return_value=mock_logger):
            from app.utils.logger import get_logger
            
            # Test logger creation
            test_logger = get_logger("test_logger")
            assert test_logger == mock_logger
            
            # Test logging calls
            test_logger.info("Test message")
            mock_logger.info.assert_called_with("Test message")

    @pytest.mark.asyncio
    async def test_cleanup_operations(self, document_processor, vector_store):
        """Test cleanup and resource management."""
        # Add some test data
        document_processor.parent_store["test-parent"] = "test content"
        
        # Clear vector store
        collection_mock = vector_store.client.collections.get.return_value
        collection_mock.data.delete_many.return_value = MagicMock(successful=1, failed=0)
        
        await vector_store.clear_all_documents()
        
        # Clear parent store
        document_processor.parent_store.clear()
        
        # Verify cleanup
        assert len(document_processor.parent_store) == 0
        collection_mock.data.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test that configuration is properly validated."""
        from app.core.document_processor import DocumentProcessor
        from app.core.vector_store import VectorStore
        
        # Test default configuration
        processor = DocumentProcessor()
        assert processor.parent_chunk_size > 0
        assert processor.child_chunk_size > 0
        assert processor.parent_overlap >= 0
        assert processor.child_overlap >= 0
        
        store = VectorStore()
        assert store.collection_name == "DocumentChunks"
        assert store.weaviate_url.startswith("http")

    @pytest.mark.asyncio
    async def test_data_consistency(self, document_processor, sample_pdf_bytes):
        """Test data consistency throughout the pipeline."""
        result = await document_processor.process_document(sample_pdf_bytes, "consistency_test.pdf")
        
        # Verify data consistency
        assert result["child_chunk_count"] == len(result["child_chunks"])
        
        # Verify each chunk has required fields
        for chunk in result["child_chunks"]:
            assert chunk["document_id"] == result["document_id"]
            assert chunk["filename"] == result["filename"]
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == 768  # Gemini embedding size
            
        # Verify parent-child relationships
        parent_ids = set(document_processor.parent_store.keys())
        for chunk in result["child_chunks"]:
            assert chunk["parent_id"] in parent_ids 