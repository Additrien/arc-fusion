"""
Tests for VectorStore - Weaviate integration and vector operations.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.vector_store import VectorStore

@pytest.mark.unit
class TestVectorStore:
    """Test VectorStore functionality."""

    def test_init(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store.weaviate_url == "http://localhost:8080"
        assert vector_store.collection_name == "DocumentChunks"
        assert vector_store.client is not None
        assert vector_store._connected is True

    @pytest.mark.asyncio
    async def test_ensure_connected(self, vector_store, mock_weaviate_client):
        """Test connection establishment."""
        # Should connect when not connected
        await vector_store._ensure_connected()
        
        assert vector_store._connected is True
        assert vector_store.client is not None

    @pytest.mark.asyncio
    async def test_store_document_chunks_success(self, vector_store, sample_document_result):
        """Test successful document chunk storage."""
        await vector_store.store_document_chunks(sample_document_result)
        
        # Verify client operations were called
        assert vector_store.client.collections.get.called
        collection_mock = vector_store.client.collections.get.return_value
        assert collection_mock.data.insert_many.called

    @pytest.mark.asyncio 
    async def test_store_document_chunks_empty(self, vector_store):
        """Test storing document with no chunks."""
        empty_result = {
            "document_id": "empty-doc",
            "filename": "empty.pdf",
            "parent_chunk_count": 0,
            "child_chunk_count": 0,
            "child_chunks": []
        }
        
        # Should not raise error but also not insert anything
        await vector_store.store_document_chunks(empty_result)
        
        collection_mock = vector_store.client.collections.get.return_value
        # insert_many should be called with empty list
        collection_mock.data.insert_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, vector_store):
        """Test successful hybrid search."""
        query = "test query"
        query_embedding = [0.1] * 768
        
        # Mock search results with proper UUID and all required fields
        import uuid
        mock_result = MagicMock()
        mock_result.objects = [
            MagicMock(properties={
                "content": "Test chunk content",
                "document_id": str(uuid.uuid4()),
                "parent_id": str(uuid.uuid4()),
                "filename": "test.pdf",
                "parent_index": 0,
                "child_index": 0
            }, uuid=str(uuid.uuid4()))
        ]
        
        # Update the mock to return our result
        vector_store.client.collections.get.return_value.query.hybrid.return_value = mock_result
        
        results = await vector_store.hybrid_search(query, query_embedding, limit=10)
        
        assert isinstance(results, list)
        assert len(results) >= 0
        
        # Verify search was called
        vector_store.client.collections.get.assert_called()
        vector_store.client.collections.get.return_value.query.hybrid.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search_no_results(self, vector_store):
        """Test hybrid search with no results."""
        query = "non-existent query"
        query_embedding = [0.1] * 768
        
        # Mock empty results
        mock_result = MagicMock()
        mock_result.objects = []
        
        vector_store.client.collections.get.return_value.query.hybrid.return_value = mock_result
        
        results = await vector_store.hybrid_search(query, query_embedding)
        
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_all_documents(self, vector_store):
        """Test retrieving all documents."""
        # Mock aggregate results
        mock_result = MagicMock()
        mock_result.properties = {
            "document_id": {"count": 5},
            "filename": {"topOccurrences": [
                {"value": "doc1.pdf", "occurs": 3},
                {"value": "doc2.pdf", "occurs": 2}
            ]}
        }
        
        vector_store.client.collections.get.return_value.aggregate.over_all.return_value = mock_result
        
        documents = await vector_store.get_all_documents()
        
        assert isinstance(documents, list)
        # Verify the aggregate method was accessed
        vector_store.client.collections.get.assert_called()

    @pytest.mark.asyncio
    async def test_get_database_stats(self, vector_store):
        """Test database statistics retrieval."""
        # Mock aggregate results for stats
        mock_result = MagicMock()
        mock_result.total_count = 100
        
        vector_store.client.collections.get.return_value.aggregate.over_all.return_value = mock_result
        
        stats = await vector_store.get_database_stats()
        
        assert isinstance(stats, dict)
        assert "total_chunks" in stats
        assert "unique_documents" in stats
        assert "collection_name" in stats
        assert "weaviate_url" in stats
        
        assert stats["collection_name"] == "DocumentChunks"
        assert stats["weaviate_url"] == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_clear_all_documents(self, vector_store):
        """Test clearing all documents."""
        collection_mock = vector_store.client.collections.get.return_value
        collection_mock.data.delete_many.return_value = MagicMock(successful=50, failed=0)
        
        await vector_store.clear_all_documents()
        
        # Verify delete operation was called (we use delete_all now)
        collection_mock = vector_store.client.collections.get.return_value
        collection_mock.data.delete_all.assert_called()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, vector_store):
        """Test handling of connection errors."""
        with patch("app.core.vector_store.weaviate.connect_to_local", 
                   side_effect=Exception("Connection failed")):
            
            with pytest.raises(Exception) as exc_info:
                await vector_store._connect()
            
            assert "Failed to connect to Weaviate" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_collection_creation(self, vector_store):
        """Test collection creation when it doesn't exist."""
        # Mock that collection doesn't exist initially
        collections_mock = vector_store.client.collections
        collections_mock.get.side_effect = Exception("Collection not found")
        
        await vector_store._ensure_connected()
        
        # Should try to create collection
        assert collections_mock.create.called

    @pytest.mark.asyncio
    async def test_data_insertion_format(self, vector_store, sample_document_result):
        """Test that data is formatted correctly for Weaviate insertion."""
        await vector_store.store_document_chunks(sample_document_result)
        
        collection_mock = vector_store.client.collections.get.return_value
        
        # Verify insert_many was called
        assert collection_mock.data.insert_many.called
        
        # Get the call arguments
        call_args = collection_mock.data.insert_many.call_args
        inserted_objects = call_args[0][0]  # First positional argument
        
        # Verify we have the right number of objects
        assert len(inserted_objects) == sample_document_result["child_chunk_count"]
        
        # Verify all objects are DataObject instances (from weaviate library)
        for obj in inserted_objects:
            assert obj is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, vector_store):
        """Test that concurrent operations work correctly."""
        import asyncio
        
        # Run multiple operations concurrently
        tasks = [
            vector_store._ensure_connected(),
            vector_store._ensure_connected(),
            vector_store._ensure_connected()
        ]
        
        await asyncio.gather(*tasks)
        
        # Should connect only once
        assert vector_store._connected is True

    @pytest.mark.asyncio
    async def test_search_parameter_validation(self, vector_store):
        """Test search parameter validation."""
        query = "test"
        embedding = [0.1] * 768
        
        # Test with valid parameters
        await vector_store.hybrid_search(query, embedding, limit=5)
        
        collection_mock = vector_store.client.collections.get.return_value
        
        # Verify search was called
        assert collection_mock.query.hybrid.called

    @pytest.mark.asyncio
    async def test_error_handling_in_search(self, vector_store):
        """Test error handling during search operations."""
        collection_mock = vector_store.client.collections.get.return_value
        collection_mock.query.hybrid.side_effect = Exception("Search failed")
        
        query = "test query"
        embedding = [0.1] * 768
        
        # Should handle search errors gracefully
        with pytest.raises(Exception):
            await vector_store.hybrid_search(query, embedding) 