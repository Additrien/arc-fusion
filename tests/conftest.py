"""
Pytest configuration and shared fixtures for Arc-Fusion tests.
"""

import pytest
import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["GOOGLE_API_KEY"] = "test_key"
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
os.environ["ENABLE_FILE_LOGGING"] = "false"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a sample PDF as bytes for testing."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Add multiple pages with different content for chunking tests
    c.drawString(100, 750, "Arc-Fusion Test Document")
    c.drawString(100, 720, "This is the first page with some sample content.")
    c.drawString(100, 700, "We need enough text to test the chunking algorithm.")
    c.drawString(100, 680, "This paragraph contains information about text-to-SQL.")
    c.drawString(100, 660, "Large language models can be prompted to generate SQL.")
    c.drawString(100, 640, "The key is providing good examples and clear instructions.")
    
    # Add more content to ensure multiple chunks
    for i in range(20):
        y_pos = 600 - (i * 20)
        if y_pos > 100:
            c.drawString(100, y_pos, f"Additional line {i+1} to create more content for chunking.")
    
    c.showPage()  # Second page
    c.drawString(100, 750, "Page 2: Advanced Prompting Techniques")
    c.drawString(100, 720, "Zero-shot prompting allows models to perform tasks without examples.")
    c.drawString(100, 700, "Few-shot prompting provides examples to guide the model.")
    c.drawString(100, 680, "Chain-of-thought prompting encourages step-by-step reasoning.")
    
    c.save()
    
    buffer.seek(0)
    return buffer.read()

@pytest.fixture
def mock_gemini_embeddings():
    """Mock Gemini embeddings to avoid API calls during testing."""
    with patch("app.core.document_processor.GoogleGenerativeAIEmbeddings") as mock_embeddings:
        # Create a mock embedding (768 dimensions for text-embedding-004)
        mock_embedding = [0.1] * 768
        
        mock_instance = AsyncMock()
        mock_instance.aembed_query = AsyncMock(return_value=mock_embedding)
        mock_embeddings.return_value = mock_instance
        
        yield mock_instance

@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client to avoid database calls during testing."""
    # Create mock client and collections
    mock_client = MagicMock()
    mock_collections = MagicMock()
    mock_collection = MagicMock()
    
    # Mock collection operations with proper return values
    mock_batch_return = MagicMock()
    mock_batch_return.failed = 0
    mock_batch_return.successful = 2
    mock_collection.data.insert_many.return_value = mock_batch_return
    
    # Mock search results  
    mock_search_result = MagicMock()
    mock_search_result.objects = []
    mock_collection.query.hybrid.return_value = mock_search_result
    
    # Mock aggregate results
    mock_aggregate_result = MagicMock()
    mock_aggregate_result.total_count = 0
    mock_aggregate_result.properties = {}
    mock_collection.aggregate.over_all.return_value = mock_aggregate_result
    
    # Mock delete results
    mock_delete_result = MagicMock()
    mock_delete_result.successful = 0
    mock_delete_result.failed = 0
    mock_collection.data.delete_many.return_value = mock_delete_result
    mock_collection.data.delete_all.return_value = mock_delete_result
    
    # Mock collections behavior
    mock_collections.get.return_value = mock_collection
    mock_collections.create = MagicMock()
    mock_collections.list_all.return_value = []
    
    mock_client.collections = mock_collections
    mock_client.is_ready.return_value = True
    
    return mock_client

@pytest.fixture
def document_processor(mock_gemini_embeddings):
    """Create DocumentProcessor instance with mocked dependencies."""
    from app.core.document_processor import DocumentProcessor
    return DocumentProcessor()

@pytest.fixture
def vector_store(mock_weaviate_client):
    """Create VectorStore instance with mocked dependencies."""
    from app.core.vector_store import VectorStore
    
    # Create vector store instance
    store = VectorStore()
    
    # Inject the mocked client
    store.client = mock_weaviate_client
    store._connected = True
    
    return store

@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    with patch("app.main.document_processor"), patch("app.main.vector_store"):
        from app.main import app
        client = TestClient(app)
        yield client

@pytest.fixture
def sample_document_result() -> Dict[str, Any]:
    """Sample document processing result for testing."""
    import uuid
    return {
        "document_id": str(uuid.uuid4()),
        "filename": "test.pdf", 
        "parent_chunk_count": 2,
        "child_chunk_count": 2,
        "child_chunks": [
            {
                "id": str(uuid.uuid4()),
                "parent_id": str(uuid.uuid4()), 
                "content": "This is chunk 1 content.",
                "embedding": [0.1] * 768,
                "document_id": str(uuid.uuid4()),
                "filename": "test.pdf",
                "parent_index": 0,
                "child_index": 0
            },
            {
                "id": str(uuid.uuid4()),
                "parent_id": str(uuid.uuid4()),
                "content": "This is chunk 2 content.", 
                "embedding": [0.2] * 768,
                "document_id": str(uuid.uuid4()),
                "filename": "test.pdf",
                "parent_index": 0,
                "child_index": 1
            }
        ]
    }

@pytest.fixture
def mock_logger():
    """Mock logger to capture log messages during testing."""
    with patch("app.utils.logger.get_logger") as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance

# Helper fixture for rate limit testing
@pytest.fixture
def slow_gemini_embeddings():
    """Mock Gemini embeddings with rate limiting behavior."""
    with patch("app.core.document_processor.GoogleGenerativeAIEmbeddings") as mock_embeddings:
        mock_instance = AsyncMock()
        
        # Simulate rate limiting on first call, success on retry
        call_count = 0
        async def mock_embed_query(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 Rate limit exceeded")
            return [0.1] * 768
            
        mock_instance.aembed_query = mock_embed_query
        mock_embeddings.return_value = mock_instance
        
        yield mock_instance 