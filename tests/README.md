# Arc-Fusion Test Suite

Comprehensive test suite for the Arc-Fusion document processing pipeline, covering PDF ingestion, chunking, embedding generation, and vector storage.

## Test Structure

### Test Categories

- **Unit Tests** (`test_document_processor.py`, `test_vector_store.py`): Test individual components in isolation
- **API Tests** (`test_api.py`): Test FastAPI endpoints and HTTP interactions  
- **Integration Tests** (`test_integration.py`): Test complete end-to-end workflows

### Test Coverage

The test suite covers:

- ✅ **PDF Processing**: Text extraction, chunking, validation
- ✅ **Embedding Generation**: Gemini API integration, rate limiting, retry logic
- ✅ **Vector Storage**: Weaviate operations, search functionality  
- ✅ **API Endpoints**: Document upload, health checks, error handling
- ✅ **Error Handling**: Network failures, API limits, invalid inputs
- ✅ **Logging Integration**: Structured logging, request tracking
- ✅ **Concurrency**: Multiple document processing, race conditions

## Running Tests

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all tests with coverage
./scripts/run_tests.sh

# Run specific test categories
./scripts/run_tests.sh --unit
./scripts/run_tests.sh --api  
./scripts/run_tests.sh --integration

# Run without coverage (faster)
./scripts/run_tests.sh --no-coverage

# Quiet mode
./scripts/run_tests.sh --quiet
```

### Manual Test Execution

```bash
# Run all tests
pytest

# Run specific categories
pytest -m unit
pytest -m api
pytest -m integration

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test files
pytest tests/test_document_processor.py
pytest tests/test_api.py::TestAPI::test_upload_document_success

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

### Test Markers

Tests are marked for selective execution:

- `@pytest.mark.unit` - Fast unit tests for individual components
- `@pytest.mark.api` - API endpoint tests with FastAPI test client
- `@pytest.mark.integration` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests that take longer to run

### Coverage Reports

Coverage reports are generated in multiple formats:

- **HTML**: `htmlcov/index.html` - Interactive web report
- **Terminal**: Real-time coverage summary
- **XML**: `coverage.xml` - For CI/CD integration

Target coverage: **80%** minimum

## Test Configuration

### Environment Variables

Tests automatically set safe defaults:

```bash
ENVIRONMENT=test
GOOGLE_API_KEY=test_key
WEAVIATE_URL=http://localhost:8080
ENABLE_FILE_LOGGING=false
```

### Mocking Strategy

- **Gemini API**: Mocked to avoid API calls and costs
- **Weaviate**: Mocked for isolated testing
- **PDF Generation**: Uses reportlab for synthetic test documents
- **Logging**: Captured and verified for proper operation

### Fixtures

Key test fixtures in `conftest.py`:

- `sample_pdf_bytes` - Generated PDF for testing
- `mock_gemini_embeddings` - Mocked embedding generation
- `mock_weaviate_client` - Mocked vector database
- `document_processor` - DocumentProcessor instance
- `vector_store` - VectorStore instance  
- `test_client` - FastAPI test client

## Test Examples

### Unit Test Example
```python
@pytest.mark.asyncio
async def test_process_document_success(self, document_processor, sample_pdf_bytes):
    """Test successful document processing."""
    result = await document_processor.process_document(sample_pdf_bytes, "test.pdf")
    
    assert result["filename"] == "test.pdf"
    assert result["parent_chunk_count"] > 0
    assert result["child_chunk_count"] > 0
```

### API Test Example
```python
def test_upload_document_success(self, test_client, sample_pdf_bytes):
    """Test successful document upload."""
    files = {"file": ("test.pdf", sample_pdf_bytes, "application/pdf")}
    response = test_client.post("/api/v1/documents", files=files)
    
    assert response.status_code == 201
    assert response.json()["status"] == "processed"
```

### Integration Test Example
```python
@pytest.mark.asyncio
async def test_complete_pipeline(self, document_processor, vector_store, sample_pdf_bytes):
    """Test complete processing pipeline."""
    # Process document
    result = await document_processor.process_document(sample_pdf_bytes, "test.pdf")
    
    # Store in vector database
    await vector_store.store_document_chunks(result)
    
    # Verify storage
    assert vector_store.client.collections.get.called
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Tests
  run: |
    pip install -r requirements.txt
    pytest --cov=app --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Docker Testing

```bash
# Run tests in Docker container
docker run --rm -v $(pwd):/app python:3.11-slim \
  bash -c "cd /app && pip install -r requirements.txt && pytest"
```

## Debugging Tests

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Async Warnings**: Use `@pytest.mark.asyncio` for async tests
3. **Mock Issues**: Verify patch targets match actual import paths
4. **Fixture Scope**: Use appropriate fixture scopes for test isolation

### Debug Mode

```bash
# Run with debug output
pytest -s -v --tb=long

# Run specific test with debugging
pytest -s tests/test_document_processor.py::TestDocumentProcessor::test_process_document_success

# Drop into debugger on failure
pytest --pdb
```

## Performance Testing

While not included in the current suite, performance tests can be added:

```python
@pytest.mark.slow
def test_large_document_processing_performance(self):
    """Test processing performance with large documents."""
    import time
    start_time = time.time()
    
    # Process large document
    result = await process_large_document()
    
    processing_time = time.time() - start_time
    assert processing_time < 30  # Should complete within 30 seconds
```

## Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.unit`, etc.)
2. Follow naming convention: `test_<functionality>_<scenario>`
3. Include docstrings explaining test purpose
4. Mock external dependencies appropriately
5. Maintain test isolation with proper fixtures
6. Add edge cases and error conditions

## Test Data

Test data is generated programmatically:

- **PDFs**: Created with reportlab for consistent testing
- **Embeddings**: Mock vectors with realistic dimensions (768)
- **API Responses**: Structured to match actual API contracts

No external test data files required - everything is self-contained. 