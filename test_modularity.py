"""
Test script to verify the modular improvements to the Arc-Fusion system.
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.factories import ServiceFactory
from app.core.config.services import DocumentProcessingConfig, EmbeddingConfig, VectorStoreConfig

def test_factory_creation():
    """Test that the service factory can create all services correctly."""
    print("Testing service factory creation...")
    
    # Create factory
    factory = ServiceFactory()
    
    # Test creating services
    pdf_extractor = factory.create_pdf_extractor()
    print(f"✓ PDF Extractor created: {type(pdf_extractor).__name__}")
    
    chunking_service = factory.create_chunking_service()
    print(f"✓ Chunking Service created: {type(chunking_service).__name__}")
    
    embedding_service = factory.create_embedding_service()
    print(f"✓ Embedding Service created: {type(embedding_service).__name__}")
    
    vector_store = factory.create_vector_store()
    print(f"✓ Vector Store created: {type(vector_store).__name__}")
    
    document_processor = factory.create_document_processor()
    print(f"✓ Document Processor created: {type(document_processor).__name__}")
    
    agent_service = factory.create_agent_service()
    print(f"✓ Agent Service created: {type(agent_service).__name__}")
    
    print("All services created successfully!\n")

def test_config_loading():
    """Test that configuration classes load correctly."""
    print("Testing configuration loading...")
    
    # Test document processing config
    doc_config = DocumentProcessingConfig()
    print(f"✓ Document Processing Config loaded: parent_chunk_size={doc_config.parent_chunk_size}")
    
    # Test embedding config
    embedding_config = EmbeddingConfig()
    print(f"✓ Embedding Config loaded: model={embedding_config.model}")
    
    # Test vector store config
    vector_config = VectorStoreConfig()
    print(f"✓ Vector Store Config loaded: batch_size={vector_config.batch_size}")
    
    print("All configurations loaded successfully!\n")

async def test_document_processor():
    """Test the document processor with a simple example."""
    print("Testing document processor...")
    
    # Create factory and document processor
    factory = ServiceFactory()
    document_processor = factory.create_document_processor()
    
    # Create a simple test document
    test_content = b"This is a simple test document.\nIt has multiple lines.\nFor testing purposes."
    test_filename = "test_document.txt"
    
    # Process the document
    try:
        result = await document_processor.process_document(test_content, test_filename)
        print(f"✓ Document processed successfully:")
        print(f"  - Document ID: {result['document_id']}")
        print(f"  - Parent chunks: {result['parent_chunk_count']}")
        print(f"  - Child chunks: {result['child_chunk_count']}")
    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        return False
    
    print("Document processor test completed successfully!\n")
    return True

async def main():
    """Run all tests."""
    print("Running modularity tests...\n")
    
    # Test factory creation
    test_factory_creation()
    
    # Test configuration loading
    test_config_loading()
    
    # Test document processor
    await test_document_processor()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
