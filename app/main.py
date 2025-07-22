from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore
from app.utils.logger import init_logging_from_env, get_logger, set_request_id
import uuid

load_dotenv()

# Initialize logging from environment
init_logging_from_env()
logger = get_logger('arc_fusion.api')

app = FastAPI(
    title="Arc-Fusion Multi-Agent RAG System",
    description="State-of-the-art multi-agent RAG system with Gemini integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services
document_processor = DocumentProcessor()
vector_store = VectorStore()

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancing."""
    return {"status": "healthy", "service": "arc-fusion-rag"}

@app.post("/api/v1/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF files with real-time ingestion."""
    request_id = set_request_id()
    
    logger.info("Document upload started", extra={
        "document_filename": file.filename,
        "content_type": file.content_type,
        "request_id": request_id
    })
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning("Invalid file type rejected", extra={
            "document_filename": file.filename,
            "content_type": file.content_type
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        logger.info("File content read", extra={
            "document_filename": file.filename,
            "file_size": len(content)
        })
        
        # Process document and generate chunks
        logger.info("Starting document processing", extra={
            "document_filename": file.filename
        })
        result = await document_processor.process_document(
            content, file.filename
        )
        
        # Store in vector database
        logger.info("Storing document chunks in vector database", extra={
            "document_id": result["document_id"],
            "parent_chunks": result["parent_chunk_count"],
            "child_chunks": result["child_chunk_count"]
        })
        await vector_store.store_document_chunks(result)
        
        response = {
            "document_id": result["document_id"],
            "filename": file.filename,
            "parent_chunks": result["parent_chunk_count"],
            "child_chunks": result["child_chunk_count"],
            "status": "processed"
        }
        
        logger.info("Document upload completed successfully", extra={
            "document_id": result["document_id"],
            "document_filename": file.filename,
            "processing_result": response
        })
        
        return response
        
    except Exception as e:
        logger.error("Document upload failed", extra={
            "document_filename": file.filename,
            "error": str(e),
            "error_type": type(e).__name__
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )

@app.get("/api/v1/documents")
async def list_documents() -> Dict[str, List[Dict[str, Any]]]:
    """List all ingested document IDs."""
    try:
        documents = await vector_store.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@app.get("/api/v1/documents/stats")
async def get_document_stats() -> Dict[str, Any]:
    """Database statistics and metrics."""
    try:
        stats = await vector_store.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}"
        )

@app.delete("/api/v1/documents", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_documents():
    """Clear all documents from database."""
    try:
        await vector_store.clear_all_documents()
        await document_processor.clear_parent_store()
        return JSONResponse(
            status_code=status.HTTP_204_NO_CONTENT,
            content=None
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear documents: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    ) 