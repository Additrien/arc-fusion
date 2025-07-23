from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore
from app.core.agent_service import agent_service
from app.utils.logger import init_logging_from_env, get_logger, set_request_id
from app.utils.performance import get_performance_summary, clear_performance_metrics
from app.utils.cache import embedding_cache, hyde_cache
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

# Pydantic models for request/response validation
class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ClearMemoryRequest(BaseModel):
    session_id: str

class AskResponse(BaseModel):
    answer: str
    session_id: str
    success: bool
    processing_time: float
    agent_path: List[str]
    citations: List[Dict[str, str]]
    confidence: float
    metadata: Dict[str, Any]

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancing."""
    return {"status": "healthy", "service": "arc-fusion-rag"}

@app.get("/api/v1/performance")
async def get_performance_metrics():
    """Get performance metrics for the system."""
    return get_performance_summary()

@app.delete("/api/v1/performance")
async def clear_performance_metrics_endpoint():
    """Clear performance metrics."""
    clear_performance_metrics()
    return {"message": "Performance metrics cleared"}

@app.get("/api/v1/cache")
async def get_cache_info():
    """Get cache statistics."""
    return {
        "embedding_cache": {
            "size": embedding_cache.size(),
            "ttl": 3600
        },
        "hyde_cache": {
            "size": hyde_cache.size(),
            "ttl": 1800
        }
    }

@app.delete("/api/v1/cache")
async def clear_cache():
    """Clear all caches."""
    embedding_cache.clear()
    hyde_cache.clear()
    return {"message": "All caches cleared"}

# Document Management Endpoints
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
        storage_result = await vector_store.store_document_chunks(result)
        
        # Log storage results
        if storage_result:
            success_rate = storage_result.get("success_rate", 0)
            stored_chunks = storage_result.get("stored_chunks", 0)
            total_chunks = storage_result.get("total_chunks", 0)
            
            logger.info("Vector storage completed", extra={
                "document_id": result["document_id"],
                "stored_chunks": stored_chunks,
                "total_chunks": total_chunks,
                "success_rate": f"{success_rate:.2%}"
            })
            
            if success_rate < 1.0:
                logger.warning("Incomplete document storage", extra={
                    "document_id": result["document_id"],
                    "missing_chunks": total_chunks - stored_chunks,
                    "success_rate": f"{success_rate:.2%}"
                })
        else:
            logger.warning("No storage result returned from vector store")
        
        response = {
            "document_id": result["document_id"],
            "filename": file.filename,
            "parent_chunks": result["parent_chunk_count"],
            "child_chunks": result["child_chunk_count"],
            "status": "processed"
        }
        
        # Add storage metrics to response if available
        if storage_result:
            response.update({
                "storage_metrics": {
                    "stored_chunks": storage_result.get("stored_chunks", 0),
                    "success_rate": storage_result.get("success_rate", 0)
                }
            })
        
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

# Multi-Agent Chat Endpoints
@app.post("/api/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question to the multi-agent RAG system.
    
    This endpoint processes user queries through our sophisticated multi-agent
    pipeline including intent routing, document retrieval, web search, and
    response synthesis.
    """
    request_id = set_request_id()
    
    logger.info("Query processing started", extra={
        "query": request.query[:100] + "..." if len(request.query) > 100 else request.query,
        "session_id": request.session_id,
        "request_id": request_id
    })
    
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        # Process through multi-agent system
        response = await agent_service.process_query(
            query=request.query.strip(),
            session_id=request.session_id
        )
        
        logger.info("Query processing completed", extra={
            "session_id": response["session_id"],
            "success": response["success"],
            "processing_time": response["processing_time"],
            "agent_path": response["agent_path"],
            "confidence": response["confidence"],
            "request_id": request_id
        })
        
        return AskResponse(**response)
        
    except Exception as e:
        logger.error("Query processing failed", extra={
            "query": request.query[:100],
            "session_id": request.session_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "request_id": request_id
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

@app.post("/api/v1/clear-memory")
async def clear_session_memory(request: ClearMemoryRequest):
    """Clear conversation memory for a specific session."""
    request_id = set_request_id()
    
    logger.info("Memory clear requested", extra={
        "session_id": request.session_id,
        "request_id": request_id
    })
    
    try:
        success = agent_service.clear_session_memory(request.session_id)
        
        if success:
            logger.info("Memory cleared successfully", extra={
                "session_id": request.session_id,
                "request_id": request_id
            })
            return {
                "success": True,
                "message": f"Memory cleared for session {request.session_id}"
            }
        else:
            logger.warning("Session not found for memory clear", extra={
                "session_id": request.session_id,
                "request_id": request_id
            })
            return {
                "success": False,
                "message": f"No active session found for {request.session_id}"
            }
        
    except Exception as e:
        logger.error("Memory clear failed", extra={
            "session_id": request.session_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "request_id": request_id
        })
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear memory: {str(e)}"
        )

# System Information Endpoints
@app.get("/api/v1/agents/info")
async def get_agents_info():
    """Get information about registered agents and their capabilities."""
    try:
        agent_info = agent_service.get_agent_info()
        return {
            "agents": agent_info["agents"],
            "capabilities": agent_info["capabilities"],
            "graph_built": agent_info["graph_built"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent info: {str(e)}"
        )

@app.get("/api/v1/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    try:
        session_info = agent_service.get_session_info(session_id)
        if session_info:
            return session_info
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("FASTAPI_HOST", "0.0.0.0"),
        port=int(os.getenv("FASTAPI_PORT", 8000)),
        reload=os.getenv("ENVIRONMENT") == "development"
    ) 