"""
Agent Service - Integration layer between AgentFramework and FastAPI.

This service provides the main interface for the multi-agent system,
handling session management and query processing for the API layer.
"""

import uuid
from typing import Dict, Any, Optional
import asyncio
import time

from ..agents.framework import AgentFramework
from ..agents.loader import load_all_agents  # This triggers agent registration
from ..utils.logger import get_logger

logger = get_logger('arc_fusion.core.agent_service')


class AgentService:
    """
    Service that manages the multi-agent system for FastAPI integration.
    
    This service provides:
    - Session management for conversation continuity
    - Query processing through the agent framework
    - Error handling and response formatting
    """
    
    def __init__(self):
        self.framework = AgentFramework()
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._graph_built = False
        
        # Initialize the framework
        self._initialize_framework()
    
    def _initialize_framework(self):
        """Initialize the agent framework and build the graph."""
        try:
            logger.info("Initializing agent framework")
            
            # Build the agent graph
            self.framework.build_graph()
            self._graph_built = True
            
            # Log framework info
            agent_info = self.framework.get_agent_info()
            logger.info(f"Agent framework initialized with {len(agent_info['agents'])} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent framework: {str(e)}")
            raise
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: User's question
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self._graph_built:
            raise RuntimeError("Agent framework not initialized")
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query for session {session_id}: {query[:100]}...")
            
            # Get session context
            session_data = self._get_session_data(session_id)
            
            # Process through agent framework
            final_state = await self.framework.process_query(
                query=query,
                session_id=session_id,
                initial_state=session_data
            )
            
            processing_time = time.time() - start_time
            
            # Update session with conversation history
            self._update_session(session_id, final_state, query, processing_time)
            
            # Format response
            response = self._format_response(final_state, processing_time)
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s for session {session_id}")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing query for session {session_id}: {str(e)}")
            
            # Return error response
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "agent_path": [],
                "citations": [],
                "confidence": 0.0
            }
    
    def clear_session_memory(self, session_id: str) -> bool:
        """
        Clear memory for a specific session.
        
        Args:
            session_id: Session to clear
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared memory for session {session_id}")
            return True
        else:
            logger.warning(f"Attempted to clear non-existent session {session_id}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        session_data = self.sessions.get(session_id)
        if session_data:
            return {
                "session_id": session_id,
                "message_count": len(session_data.get("conversation_history", [])),
                "created_at": session_data.get("created_at"),
                "last_active": session_data.get("last_active")
            }
        return None
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about registered agents and capabilities."""
        return self.framework.get_agent_info()
    
    def _get_session_data(self, session_id: str) -> Dict[str, Any]:
        """Get or create session data."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "conversation_history": [],
                "created_at": time.time(),
                "last_active": time.time()
            }
        
        return self.sessions[session_id]
    
    def _update_session(self, session_id: str, final_state: Dict[str, Any], 
                       query: str, processing_time: float):
        """Update session with new conversation data."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {"conversation_history": []}
        
        # Add to conversation history
        conversation_entry = {
            "query": query,
            "answer": final_state.get("final_answer", ""),
            "timestamp": time.time(),
            "processing_time": processing_time,
            "agent_path": final_state.get("agent_path", []),
            "confidence": final_state.get("answer_confidence", 0.0)
        }
        
        self.sessions[session_id]["conversation_history"].append(conversation_entry)
        self.sessions[session_id]["last_active"] = time.time()
        
        # Keep only last 10 conversations to prevent memory bloat
        if len(self.sessions[session_id]["conversation_history"]) > 10:
            self.sessions[session_id]["conversation_history"] = \
                self.sessions[session_id]["conversation_history"][-10:]
    
    def _format_response(self, final_state: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Format the final state into a clean API response."""
        
        return {
            "answer": final_state.get("final_answer", "No answer generated."),
            "session_id": final_state.get("session_id"),
            "success": True,
            "processing_time": processing_time,
            "agent_path": final_state.get("agent_path", []),
            "citations": final_state.get("citations", []),
            "confidence": final_state.get("answer_confidence", 0.0),
            "metadata": {
                "step_count": final_state.get("step_count", 0),
                "intent": final_state.get("intent"),
                "has_document_context": bool(final_state.get("retrieved_context")),
                "has_web_context": bool(final_state.get("web_context")),
                "document_sources": len(final_state.get("document_sources", [])),
                "web_sources": len(final_state.get("web_sources", [])),
                "errors": final_state.get("error_info", {})
            }
        }


# Global agent service instance
agent_service = AgentService() 