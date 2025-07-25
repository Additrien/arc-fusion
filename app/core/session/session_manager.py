"""
Session management service for Arc-Fusion.

This module handles session lifecycle and conversation history management,
separating this responsibility from the agent service.
"""

import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from app.core.config.services import SessionConfig
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.session.session_manager')


@dataclass
class ConversationEntry:
    """Represents a single conversation entry."""
    query: str
    answer: str
    timestamp: float
    processing_time: float
    agent_path: List[str]
    confidence: float


@dataclass
class SessionData:
    """Represents session data."""
    id: str
    conversation_history: List[ConversationEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionManager:
    """Service for managing user sessions and conversation history."""
    
    def __init__(self, config: SessionConfig):
        """
        Initialize the session manager.
        
        Args:
            config: Session configuration
        """
        self.config = config
        self.sessions: Dict[str, SessionData] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionData:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID, generates one if not provided
            
        Returns:
            SessionData for the session
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionData(id=session_id)
            logger.info(f"Created new session {session_id}")
        else:
            # Update last active time
            self.sessions[session_id].last_active = time.time()
            logger.debug(f"Retrieved existing session {session_id}")
        
        return self.sessions[session_id]
    
    def add_conversation_entry(self, session_id: str, entry: ConversationEntry) -> None:
        """
        Add a conversation entry to a session.
        
        Args:
            session_id: Session identifier
            entry: Conversation entry to add
        """
        session = self.get_or_create_session(session_id)
        
        # Add to conversation history
        session.conversation_history.append(entry)
        session.last_active = time.time()
        
        # Keep only last N conversations to prevent memory bloat
        if len(session.conversation_history) > self.config.max_history_length:
            session.conversation_history = session.conversation_history[-self.config.max_history_length:]
            logger.debug(f"Trimmed conversation history for session {session_id}")
        
        logger.info(f"Added conversation entry to session {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session information or None if session doesn't exist
        """
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.id,
            "message_count": len(session.conversation_history),
            "created_at": session.created_at,
            "last_active": session.last_active
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session's conversation history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session was cleared, False if session didn't exist
        """
        if session_id in self.sessions:
            self.sessions[session_id].conversation_history.clear()
            logger.info(f"Cleared conversation history for session {session_id}")
            return True
        else:
            logger.warning(f"Attempted to clear non-existent session {session_id}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions based on timeout configuration.
        
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_active > self.config.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active sessions.
        
        Returns:
            List of session information dictionaries
        """
        return [self.get_session_info(session_id) for session_id in self.sessions.keys()]
