"""
Graph State Management for Multi-Agent RAG System.

This module defines the state structure that flows between agents in our
LangGraph-based multi-agent system.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    The state object that flows through our multi-agent graph.
    
    This comprehensive state allows agents to communicate and share context
    while maintaining the conversation flow and accumulated knowledge.
    """
    
    # Core conversation data
    messages: Annotated[List[Dict[str, Any]], add_messages]
    query: str
    session_id: str
    
    # Routing and intent
    intent: Optional[str]  # "retrieve_corpus", "search_web", "clarify", "end"
    required_capability: Optional[str]  # For dynamic routing
    confidence_score: Optional[float]  # Agent confidence in routing decision
    
    # Context retrieval
    retrieved_context: List[str]  # RAG context chunks
    retrieval_scores: Optional[List[float]]  # Hybrid search scores for chunks (0-1 range)
    best_retrieval_score: Optional[float]  # Highest hybrid search score for routing
    parent_chunks: Optional[List[str]]  # Full parent context
    document_sources: Optional[List[Dict[str, Any]]]  # Source metadata
    
    # Web search
    # --- CHANGE 1: web_context is now a LIST of strings, not a single string. ---
    # This allows us to keep each web source's text separate for citations.
    web_context: Optional[List[str]]  # Web search results
    search_query: Optional[str]  # Optimized search query
    web_sources: Optional[List[Dict[str, Any]]]  # Web source metadata
    
    # Response generation
    final_answer: Optional[str]  # Generated response
    # --- CHANGE 2: The citation dictionary now uses 'Any' for its values. ---
    # This is because it now includes the source 'text' (string) and 'score' (float),
    # not just strings.
    citations: Optional[List[Dict[str, Any]]]  # Source citations with text
    answer_confidence: Optional[float]  # Confidence in final answer
    
    # System metadata
    step_count: int  # For debugging and monitoring
    processing_time: Optional[float]  # Total processing time
    agent_path: List[str]  # Track which agents were used
    error_info: Optional[Dict[str, Any]]  # Error tracking
    fallback_reason: Optional[str]  # Reason for web search fallback (no_results, low_quality, etc.)
    
    # Orchestration state for multi-step information gathering
    tasks_to_run: List[str]
    tasks_completed: List[str]
    
    # Future extensibility - agents can add custom fields
    custom_data: Optional[Dict[str, Any]]


# Convenience type for agent functions
from typing import Callable
AgentFunction = Callable[[GraphState], GraphState]