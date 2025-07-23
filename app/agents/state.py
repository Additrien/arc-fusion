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
    retrieval_scores: Optional[List[float]]  # Hybrid search scores for chunks
    cross_encoder_scores: Optional[List[float]]  # Cross-encoder precision scores
    best_cross_encoder_score: Optional[float]  # Highest cross-encoder score
    parent_chunks: Optional[List[str]]  # Full parent context
    document_sources: Optional[List[Dict[str, Any]]]  # Source metadata
    
    # Web search
    web_context: Optional[str]  # Web search results
    search_query: Optional[str]  # Optimized search query
    web_sources: Optional[List[Dict[str, Any]]]  # Web source metadata
    
    # Response generation
    final_answer: Optional[str]  # Generated response
    citations: Optional[List[Dict[str, str]]]  # Source citations
    answer_confidence: Optional[float]  # Confidence in final answer
    
    # System metadata
    step_count: int  # For debugging and monitoring
    processing_time: Optional[float]  # Total processing time
    agent_path: List[str]  # Track which agents were used
    error_info: Optional[Dict[str, Any]]  # Error tracking
    fallback_reason: Optional[str]  # Reason for web search fallback (no_results, low_quality, etc.)
    
    # Future extensibility - agents can add custom fields
    custom_data: Optional[Dict[str, Any]]


# Convenience type for agent functions
from typing import Callable
AgentFunction = Callable[[GraphState], GraphState] 