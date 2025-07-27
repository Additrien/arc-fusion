"""
Synthesis Agent - Final answer generation using Gemini Pro.

This agent synthesizes the final response using context from document retrieval
and/or web search, employing Gemini 2.5 Pro for high-quality answer generation.
"""

import os
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from .registry import AgentRegistry
from .state import GraphState
from ..utils.logger import get_logger
from ..utils.performance import time_async_function, time_async_block
from .. import config
from ..prompts import (
    SYNTHESIS_PROMPT_HEADER, SYNTHESIS_CONVERSATION_HISTORY_SECTION,
    SYNTHESIS_DOCUMENT_CONTEXT_SECTION, SYNTHESIS_WEB_CONTEXT_SECTION,
    SYNTHESIS_INSTRUCTIONS, SYNTHESIS_NO_CONTEXT_SECTION,
    format_conversation_history, format_document_context, format_web_context
)

logger = get_logger('arc_fusion.agents.synthesis')

# Configure Gemini
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class SynthesisService:
    """Service for generating final answers using retrieved context."""
    
    def __init__(self):
        # Use configured synthesis model
        self.synthesis_model = config.SYNTHESIS_MODEL
        
        # Optimized generation configuration for speed
        self.generation_config = {
            "temperature": config.SYNTHESIS_TEMP,
            "top_p": config.SYNTHESIS_TOP_P,
            "max_output_tokens": config.SYNTHESIS_MAX_TOKENS,
            "thinking_config": {
                "thinking_budget": 0
            }
        }
    @time_async_function("synthesis.synthesize_response")
    async def synthesize_response(self, state: GraphState) -> GraphState:
        """
        Generate the final answer using available context.
        
        Args:
            state: Current graph state with context from retrieval/search
            
        Returns:
            Final state with synthesized answer and citations
        """
        query = state["query"]
        session_id = state["session_id"]
        
        logger.info(f"Starting response synthesis for session {session_id}")
        
        try:
            # Gather available context
            context_info = self._gather_context(state)
            
            # Get conversation history from state for follow-up context and ensure it's in the right format
            conversation_history = state.get("conversation_messages", [])
            
            # Defensive handling: convert any LangGraph message objects to plain dicts
            safe_conversation_history = []
            for msg in conversation_history:
                if hasattr(msg, 'content') and hasattr(msg, 'type'):  # LangGraph message object
                    # Convert LangGraph message to plain dict
                    safe_conversation_history.append({
                        "role": getattr(msg, 'type', 'unknown'),
                        "content": str(msg.content),
                        "timestamp": getattr(msg, 'timestamp', None)
                    })
                elif isinstance(msg, dict):  # Already a plain dict
                    safe_conversation_history.append(msg)
                else:
                    # Skip unknown message types
                    continue
            
            conversation_history = safe_conversation_history
            
            # Create synthesis prompt with conversation history
            prompt = self._create_synthesis_prompt(query, context_info, conversation_history)
            
            # Generate response using Gemini Pro
            logger.info("Generating response with Gemini Pro")
            response = client.models.generate_content(
                model=self.synthesis_model,
                contents=prompt,
                config=types.GenerateContentConfig(**self.generation_config)
            )
            
            final_answer = response.text.strip()
            
            # Extract citations and confidence
            citations = self._generate_citations(context_info)
            confidence = self._assess_confidence(context_info, final_answer)
            
            logger.info(f"Response synthesized successfully with {len(citations)} citations")
            
            # Update state with final response
            updated_state = state.copy()
            updated_state.update({
                "final_answer": final_answer,
                "citations": citations,
                "answer_confidence": confidence,
                "step_count": state.get("step_count", 0) + 1
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {str(e)}")
            return self._create_error_state(state, e)
    
    def _gather_context(self, state: GraphState) -> Dict[str, Any]:
        """
        Gather all available context from previous agents.
        
        Returns:
            Dictionary containing organized context information
        """
        context_info = {
            "has_document_context": False,
            "has_web_context": False,
            "document_chunks": [],
            "document_sources": [],
            "web_content": [],
            "web_sources": [],
            "total_sources": 0
        }
        
        # Document context
        retrieved_context = state.get("retrieved_context", [])
        document_sources = state.get("document_sources", [])
        
        if retrieved_context:
            context_info.update({
                "has_document_context": True,
                "document_chunks": retrieved_context,
                "document_sources": document_sources
            })
            logger.debug(f"Found {len(retrieved_context)} document chunks")
        
        # Web context
        web_context = state.get("web_context", [])
        web_sources = state.get("web_sources", [])
        
        if web_context:
            context_info.update({
                "has_web_context": True,
                "web_content": web_context,
                "web_sources": web_sources
            })
            logger.debug(f"Found web context with {len(web_sources)} sources")
        
        context_info["total_sources"] = len(document_sources) + len(web_sources)
        return context_info
    
    def _create_synthesis_prompt(self, query: str, context_info: Dict[str, Any], conversation_history: List[Dict[str, Any]] = None) -> str:
        """
        Create the synthesis prompt for Gemini Pro using centralized prompts.
        
        Args:
            query: User's original query
            context_info: Organized context information
            conversation_history: Previous conversation messages for context
            
        Returns:
            Formatted prompt for response generation
        """
        prompt_parts = [SYNTHESIS_PROMPT_HEADER]
        
        # Add conversation history section if available
        if conversation_history:
            formatted_history = format_conversation_history(conversation_history)
            prompt_parts.append(SYNTHESIS_CONVERSATION_HISTORY_SECTION.format(
                conversation_history=formatted_history
            ))
        
        # Add document context section if available
        if context_info["has_document_context"]:
            formatted_doc_context = format_document_context(
                context_info["document_chunks"], 
                context_info["document_sources"]
            )
            prompt_parts.append(SYNTHESIS_DOCUMENT_CONTEXT_SECTION.format(
                document_context=formatted_doc_context
            ))
        
        # Add web context section if available
        if context_info["has_web_context"]:
            formatted_web_context = format_web_context(
                context_info["web_content"], 
                context_info["web_sources"]
            )
            prompt_parts.append(SYNTHESIS_WEB_CONTEXT_SECTION.format(
                web_context=formatted_web_context
            ))
        
        # Add instructions section
        prompt_parts.append(SYNTHESIS_INSTRUCTIONS.format(query=query))
        
        # Handle empty context case
        if not context_info["has_document_context"] and not context_info["has_web_context"]:
            prompt_parts.append(SYNTHESIS_NO_CONTEXT_SECTION)
        
        return "\n".join(prompt_parts)
    
    def _generate_citations(self, context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate citations for the sources used, including the source text.
        
        Args:
            context_info: Context information with sources
            
        Returns:
            List of citation objects with text content.
        """
        citations = []
        
        # Document citations
        doc_chunks = context_info.get("document_chunks", [])
        doc_sources = context_info.get("document_sources", [])
        
        for i, source in enumerate(doc_sources):
            document_id = source.get("document_id", "")
            if document_id:
                document_id = str(document_id)
            
            citations.append({
                "type": "document",
                "id": f"doc_{i+1}",
                "filename": source.get("filename", "Unknown Document"),
                "document_id": document_id,
                "score": source.get("score", 0.0),
                "text": doc_chunks[i] if i < len(doc_chunks) else ""
            })
        
        # Web citations
        web_sources = context_info.get("web_sources", [])
        for i, source in enumerate(web_sources):
            citations.append({
                "type": "web",
                "id": f"web_{i+1}",
                "title": source.get("title", "Unknown Title"),
                "url": source.get("url", ""),
                "score": source.get("score", 0.0),
                "text": source.get("content", "") # This content is now the full text
            })
            
        return citations
    
    def _assess_confidence(self, context_info: Dict[str, Any], answer: str) -> float:
        """
        Assess confidence in the generated answer.
        
        Args:
            context_info: Available context information
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        
        confidence = config.SYNTHESIS_BASE_CONFIDENCE
        
        # Boost confidence based on context availability
        if context_info["has_document_context"]:
            confidence += config.SYNTHESIS_DOC_CONTEXT_BONUS
            # More sources = higher confidence
            doc_count = len(context_info["document_sources"])
            confidence += min(config.SYNTHESIS_DOC_COUNT_BONUS * doc_count, config.SYNTHESIS_MAX_DOC_BONUS)
        
        if context_info["has_web_context"]:
            confidence += config.SYNTHESIS_WEB_CONTEXT_BONUS
            # More sources = higher confidence
            web_count = len(context_info["web_sources"])
            confidence += min(config.SYNTHESIS_WEB_COUNT_BONUS * web_count, config.SYNTHESIS_MAX_WEB_BONUS)
        
        # Reduce confidence for short answers (might indicate insufficient info)
        if len(answer) < config.SYNTHESIS_SHORT_ANSWER_LENGTH:
            confidence -= config.SYNTHESIS_SHORT_ANSWER_PENALTY
        
        # Reduce confidence if answer mentions lack of information
        if any(phrase in answer.lower() for phrase in config.SYNTHESIS_UNCERTAINTY_PHRASES):
            confidence -= config.SYNTHESIS_UNCERTAINTY_PENALTY
        
        return max(0.0, min(1.0, confidence))
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """Create state when synthesis fails."""
        updated_state = state.copy()
        updated_state.update({
            "final_answer": config.SYNTHESIS_ERROR_MESSAGE,
            "citations": [],
            "answer_confidence": 0.0,
            "step_count": state.get("step_count", 0) + 1,
            "error_info": {
                **state.get("error_info", {}),
                "synthesis_error": str(error)
            }
        })
        return updated_state


# Create service instance
synthesis_service = SynthesisService()


@AgentRegistry.register(
    name="synthesis",
    capabilities=["response_synthesis", "answer_generation"],
    priority=10
)
def synthesis_agent(state: GraphState) -> GraphState:
    """
    Synthesis agent that generates the final answer using Gemini Pro.
    
    This agent combines context from document retrieval and web search to
    create comprehensive, well-cited responses.
    """
    # Note: LangGraph expects sync functions, so we need to handle async
    import asyncio
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async synthesis
    return loop.run_until_complete(synthesis_service.synthesize_response(state))
