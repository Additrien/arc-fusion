"""
Routing Agent - The entry point that classifies user intent.

This agent uses Gemini 2.5 Flash Lite (fast, cost-effective) to analyze the user's
query and determine the appropriate next action in our multi-agent pipeline.
"""

import os
from typing import Dict, Any
from google import genai
from google.genai import types
from .registry import AgentRegistry
from .state import GraphState
from ..utils.logger import get_logger
from .. import config
from ..prompts import ROUTING_ANALYSIS_PROMPT, format_conversation_history

logger = get_logger('arc_fusion.agents.routing')

# Configure Gemini with new API
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class RoutingService:
    """Service for query analysis and intent routing."""
    
    def __init__(self):
        # Use routing model from central config with new API
        self.model = config.ROUTING_MODEL
        
    def analyze_query(self, state: GraphState) -> GraphState:
        """
        Analyze user query and determine routing intent.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with intent and routing information
        """
        query = state["query"]
        session_id = state["session_id"]
        
        logger.info(f"Analyzing query intent for session {session_id}")
        
        try:
            # Get conversation history for context and ensure it's in the right format
            conversation_history = state.get("conversation_messages", [])
            
            # Debug: Log the exact types and structure we're getting
            logger.info(f"conversation_history type: {type(conversation_history)}")
            for i, msg in enumerate(conversation_history[:3]):  # First 3 messages
                logger.info(f"Message {i}: type={type(msg)}, has_content={hasattr(msg, 'content')}, has_type={hasattr(msg, 'type')}, is_dict={isinstance(msg, dict)}")
                if hasattr(msg, '__dict__'):
                    logger.info(f"Message {i} attrs: {list(msg.__dict__.keys())}")
                if isinstance(msg, dict):
                    logger.info(f"Message {i} dict keys: {list(msg.keys())}")
            
            # Defensive handling: convert any LangGraph message objects to plain dicts
            safe_conversation_history = []
            for i, msg in enumerate(conversation_history):
                try:
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
                        logger.warning(f"Skipping unknown message type at index {i}: {type(msg)}")
                        continue
                except Exception as e:
                    logger.error(f"Error processing message at index {i}: {e}")
                    continue
            
            conversation_history = safe_conversation_history
            
            # Create prompt for intent classification with conversation context
            prompt = self._create_routing_prompt(query, conversation_history)
            
            # Generate response using Gemini Flash with new API
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking for speed
                )
            )
            
            # Parse the response to extract intent
            intent = self._parse_intent_response(response.text)
            
            logger.info(f"Intent classified as: {intent}")
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "intent": intent,
                "step_count": state.get("step_count", 0) + 1,
                "confidence_score": self._calculate_confidence(response.text)
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in routing analysis: {str(e)}")
            # Default to corpus retrieval on error
            updated_state = state.copy()
            updated_state.update({
                "intent": "retrieve_corpus",
                "step_count": state.get("step_count", 0) + 1,
                "error_info": {
                    "routing_error": str(e)
                }
            })
            return updated_state
    
    def _create_routing_prompt(self, query: str, conversation_history: list = None) -> str:
        """Create the prompt for intent classification using centralized prompts."""
        formatted_history = format_conversation_history(conversation_history) if conversation_history else ""
        
        return ROUTING_ANALYSIS_PROMPT.format(
            conversation_history=formatted_history,
            query=query
        )
    
    def _parse_intent_response(self, response_text: str) -> str:
        """Parse the LLM response to extract intent."""
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            if line.startswith('INTENT:'):
                intent = line.split(':', 1)[1].strip().lower()
                # Validate intent
                valid_intents = ['retrieve_corpus', 'search_web', 'corpus_and_web_search', 'clarify', 'end']
                if intent in valid_intents:
                    return intent
        
        # Default fallback
        logger.warning(f"Could not parse intent from response: {response_text[:200]}")
        return "retrieve_corpus"
    
    def _calculate_confidence(self, response_text: str) -> float:
        """Extract confidence score from LLM response."""
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            if line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                    return min(max(confidence, 0.0), 1.0)  # Clamp between 0-1
                except (ValueError, IndexError):
                    pass
        
        # Default confidence
        return 0.8


# Create service instance
routing_service = RoutingService()


@AgentRegistry.register(
    name="routing",
    capabilities=["intent_classification", "query_analysis"],
    priority=10
)
def routing_agent(state: GraphState) -> GraphState:
    """
    Entry point agent that classifies user queries and determines routing.
    
    This agent uses Gemini 2.5 Flash Lite for fast, cost-effective intent classification.
    """
    return routing_service.analyze_query(state)
