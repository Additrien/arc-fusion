"""
Clarification Agent - Handles ambiguous queries by asking clarifying questions.

This agent analyzes queries that have been flagged as ambiguous by the routing agent
and generates targeted clarifying questions to help users provide the missing context.
"""

import os
from typing import Dict, Any, List
from google import genai
from google.genai import types

from .registry import AgentRegistry
from .state import GraphState
from ..utils.logger import get_logger
from .. import config

logger = get_logger('arc_fusion.agents.clarification')

# Configure Gemini
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class ClarificationService:
    """Service for generating clarifying questions for ambiguous queries."""
    
    def __init__(self):
        # Use routing model for fast clarification analysis
        self.model = config.ROUTING_MODEL
        
        # Generation config for precise, short responses
        self.generation_config = {
            "temperature": 0.1,  # Low temperature for consistent clarification
            "top_p": 0.8,
            "max_output_tokens": 300,  # Short, focused questions
            "thinking_config": {
                "thinking_budget": 0
            }
        }
    
    def generate_clarification(self, state: GraphState) -> GraphState:
        """
        Analyze the ambiguous query and generate clarifying questions.
        
        Args:
            state: Current graph state with the ambiguous query
            
        Returns:
            Updated state with clarifying questions as the final answer
        """
        query = state["query"]
        session_id = state["session_id"]
        
        logger.info(f"Generating clarification for ambiguous query in session {session_id}")
        
        try:
            # Create clarification prompt
            prompt = self._create_clarification_prompt(query)
            
            # Generate clarifying questions using Gemini
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**self.generation_config)
            )
            
            clarification_response = response.text.strip()
            
            logger.info("Clarification questions generated successfully")
            
            # Update state with clarification response
            updated_state = state.copy()
            updated_state.update({
                "final_answer": clarification_response,
                "intent": "clarify_response",  # Mark as clarification response
                "requires_clarification": True,
                "step_count": state.get("step_count", 0) + 1,
                "citations": [],  # No citations for clarification
                "answer_confidence": 0.9  # High confidence in asking the right questions
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in clarification generation: {str(e)}")
            return self._create_error_state(state, e)
    
    def _create_clarification_prompt(self, query: str) -> str:
        """
        Create a prompt for generating clarifying questions.
        
        Args:
            query: The ambiguous user query
            
        Returns:
            Formatted prompt for clarification generation
        """
        
        return f"""
You are a helpful research assistant. A user has asked a question that is too vague or ambiguous to answer directly. Your job is to ask 2-3 specific clarifying questions that will help you understand what they really want to know.

The ambiguous query is: "{query}"

Common types of ambiguity to address:
1. **Missing Context**: What domain, dataset, or specific area are they asking about?
2. **Vague Terms**: What do they mean by "best", "enough", "better", "it", etc.?
3. **Missing Scope**: Are they asking about a specific paper, method, time period, or comparison?
4. **Undefined Referents**: What does "this", "that", "it", or "they" refer to?

Your response should:
- Be friendly and helpful
- Ask 2-3 specific, targeted questions
- Explain why the additional information is needed
- Provide examples when helpful

Format your response as a natural conversation, not as a numbered list.

Example good clarification:
"I'd be happy to help you with that! To give you the most accurate answer, I need a bit more context. Are you asking about examples for training a specific type of model (like text-to-SQL, question answering, etc.)? And when you say 'good accuracy,' what accuracy threshold or benchmark are you targeting? Also, are you working with a particular dataset or domain?"

Generate a helpful clarification response now:
"""
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """Create state when clarification fails."""
        updated_state = state.copy()
        updated_state.update({
            "final_answer": "I'm sorry, but I'm having trouble understanding your question. Could you please provide more specific details about what you're looking for?",
            "requires_clarification": True,
            "step_count": state.get("step_count", 0) + 1,
            "citations": [],
            "answer_confidence": 0.3,
            "error_info": {
                **state.get("error_info", {}),
                "clarification_error": str(error)
            }
        })
        return updated_state


# Create service instance
clarification_service = ClarificationService()


@AgentRegistry.register(
    name="clarification",
    capabilities=["query_clarification", "ambiguity_resolution"],
    priority=10
)
def clarification_agent(state: GraphState) -> GraphState:
    """
    Clarification agent that handles ambiguous queries by asking clarifying questions.
    
    This agent is triggered when the routing agent determines that a query is too
    vague or ambiguous to process directly.
    """
    return clarification_service.generate_clarification(state) 