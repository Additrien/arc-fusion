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
            # Create prompt for intent classification
            prompt = self._create_routing_prompt(query)
            
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
    
    def _create_routing_prompt(self, query: str) -> str:
        """Create the prompt for intent classification."""
        
        return f"""
You are a query analysis agent for an AI research assistant. Your job is to classify user queries into the appropriate action category.

Analyze this user query and classify it into ONE of these categories:

**retrieve_corpus**: Query can be answered from academic papers in our database
- Examples: "What did Zhang et al. report about prompt templates?", "Which method achieved highest accuracy in the Spider dataset?", "Explain the methodology used in paper X"
- ALSO INCLUDES general topic questions that could be found in academic papers: "What are SQL challenges?", "What methods exist for text-to-SQL?", "How does neural machine translation work?"

**search_web**: Query requires current/external information not in papers
- When user explicitly requests web search: "Search online for...", "Look up on the web..."

**corpus_and_web_search**: Query explicitly asks to combine internal knowledge with web search results.
- Examples: "Compare the findings in your documents with what's on the web about text-to-sql", "What are text-to-SQL challenges based on your internal knowledge and your search on internet?"

**clarify**: Query is too vague or ambiguous to process AND lacks sufficient context
- Examples: "How many examples are enough?" (without specifying for what task), "What's the best method?" (without domain context), "How does it work?" (without specifying what "it" refers to)
- Missing critical context that makes the query unanswerable

**end**: Conversation ending or greeting
- Examples: "thanks", "goodbye", "that's all", "hello"

IMPORTANT: General topic questions about research domains should be classified as "retrieve_corpus", not "clarify". Only use "clarify" for genuinely ambiguous queries that lack essential context.

User Query: "{query}"

Respond with ONLY the category name (retrieve_corpus, search_web, corpus_and_web_search, clarify, or end) followed by a confidence score (0-1).

Format: INTENT: <category>
CONFIDENCE: <score>
REASONING: <brief explanation>
"""
    
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