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
from ..prompts import (
    CLARIFICATION_PROMPT_BASIC, CLARIFICATION_ADVANCED_PROMPT
)

logger = get_logger('arc_fusion.agents.clarification')

# Configure Gemini
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class ClarificationService:
    """Enhanced service for context-aware clarification with specific options."""
    
    def __init__(self):
        # Use routing model for fast clarification analysis
        self.model = config.ROUTING_MODEL
        
        # Generation config for precise, thoughtful responses
        self.generation_config = {
            "temperature": config.CLARIFICATION_TEMP,
            "top_p": config.CLARIFICATION_TOP_P,
            "max_output_tokens": config.CLARIFICATION_MAX_TOKENS,
            "thinking_config": {
                "thinking_budget": 0
            }
        }
    
    def generate_clarification(self, state: GraphState) -> GraphState:
        """
        Advanced clarification that analyzes ambiguity and provides specific options.
        
        Args:
            state: Current graph state with the ambiguous query
            
        Returns:
            Updated state with enhanced clarification analysis and options
        """
        query = state["query"]
        session_id = state["session_id"]
        clarification_round = state.get("clarification_round", 0)
        
        logger.info(f"Generating advanced clarification (round {clarification_round + 1}) for session {session_id}")
        
        try:
            # Analyze what context is available to inform clarification
            context_analysis = self._analyze_available_context(state)
            
            # Perform ambiguity analysis
            ambiguity_analysis = self._analyze_ambiguity(query, context_analysis)
            
            # Generate clarification with context awareness
            clarification_result = self._generate_contextual_clarification(
                query, context_analysis, ambiguity_analysis, clarification_round
            )
            
            logger.info("Advanced clarification generated successfully")
            
            # Record observation for ReAct system
            observation = {
                "agent": "clarification",
                "status": "clarification_provided",
                "details": f"Identified {len(ambiguity_analysis.get('ambiguous_terms', []))} ambiguous terms",
                "ambiguity_types": ambiguity_analysis.get("types", []),
                "clarification_strategy": clarification_result.get("strategy", "unknown")
            }
            
            # Update state with enhanced clarification information
            updated_state = state.copy()
            updated_state.update({
                "final_answer": clarification_result["response"],
                "intent": "clarify_response",
                "requires_clarification": True,
                "step_count": state.get("step_count", 0) + 1,
                "citations": [],
                "answer_confidence": config.CLARIFICATION_RESPONSE_CONFIDENCE,
                "clarification_round": clarification_round + 1,
                "ambiguity_analysis": ambiguity_analysis,
                "clarification_options": clarification_result.get("options", []),
                "context_summary": context_analysis.get("summary", ""),
                "disambiguation_context": context_analysis.get("disambiguation_hints", []),
                "observations": state.get("observations", []) + [observation]
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in advanced clarification generation: {str(e)}")
            return self._create_error_state(state, e)
    
    def _analyze_available_context(self, state: GraphState) -> Dict[str, Any]:
        """Analyze what context is available to inform clarification."""
        # Check conversation history for patterns
        conversation_messages = state.get("conversation_messages", [])
        retrieved_context = state.get("retrieved_context", [])
        document_sources = state.get("document_sources", [])
        
        # Extract topics and patterns from conversation
        topic_patterns = self._extract_conversation_patterns(conversation_messages)
        
        # Check if we have relevant documents that could help clarify
        available_papers = [doc.get("filename", "") for doc in document_sources] if document_sources else []
        
        # Summarize available context
        context_summary = f"Conversation history: {len(conversation_messages)} messages"
        if available_papers:
            context_summary += f", Available papers: {len(available_papers)}"
        if retrieved_context:
            context_summary += f", Retrieved context: {len(retrieved_context)} chunks"
        
        return {
            "conversation_patterns": topic_patterns,
            "available_papers": available_papers,
            "has_context": len(retrieved_context) > 0,
            "summary": context_summary,
            "disambiguation_hints": self._generate_disambiguation_hints(available_papers)
        }
    
    def _extract_conversation_patterns(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract topic patterns from conversation history."""
        patterns = []
        for msg in messages[-config.CLARIFICATION_CONTEXT_MESSAGES:]:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                # Simple pattern extraction - could be enhanced with NLP
                if "methodology" in content or "method" in content:
                    patterns.append("methodology_focus")
                if "comparison" in content or "compare" in content:
                    patterns.append("comparison_focus")
                if "accuracy" in content or "performance" in content:
                    patterns.append("performance_focus")
        return patterns
    
    def _generate_disambiguation_hints(self, available_papers: List[str]) -> List[Dict[str, Any]]:
        """Generate hints based on available papers to help disambiguation."""
        hints = []
        for paper in available_papers:
            if "zhang" in paper.lower():
                hints.append({
                    "type": "paper_reference",
                    "value": "Zhang et al.",
                    "context": f"Available in {paper}"
                })
            # Add more paper-specific hints as needed
        return hints
    
    def _analyze_ambiguity(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what makes the query ambiguous."""
        ambiguous_terms = []
        ambiguity_types = []
        
        query_lower = query.lower()
        
        # Detect vague quantifiers
        for term in config.VAGUE_QUANTIFIERS:
            if term in query_lower:
                ambiguous_terms.append(term)
                ambiguity_types.append("vague_quantifier")
        
        # Detect undefined referents
        for ref in config.UNDEFINED_REFERENTS:
            if f" {ref} " in f" {query_lower} ":
                ambiguous_terms.append(ref)
                ambiguity_types.append("undefined_referent")
        
        # Detect comparative terms without clear comparison
        for comp in config.UNCLEAR_COMPARISONS:
            if comp in query_lower:
                ambiguous_terms.append(comp)
                ambiguity_types.append("unclear_comparison")
        
        # Detect domain ambiguity
        if any(domain in query_lower for domain in ["model", "method", "approach", "technique"]):
            if not any(specific in query_lower for specific in ["llm", "neural", "transformer", "gpt"]):
                ambiguity_types.append("domain_ambiguity")
        
        return {
            "ambiguous_terms": ambiguous_terms,
            "types": list(set(ambiguity_types)),
            "complexity_score": len(ambiguous_terms) + len(set(ambiguity_types))
        }
    
    def _generate_contextual_clarification(self, query: str, context: Dict[str, Any], 
                                          ambiguity: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """Generate contextual clarification with specific options."""
        
        prompt = self._create_advanced_clarification_prompt(query, context, ambiguity, round_num)
        
        # Generate clarification using Gemini
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**self.generation_config)
        )
        
        # Parse response to extract structured clarification
        parsed = self._parse_clarification_response(response.text)
        
        return {
            "response": parsed.get("response", response.text),
            "options": parsed.get("options", []),
            "strategy": parsed.get("strategy", "general_clarification")
        }
    
    def _create_advanced_clarification_prompt(self, query: str, context: Dict[str, Any], 
                                            ambiguity: Dict[str, Any], round_num: int) -> str:
        """Create advanced clarification prompt with context awareness."""
        
        ambiguous_terms = ambiguity.get("ambiguous_terms", [])
        ambiguity_types = ambiguity.get("types", [])
        available_papers = context.get("available_papers", [])
        conversation_patterns = context.get("conversation_patterns", [])
        
        context_info = f"""
**Available Context:**
- Papers in knowledge base: {len(available_papers)}
- Conversation patterns: {', '.join(conversation_patterns) if conversation_patterns else 'None detected'}
- Available papers: {', '.join(available_papers[:config.CLARIFICATION_MAX_PAPERS_TO_SHOW]) if available_papers else 'None'}
{'... and more' if len(available_papers) > config.CLARIFICATION_MAX_PAPERS_TO_SHOW else ''}

**Ambiguity Analysis:**
- Ambiguous terms detected: {', '.join(ambiguous_terms) if ambiguous_terms else 'None specific'}
- Ambiguity types: {', '.join(ambiguity_types) if ambiguity_types else 'General vagueness'}
- Clarification round: {round_num + 1}
"""
        
        return f"""
You are an expert research assistant specialized in disambiguating academic queries. A user has asked a question that needs clarification to provide an accurate answer.

**User Query:** "{query}"

{context_info}

**Your Task:**
Provide a helpful clarification that:
1. Acknowledges what you understand from their query
2. Identifies the specific ambiguities that need resolution
3. Offers 2-3 concrete, specific options when possible
4. Uses available context to guide clarification

**Clarification Strategies:**
- If papers are available: Reference specific papers they might be asking about
- If vague quantifiers: Ask for specific thresholds or ranges
- If unclear comparisons: Ask what they're comparing against
- If undefined referents: Ask what specific entity they're referring to

**Response Format:**
Provide a natural, conversational clarification response. If you can offer specific options, structure them clearly.

**Example Good Response:**
"I can help you with that! I see you're asking about accuracy requirements. Based on the papers in our knowledge base, I found discussions about accuracy in different contexts. Are you asking about:

1. **Few-shot learning accuracy** (Zhang et al. discusses 3-5 examples achieving 85%+ accuracy)
2. **Training dataset size** for fine-tuning (Smith et al. suggests 1000+ examples for stable training)
3. **Evaluation accuracy** benchmarks for a specific task

Also, what accuracy threshold are you targeting - research-grade (90%+) or production-ready (95%+)?"

Generate your clarification response now:
"""
    
    def _parse_clarification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the clarification response to extract structured information."""
        # Simple parsing - could be enhanced with more sophisticated NLP
        lines = response_text.strip().split('\n')
        
        options = []
        current_option = None
        
        for line in lines:
            # Look for numbered or bulleted options
            if any(line.strip().startswith(marker) for marker in ['1.', '2.', '3.', '- ', '• ']):
                if current_option:
                    options.append(current_option)
                current_option = line.strip()
            elif current_option and line.strip():
                current_option += " " + line.strip()
        
        if current_option:
            options.append(current_option)
        
        return {
            "response": response_text,
            "options": options[:config.CLARIFICATION_MAX_OPTIONS],
            "strategy": "contextual_options" if options else "general_clarification"
        }
    
    def _create_clarification_prompt(self, query: str) -> str:
        """
        Create a prompt for generating clarifying questions using centralized prompts.
        
        Args:
            query: The ambiguous user query
            
        Returns:
            Formatted prompt for clarification generation
        """
        return CLARIFICATION_PROMPT_BASIC.format(query=query)
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """Create state when clarification fails."""
        updated_state = state.copy()
        updated_state.update({
            "final_answer": config.CLARIFICATION_ERROR_MESSAGE,
            "requires_clarification": True,
            "step_count": state.get("step_count", 0) + 1,
            "citations": [],
            "answer_confidence": config.CLARIFICATION_ERROR_CONFIDENCE,
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
