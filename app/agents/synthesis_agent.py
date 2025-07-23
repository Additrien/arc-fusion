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

logger = get_logger('arc_fusion.agents.synthesis')

# Configure Gemini
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class SynthesisService:
    """Service for generating final answers using retrieved context."""
    
    def __init__(self):
        # Use Gemini 2.5 Pro for high-quality response synthesis
        self.synthesis_model = 'gemini-2.5-pro'
        
        # Generation configuration for balanced quality and creativity
        self.generation_config = {
            "temperature": 0.1,  # Low temperature for factual accuracy
            "top_p": 0.8,
            "max_output_tokens": 2048,
        }
        
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
            
            # Create synthesis prompt
            prompt = self._create_synthesis_prompt(query, context_info)
            
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
            "web_content": None,
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
        web_context = state.get("web_context")
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
    
    def _create_synthesis_prompt(self, query: str, context_info: Dict[str, Any]) -> str:
        """
        Create the synthesis prompt for Gemini Pro.
        
        Args:
            query: User's original query
            context_info: Organized context information
            
        Returns:
            Formatted prompt for response generation
        """
        
        prompt_parts = [
            "You are an expert research assistant. Your task is to provide a comprehensive, accurate answer to the user's question using the provided context."
        ]
        
        # Add context sections
        if context_info["has_document_context"]:
            prompt_parts.extend([
                "\n## DOCUMENT CONTEXT",
                "The following information is from academic papers and documents in our database:"
            ])
            
            for i, chunk in enumerate(context_info["document_chunks"], 1):
                source = context_info["document_sources"][i-1] if i-1 < len(context_info["document_sources"]) else {}
                filename = source.get("filename", "Unknown Document")
                prompt_parts.append(f"\n**Document {i}** ({filename}):\n{chunk}")
        
        if context_info["has_web_context"]:
            prompt_parts.extend([
                "\n## WEB SEARCH CONTEXT",
                "The following information is from recent web search results:",
                f"\n{context_info['web_content']}"
            ])
        
        # Add guidelines
        prompt_parts.extend([
            f"\n## USER QUESTION",
            f"{query}",
            "\n## INSTRUCTIONS",
            "1. Provide a comprehensive answer based on the available context",
            "2. Be specific and cite information when referencing sources",
            "3. If information is conflicting, acknowledge the discrepancy",
            "4. If the context doesn't fully answer the question, say so explicitly",
            "5. For academic content, be precise about methodologies, results, and conclusions",
            "6. Use clear, professional language appropriate for the topic",
            "",
            "IMPORTANT: Only use information from the provided context. Do not add information from your training data that is not supported by the context."
        ])
        
        # Handle empty context case
        if not context_info["has_document_context"] and not context_info["has_web_context"]:
            prompt_parts.extend([
                "\n## NO CONTEXT AVAILABLE",
                "No relevant information was found in our documents or web search.",
                "Provide a helpful response explaining that the information is not available and suggest alternative approaches."
            ])
        
        return "\n".join(prompt_parts)
    
    def _generate_citations(self, context_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate citations for the sources used.
        
        Args:
            context_info: Context information with sources
            
        Returns:
            List of citation objects
        """
        citations = []
        
        # Document citations
        for i, source in enumerate(context_info.get("document_sources", []), 1):
            # Convert document_id to string in case it's a UUID object
            document_id = source.get("document_id", "")
            if document_id:
                document_id = str(document_id)
            
            citations.append({
                "type": "document",
                "id": f"doc_{i}",
                "filename": source.get("filename", "Unknown Document"),
                "document_id": document_id,
                "score": str(source.get("score", 0.0))
            })
        
        # Web citations
        for i, source in enumerate(context_info.get("web_sources", []), 1):
            citations.append({
                "type": "web",
                "id": f"web_{i}",
                "title": source.get("title", "Unknown Title"),
                "url": source.get("url", ""),
                "score": str(source.get("score", 0.0))
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
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on context availability
        if context_info["has_document_context"]:
            confidence += 0.2
            # More sources = higher confidence
            doc_count = len(context_info["document_sources"])
            confidence += min(0.1 * doc_count, 0.2)
        
        if context_info["has_web_context"]:
            confidence += 0.15
            # More sources = higher confidence
            web_count = len(context_info["web_sources"])
            confidence += min(0.05 * web_count, 0.15)
        
        # Reduce confidence for short answers (might indicate insufficient info)
        if len(answer) < 100:
            confidence -= 0.1
        
        # Reduce confidence if answer mentions lack of information
        uncertainty_phrases = ["not available", "insufficient information", "cannot determine", "unclear"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """Create state when synthesis fails."""
        updated_state = state.copy()
        updated_state.update({
            "final_answer": "I apologize, but I encountered an error while processing your request. Please try rephrasing your question or try again.",
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