"""
Web Search Agent - External information retrieval using Tavily API.

This agent handles queries that require current or external information not
available in the document corpus, using Tavily's search API optimized for RAG.
"""

import os
import json
from typing import Dict, Any, List
import httpx
from google import genai
from google.genai import types

from .registry import AgentRegistry
from .state import GraphState
from ..utils.logger import get_logger
from ..prompts import WEB_SEARCH_OPTIMIZATION_PROMPT
from .. import config

logger = get_logger('arc_fusion.agents.web_search')

# Configure Gemini for query optimization
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


class WebSearchService:
    """Service for web search using Tavily API."""
    
    def __init__(self):
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set - web search will be disabled")
        
        # Use Gemini Flash for query optimization
        self.query_optimizer_model = config.WEB_SEARCH_QUERY_MODEL
        
        # Tavily API configuration
        self.tavily_base_url = config.TAVILY_API_URL
        self.max_results = config.TAVILY_MAX_RESULTS
        
    async def search_web(self, state: GraphState) -> GraphState:
        """
        Perform web search for external information.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with web search results
        """
        query = state["query"]
        session_id = state["session_id"]
        
        logger.info(f"Starting web search for session {session_id}")
        
        if not self.tavily_api_key:
            return self._create_disabled_state(state)
        
        try:
            # Step 1: Optimize query for web search
            logger.info("Optimizing query for web search")
            optimized_query = await self._optimize_search_query(query)
            
            # Step 2: Perform Tavily search
            logger.info(f"Searching web with query: {optimized_query}")
            search_results = await self._tavily_search(optimized_query)
            
            if not search_results:
                logger.warning("No web search results found")
                return self._create_empty_result_state(state)
            
            # Step 3: Process and format results
            logger.info("Processing web search results")
            web_context_list, web_sources = self._process_search_results(search_results)
            
            logger.info(f"Retrieved web context from {len(web_sources)} sources")
            
            # Update state
            updated_state = state.copy()
            updated_state.update({
                "web_context": web_context_list,
                "web_sources": web_sources,
                "search_query": optimized_query,
                "step_count": state.get("step_count", 0) + 1,
                "intent": "synthesize",  # Keep for compatibility, but orchestrator will manage flow
                "tasks_completed": state.get("tasks_completed", []) + ["web_search"]
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return self._create_error_state(state, e)
    
    async def _optimize_search_query(self, query: str) -> str:
        """
        Optimize the user query for web search using centralized prompts.
        
        Transform the query into terms that are more likely to return
        relevant results from web search engines.
        """
        optimization_prompt = WEB_SEARCH_OPTIMIZATION_PROMPT.format(query=query)
        
        try:
            response = client.models.generate_content(
                model=self.query_optimizer_model,
                contents=optimization_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            optimized = response.text.strip().replace('"', '').replace("'", "")
            
            logger.debug(f"Query optimized: '{query}' -> '{optimized}'")
            return optimized
            
        except Exception as e:
            logger.warning(f"Query optimization failed, using original: {str(e)}")
            return query
    
    async def _tavily_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform search using Tavily API.
        
        Args:
            query: Search query
            
        Returns:
            List of search results from Tavily
        """
        
        search_data = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",  # More comprehensive search
            "include_answer": True,      # Get AI-generated answer
            "include_raw_content": False, # Don't need full HTML
            "max_results": self.max_results,
            "include_domains": [],
            "exclude_domains": []
        }
        
        async with httpx.AsyncClient(timeout=config.TAVILY_TIMEOUT) as client:
            try:
                response = await client.post(
                    f"{self.tavily_base_url}/search",
                    json=search_data,
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"Tavily search returned {len(result.get('results', []))} results")
                
                return result.get('results', [])
                
            except httpx.TimeoutException:
                logger.error("Tavily search timed out")
                raise Exception("Web search timed out")
            except httpx.HTTPStatusError as e:
                logger.error(f"Tavily API error: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Web search API error: {e.response.status_code}")
            except Exception as e:
                logger.error(f"Unexpected error in Tavily search: {str(e)}")
                raise
    
    def _process_search_results(self, results: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Process and format Tavily search results.
        
        Args:
            results: Raw results from Tavily API
            
        Returns:
            Tuple of (list_of_context_strings, source_metadata)
        """
        
        web_context_list = []
        web_sources = []
        
        for i, result in enumerate(results):
            title = result.get('title', 'No Title')
            content = result.get('content', '')
            url = result.get('url', '')
            score = result.get('score', 0.0)
            
            if content and len(content.strip()) > 0:
                web_context_list.append(content)
                
                web_sources.append({
                    "title": title,
                    "url": url,
                    "content": content, # Store the full content for citation
                    "score": score,
                    "index": i + 1
                })
        
        return web_context_list, web_sources
    
    def _create_disabled_state(self, state: GraphState) -> GraphState:
        """Create state when web search is disabled (no API key)."""
        updated_state = state.copy()
        updated_state.update({
            "web_context": [config.WEB_SEARCH_DISABLED_MESSAGE],
            "web_sources": [],
            "search_query": None,
            "step_count": state.get("step_count", 0) + 1,
            "intent": "synthesize",
            "error_info": {
                "web_search_error": "Tavily API key not configured"
            }
        })
        return updated_state
    
    def _create_empty_result_state(self, state: GraphState) -> GraphState:
        """Create state when no web results are found."""
        updated_state = state.copy()
        updated_state.update({
            "web_context": [],
            "web_sources": [],
            "search_query": state["query"],
            "step_count": state.get("step_count", 0) + 1,
            "intent": "synthesize"
        })
        return updated_state
    
    def _create_error_state(self, state: GraphState, error: Exception) -> GraphState:
        """Create state when an error occurs."""
        updated_state = state.copy()
        updated_state.update({
            "web_context": [config.WEB_SEARCH_ERROR_MESSAGE],
            "web_sources": [],
            "search_query": None,
            "step_count": state.get("step_count", 0) + 1,
            "error_info": {
                "web_search_error": str(error)
            },
            "intent": "synthesize"
        })
        return updated_state


# Create service instance
web_search_service = WebSearchService()


@AgentRegistry.register(
    name="web_search",
    capabilities=["web_search", "external_search"],
    priority=10
)
def web_search_agent(state: GraphState) -> GraphState:
    """
    Web search agent for retrieving external/current information using Tavily API.
    
    This agent handles queries that require information not available in the document corpus.
    """
    # Note: LangGraph expects sync functions, so we need to handle async
    import asyncio
    
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, use run_coroutine_threadsafe
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(
            web_search_service.search_web(state),
            loop
        )
        return future.result()
    except RuntimeError:
        # No running loop, safe to use run_until_complete
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(web_search_service.search_web(state))
