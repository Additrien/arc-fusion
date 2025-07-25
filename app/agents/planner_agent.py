"""
Planner Agent - LLM-based orchestration for dynamic agent execution.

This agent replaces the rule-based orchestrator with an LLM that dynamically
plans which agents to execute and in what order based on the user's query
and current state.
"""

import os
from typing import Dict, Any, List
from google import genai
from google.genai import types
from .registry import AgentRegistry
from .state import GraphState
from ..utils.logger import get_logger
from .. import config

logger = get_logger('arc_fusion.agents.planner')


class PlannerService:
    """Service for LLM-based planning of agent execution."""
    
    def __init__(self):
        # Use primary model for planning
        self.model = config.PRIMARY_MODEL
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
    def plan_next_steps(self, state: GraphState) -> GraphState:
        """
        Use LLM to plan the next steps in the agent workflow.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with planned tasks
        """
        session_id = state["session_id"]
        logger.info(f"Planning next steps for session {session_id}")
        
        try:
            # Create prompt for planning
            prompt = self._create_planning_prompt(state)
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking for speed
                )
            )
            
            # Parse the response to extract plan
            plan = self._parse_planning_response(response.text)
            
            logger.info(f"Generated plan: {plan}")
            
            # Update state with the plan
            updated_state = state.copy()
            updated_state.update({
                "tasks_to_run": plan.get("tasks", []),
                "tasks_completed": plan.get("completed_tasks", state.get("tasks_completed", [])),
                "step_count": state.get("step_count", 0) + 1,
                "planning_reasoning": plan.get("reasoning", "")
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in planning: {str(e)}")
            # Fallback to default plan
            return self._create_fallback_plan(state)
    
    def _create_planning_prompt(self, state: GraphState) -> str:
        """Create the prompt for planning agent execution."""
        # Get available agents and their capabilities
        agents_info = self._get_agents_info()
        
        # Get current state information
        query = state.get("query", "")
        intent = state.get("intent", "unknown")
        retrieved_context = state.get("retrieved_context", [])
        web_context = state.get("web_context", [])
        tasks_completed = state.get("tasks_completed", [])
        
        return f"""
You are an expert AI agent planner orchestrating a multi-agent system for a RAG (Retrieval-Augmented Generation) application. Your job is to dynamically plan which agents to execute and in what order to best satisfy the user's query.

**Available Agents and Their Capabilities:**
{agents_info}

**Current State:**
- User Query: "{query}"
- Classified Intent: {intent}
- Tasks Already Completed: {tasks_completed}
- Retrieved Document Context Available: {"Yes" if retrieved_context else "No"}
- Web Search Context Available: {"Yes" if web_context else "No"}

**Planning Rules:**
1. ONLY use the exact agent names from the list above
2. Plan the minimal set of agents needed to answer the query
3. Consider dependencies between agents (e.g., you typically need retrieval or search before synthesis)
4. You can add web_search as a fallback if document retrieval quality is low
5. Use clarification agent only for truly ambiguous queries
6. End with synthesis agent to generate the final response

**Response Format:**
PLAN: <comma-separated list of agent names in execution order>
REASONING: <brief explanation of why this plan was chosen>
CONFIDENCE: <0-1 score indicating confidence in this plan>

Example Response:
PLAN: corpus_retrieval,synthesis
REASONING: The query can be answered from academic papers, so we'll retrieve relevant documents and synthesize an answer.
CONFIDENCE: 0.95
"""
    
    def _get_agents_info(self) -> str:
        """Get formatted information about available agents."""
        agents = AgentRegistry.get_all_agents()
        info_lines = []
        
        for agent_name, config in agents.items():
            capabilities = ", ".join(config['capabilities'])
            info_lines.append(f"  - {agent_name}: {capabilities}")
        
        return "\n".join(info_lines)
    
    def _parse_planning_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the plan."""
        plan = {
            "tasks": [],
            "reasoning": "",
            "confidence": 0.8
        }
        
        lines = response_text.strip().split('\n')
        
        for line in lines:
            if line.startswith('PLAN:'):
                tasks_str = line.split(':', 1)[1].strip()
                plan["tasks"] = [task.strip() for task in tasks_str.split(',') if task.strip()]
            elif line.startswith('REASONING:'):
                plan["reasoning"] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    plan["confidence"] = float(line.split(':', 1)[1].strip())
                except (ValueError, IndexError):
                    pass
        
        return plan
    
    def _create_fallback_plan(self, state: GraphState) -> GraphState:
        """Create a fallback plan when LLM planning fails."""
        intent = state.get("intent", "retrieve_corpus")
        
        # Default plans based on intent
        default_plans = {
            "retrieve_corpus": ["corpus_retrieval", "synthesis"],
            "search_web": ["web_search", "synthesis"],
            "corpus_and_web_search": ["corpus_retrieval", "web_search", "synthesis"],
            "clarify": ["clarification"],
            "end": []
        }
        
        tasks = default_plans.get(intent, ["corpus_retrieval", "synthesis"])
        
        updated_state = state.copy()
        updated_state.update({
            "tasks_to_run": tasks,
            "tasks_completed": state.get("tasks_completed", []),
            "step_count": state.get("step_count", 0) + 1,
            "planning_reasoning": "Fallback plan due to planning error",
            "error_info": {
                **state.get("error_info", {}),
                "planning_error": "Using fallback plan"
            }
        })
        
        logger.warning(f"Using fallback plan: {tasks}")
        return updated_state


# Create service instance
planner_service = PlannerService()


@AgentRegistry.register(
    name="planner",
    capabilities=["task_planning", "workflow_orchestration"],
    priority=5
)
def planner_agent(state: GraphState) -> GraphState:
    """
    LLM-based planner that dynamically determines which agents to execute.
    
    This agent replaces the rule-based orchestrator with intelligent planning
    using the primary LLM model.
    """
    return planner_service.plan_next_steps(state)
