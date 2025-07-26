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
    """Enhanced service for ReAct-based planning with iterative reasoning."""
    
    def __init__(self):
        # Use primary model for planning
        self.model = config.PRIMARY_MODEL
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        
    def plan_next_steps(self, state: GraphState) -> GraphState:
        """
        Enhanced ReAct planner that can observe results and iteratively refine plans.
        
        Args:
            state: Current graph state with observations from previous actions
            
        Returns:
            Updated state with refined plan based on observations
        """
        session_id = state["session_id"]
        plan_iteration = state.get("plan_iterations", 0)
        max_iterations = state.get("max_plan_iterations", 3)
        
        logger.info(f"ReAct Planning iteration {plan_iteration + 1} for session {session_id}")
        
        # Initialize ReAct state if first time
        if plan_iteration == 0:
            state = self._initialize_react_state(state)
        
        try:
            # Check if we need to replan based on observations
            needs_replanning = self._analyze_observations(state)
            
            if needs_replanning and plan_iteration < max_iterations:
                # Create ReAct planning prompt with observations
                prompt = self._create_react_planning_prompt(state)
                logger.info("Creating new plan based on observations")
            else:
                # Standard planning for first iteration or when no replanning needed
                prompt = self._create_planning_prompt(state)
                logger.info("Using standard planning approach")
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            
            # Parse the response to extract plan and reasoning
            plan = self._parse_planning_response(response.text)
            
            # Record this reasoning step
            reasoning_entry = {
                "iteration": plan_iteration + 1,
                "reasoning": plan.get("reasoning", ""),
                "plan": plan.get("tasks", []),
                "confidence": plan.get("confidence", 0.8),
                "timestamp": state.get("step_count", 0)
            }
            
            logger.info(f"Generated plan (iteration {plan_iteration + 1}): {plan}")
            
            # Update state with enhanced ReAct information
            updated_state = state.copy()
            updated_state.update({
                "tasks_to_run": plan.get("tasks", []),
                "tasks_completed": plan.get("completed_tasks", state.get("tasks_completed", [])),
                "step_count": state.get("step_count", 0) + 1,
                "planning_reasoning": plan.get("reasoning", ""),
                "plan_iterations": plan_iteration + 1,
                "max_plan_iterations": max_iterations,
                "needs_replanning": False,  # Reset flag
                "current_focus": plan.get("focus", ""),
                "reasoning_log": state.get("reasoning_log", []) + [reasoning_entry]
            })
            
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in ReAct planning: {str(e)}")
            # Fallback to default plan
            return self._create_fallback_plan(state)
    
    def _initialize_react_state(self, state: GraphState) -> GraphState:
        """Initialize ReAct-specific state fields."""
        updated_state = state.copy()
        updated_state.update({
            "reasoning_log": [],
            "observations": [],
            "plan_iterations": 0,
            "max_plan_iterations": 3,
            "needs_replanning": False,
            "gathered_evidence": []
        })
        return updated_state
    
    def _analyze_observations(self, state: GraphState) -> bool:
        """
        Analyze observations from previous actions to determine if replanning is needed.
        
        Returns:
            True if replanning is needed based on observations
        """
        observations = state.get("observations", [])
        plan_iterations = state.get("plan_iterations", 0)
        
        if plan_iterations == 0 or not observations:
            return False
        
        # Check latest observation for triggers that require replanning
        latest_obs = observations[-1] if observations else {}
        
        # Triggers for replanning:
        triggers = [
            latest_obs.get("quality_too_low", False),  # Retrieval quality insufficient
            latest_obs.get("context_insufficient", False),  # Need more context
            latest_obs.get("contradictory_evidence", False),  # Found conflicting info
            latest_obs.get("missing_critical_info", False),  # Key information missing
            state.get("needs_replanning", False)  # Explicit flag from other agents
        ]
        
        needs_replanning = any(triggers)
        
        if needs_replanning:
            logger.info(f"ReAct: Replanning triggered by observations: {latest_obs}")
        
        return needs_replanning
    
    def _create_react_planning_prompt(self, state: GraphState) -> str:
        """Create enhanced ReAct planning prompt with observations and reasoning."""
        # Get available agents and their capabilities
        agents_info = self._get_agents_info()
        
        # Get current state information
        query = state.get("query", "")
        intent = state.get("intent", "unknown")
        tasks_completed = state.get("tasks_completed", [])
        observations = state.get("observations", [])
        reasoning_log = state.get("reasoning_log", [])
        gathered_evidence = state.get("gathered_evidence", [])
        plan_iteration = state.get("plan_iterations", 0)
        
        # Format observations for context
        obs_text = self._format_observations(observations)
        reasoning_text = self._format_reasoning_log(reasoning_log)
        evidence_text = self._format_evidence(gathered_evidence)
        
        return f"""
You are an expert ReAct (Reason + Act) planner for a multi-agent RAG system. You can observe the results of actions and iteratively refine your plan to better satisfy complex user queries.

**Available Agents and Their Capabilities:**
{agents_info}

**Current Situation:**
- User Query: "{query}"
- Classified Intent: {intent}
- Planning Iteration: {plan_iteration + 1}
- Tasks Already Completed: {tasks_completed}

**Previous Reasoning:**
{reasoning_text}

**Observations from Previous Actions:**
{obs_text}

**Evidence Gathered So Far:**
{evidence_text}

**ReAct Planning Instructions:**
1. REASON: Analyze the observations from previous actions
2. ASSESS: Determine if the current approach is working or needs adjustment
3. PLAN: Design the next steps based on what you've learned
4. FOCUS: Identify what specific aspect needs attention

**Key ReAct Patterns:**
- If retrieval quality was low → try different search strategy or move to web search
- If information is incomplete → gather more specific context before synthesis
- If conflicting evidence found → investigate deeper or seek clarification
- If user query is complex → break into sub-questions and tackle systematically

**Response Format:**
REASONING: <analyze what the observations tell us and what we should do next>
FOCUS: <what specific aspect should we concentrate on now>
PLAN: <comma-separated list of agent names in execution order>
CONFIDENCE: <0-1 score indicating confidence in this plan>

Example Response:
REASONING: The corpus retrieval found some relevant information but with low confidence scores (0.3). The user is asking about methodologies comparison, so we need more comprehensive coverage. Let me try web search to supplement the corpus data.
FOCUS: comparative methodology analysis
PLAN: web_search,synthesis
CONFIDENCE: 0.85
"""
        
    def _format_observations(self, observations: List[Dict[str, Any]]) -> str:
        """Format observations for the planning prompt."""
        if not observations:
            return "No observations yet."
        
        formatted = []
        for i, obs in enumerate(observations, 1):
            agent = obs.get("agent", "unknown")
            status = obs.get("status", "unknown")
            details = obs.get("details", "")
            formatted.append(f"  {i}. {agent}: {status} - {details}")
        
        return "\n".join(formatted)
    
    def _format_reasoning_log(self, reasoning_log: List[Dict[str, Any]]) -> str:
        """Format previous reasoning for context."""
        if not reasoning_log:
            return "No previous reasoning."
        
        formatted = []
        for entry in reasoning_log:
            iteration = entry.get("iteration", "?")
            reasoning = entry.get("reasoning", "")
            formatted.append(f"  Iteration {iteration}: {reasoning}")
        
        return "\n".join(formatted)
    
    def _format_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        """Format gathered evidence."""
        if not evidence:
            return "No evidence gathered yet."
        
        formatted = []
        for i, item in enumerate(evidence, 1):
            source = item.get("source", "unknown")
            summary = item.get("summary", "")
            quality = item.get("quality_score", 0)
            formatted.append(f"  {i}. {source} (quality: {quality:.2f}): {summary}")
        
        return "\n".join(formatted)
    
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
                raw_tasks = [task.strip() for task in tasks_str.split(',') if task.strip()]
                # Filter out invalid tasks - routing is entry point only
                valid_tasks = []
                for task in raw_tasks:
                    if task.lower() in ['routing', 'query_analysis']:
                        logger.warning(f"Filtering out invalid task '{task}' - routing is entry point only")
                        continue
                    valid_tasks.append(task)
                plan["tasks"] = valid_tasks
            elif line.startswith('REASONING:'):
                plan["reasoning"] = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    plan["confidence"] = float(line.split(':', 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith('FOCUS:'):
                plan["focus"] = line.split(':', 1)[1].strip()
        
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
