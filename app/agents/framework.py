"""
Agent Framework - The orchestration engine for our multi-agent system.

This module provides the core framework that manages agent registration,
graph construction, and execution orchestration using LangGraph.
"""

from typing import Dict, Any, Optional, List
import logging
from langgraph.graph import StateGraph, END

from .registry import AgentRegistry
from .state import GraphState, AgentFunction
from ..utils.logger import get_logger
from .. import config

logger = get_logger('arc_fusion.agents.framework')


class AgentFramework:
    """
    The central orchestration engine for our multi-agent RAG system.
    
    This framework:
    1. Automatically discovers registered agents
    2. Builds the LangGraph dynamically based on agent capabilities
    3. Handles routing between agents
    4. Manages state flow and error handling
    """
    
    def __init__(self):
        self.graph: Optional[Any] = None
        self._routing_rules: Dict[str, str] = {}
        self._built = False
        
    def build_graph(self) -> Any:
        """
        Dynamically build the LangGraph based on registered agents.
        
        Returns:
            Compiled graph ready for execution
        """
        logger.info("Building multi-agent graph")
        
        # Validate agent dependencies
        missing_deps = AgentRegistry.validate_dependencies()
        if missing_deps:
            raise ValueError(f"Missing agent dependencies: {missing_deps}")
        
        # Create the graph
        graph = StateGraph(GraphState)
        
        # Add all registered agents as nodes (except planner which is handled specially)
        agents = AgentRegistry.get_all_agents()
        logger.info(f"Adding {len(agents)} agents as graph nodes")
        
        for agent_name, config in agents.items():
            # Skip the planner agent as it's handled specially
            if agent_name == "planner":
                continue
            graph.add_node(agent_name, config['function'])
            logger.debug(f"Added agent node: {agent_name}")

        # Add the planner node that manages the task list
        graph.add_node("planner", self.planner_node)

        # Set entry point (must have a routing agent)
        if 'routing' not in agents:
            raise ValueError("No routing agent found. Please register a routing agent.")
        
        graph.set_entry_point('routing')
        
        # Define the graph structure using edges
        self._build_graph_edges(graph, agents)
        
        # Compile the graph
        self.graph = graph.compile()
        self._built = True
        
        logger.info("Multi-agent graph built successfully")
        return self.graph

    def _build_graph_edges(self, graph: StateGraph, agents: Dict[str, Any]):
        """Define the edges and control flow of the multi-agent graph."""
        retrieval_agent = self._get_agent_for_capability("document_search")
        web_agent = self._get_agent_for_capability("web_search")
        synthesis_agent = self._get_agent_for_capability("response_synthesis")
        clarification_agent = self._get_agent_for_capability("query_clarification")
        planner_agent = self._get_agent_for_capability("task_planning")

        # 1. The routing agent goes to the planner to create the execution plan.
        graph.add_edge('routing', 'planner')

        # 2. After the planner creates the plan, a conditional router decides the next agent.
        graph.add_conditional_edges(
            "planner",
            self._routing_logic,
            {
                # Map potential destinations to themselves
                **({retrieval_agent: retrieval_agent} if retrieval_agent else {}),
                **({web_agent: web_agent} if web_agent else {}),
                **({synthesis_agent: synthesis_agent} if synthesis_agent else {}),
                **({clarification_agent: clarification_agent} if clarification_agent else {}),
                **({planner_agent: planner_agent} if planner_agent else {}),
                END: END
            }
        )

        # 3. Retrieval and search agents loop back to the planner to update the task list status.
        if retrieval_agent:
            graph.add_edge(retrieval_agent, 'planner')
        if web_agent:
            graph.add_edge(web_agent, 'planner')

        # 4. The synthesis agent is the final step before ending.
        if synthesis_agent:
            graph.add_edge(synthesis_agent, END)
        
        # 5. The clarification agent goes directly to END (no synthesis needed)
        if clarification_agent:
            graph.add_edge(clarification_agent, END)

    def planner_node(self, state: GraphState) -> Dict[str, Any]:
        """
        This node uses the PlannerAgent to dynamically plan the execution workflow.
        It replaces the rule-based orchestrator with LLM-based planning.
        """
        logger.debug(f"Planner Node: tasks_to_run={state.get('tasks_to_run', [])}, tasks_completed={state.get('tasks_completed', [])}")
        
        # Use the PlannerAgent to determine the next steps
        planner_agent_func = AgentRegistry.get_agent_function("planner")
        if not planner_agent_func:
            logger.error("Planner agent not found, using fallback logic")
            return self._fallback_planning(state)
        
        # Execute the planner agent
        updated_state = planner_agent_func(state)
        return updated_state

    def _fallback_planning(self, state: GraphState) -> Dict[str, Any]:
        """
        Fallback planning logic when PlannerAgent is not available.
        """
        # First pass: Initialize tasks based on the router's intent.
        if not state.get('tasks_to_run'):
            intent = state.get('intent', 'end')
            tasks = []
            if intent == 'retrieve_corpus':
                tasks = ['corpus_retrieval', 'synthesis']
            elif intent == 'search_web':
                tasks = ['web_search', 'synthesis']
            elif intent == 'corpus_and_web_search':
                tasks = ['corpus_retrieval', 'web_search', 'synthesis']
            elif intent == 'clarify':
                tasks = ['clarification']
            
            logger.info(f"Fallback planner initialized tasks for intent '{intent}': {tasks}")
            return {
                "tasks_to_run": tasks,
                "tasks_completed": []
            }

        # Subsequent passes: Check for quality-based fallback.
        completed_tasks = state.get('tasks_completed', [])
        tasks_to_run = state.get('tasks_to_run', [])
        
        if 'corpus_retrieval' in completed_tasks and 'web_search' not in tasks_to_run:
            best_score = state.get('best_retrieval_score', 1.0)
            if best_score < config.RELEVANCE_THRESHOLD:
                logger.info(f"Low retrieval score ({best_score:.2f}), adding web_search as a fallback task.")
                return {"tasks_to_run": tasks_to_run + ['web_search']}

        # If no state changes are needed, return an empty dict.
        return {}

    def _routing_logic(self, state: GraphState) -> str:
        """
        This is the conditional edge logic. It inspects the state and returns the
        string name of the node to execute next.
        """
        tasks_to_run = state.get('tasks_to_run', [])
        tasks_completed = state.get('tasks_completed', [])

        # Find the next task that hasn't been completed.
        for task in tasks_to_run:
            if task not in tasks_completed:
                next_agent = self._get_agent_for_capability_or_name(task)
                if next_agent != END:
                    logger.info(f"Routing logic determined next agent: {next_agent}")
                    state['agent_path'].append(next_agent)
                    return next_agent
                else:
                    break # Agent not found, end the process.

        # If all tasks are completed, decide whether to synthesize or end.
        has_context = state.get('retrieved_context') or state.get('web_context')
        if has_context:
            synthesis_agent = self._get_agent_for_capability("response_synthesis")
            logger.info("All tasks completed. Routing to synthesis.")
            state['agent_path'].append(synthesis_agent)
            return synthesis_agent
        else:
            logger.info("All tasks completed, but no context found. Routing to END.")
            return END
    
    def _get_agent_for_capability_or_name(self, task_name: str) -> str:
        """Helper to resolve a task name to an agent name."""
        if AgentRegistry.get_agent(task_name):
            return task_name
        
        agent = self._get_agent_for_capability(task_name)
        if not agent:
            logger.error(f"No agent found for capability '{task_name}', ending execution.")
            return END
        
        # Prevent routing back to the routing agent (entry point only)
        if agent == 'routing':
            logger.warning(f"Task '{task_name}' resolved to routing agent, but routing is entry-point only. Ending execution.")
            return END
            
        return agent

    def _get_agent_for_capability(self, capability: str) -> Optional[str]:
        """Get the best agent for a specific capability."""
        return AgentRegistry.get_best_agent_for_capability(capability)
    
    async def process_query(self, 
                           query: str, 
                           session_id: str,
                           initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-agent system.
        
        Args:
            query: User's question
            session_id: Session identifier for conversation continuity
            initial_state: Optional initial state data
            
        Returns:
            Final state containing the response and metadata
        """
        if not self._built or not self.graph:
            raise RuntimeError("Graph not built. Call build_graph() first.")
        
        # Initialize state, starting the agent_path with the entrypoint.
        state = GraphState(
            messages=[],
            query=query,
            session_id=session_id,
            intent=None,
            required_capability=None,
            retrieved_context=[],
            step_count=0,
            agent_path=['routing'], # Start the path with the entry point agent
            tasks_to_run=[],
            tasks_completed=[],
            **(initial_state or {})
        )
        
        logger.info(f"Processing query for session {session_id}: {query[:100]}...")
        
        try:
            # Execute the graph
            final_state = await self.graph.ainvoke(state)
            
            logger.info(f"Query processed successfully. Agent path: {final_state.get('agent_path', [])}")
            return final_state
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            # Return error state
            return {
                **state,
                "error_info": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                "final_answer": "I apologize, but I encountered an error processing your request. Please try again."
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about registered agents and capabilities."""
        return {
            "agents": AgentRegistry.get_all_agents(),
            "capabilities": AgentRegistry.get_capabilities_summary(),
            "graph_built": self._built
        }
    
    def rebuild_graph(self) -> Any:
        """Rebuild the graph (useful when agents are added/removed)."""
        logger.info("Rebuilding multi-agent graph")
        return self.build_graph()
