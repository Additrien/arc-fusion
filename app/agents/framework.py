"""
Agent Framework - The orchestration engine for our multi-agent system.

This module provides the core framework that manages agent registration,
graph construction, and execution orchestration using LangGraph.
"""

from typing import Dict, Any, Optional, List
import logging
import asyncio
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
        
        # Add the parallel executor node for concurrent task execution
        graph.add_node("parallel_executor", self.parallel_executor_node)

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
                "parallel_executor": "parallel_executor",  # Add parallel executor
                END: END
            }
        )

        # 3. Retrieval and search agents loop back to the planner to update the task list status.
        if retrieval_agent:
            graph.add_edge(retrieval_agent, 'planner')
        if web_agent:
            graph.add_edge(web_agent, 'planner')
        
        # 4. Parallel executor also loops back to planner after completion
        graph.add_edge('parallel_executor', 'planner')

        # 4. The synthesis agent is the final step before ending.
        if synthesis_agent:
            graph.add_edge(synthesis_agent, END)
        
        # 5. The clarification agent goes directly to END (no synthesis needed)
        if clarification_agent:
            graph.add_edge(clarification_agent, END)

    def planner_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Enhanced ReAct planner node that handles iterative reasoning and replanning.
        This replaces the rule-based orchestrator with LLM-based planning.
        """
        tasks_to_run = state.get('tasks_to_run', [])
        tasks_completed = state.get('tasks_completed', [])
        plan_iterations = state.get('plan_iterations', 0)
        needs_replanning = state.get('needs_replanning', False)
        
        logger.debug(f"ReAct Planner Node: tasks_to_run={tasks_to_run}, tasks_completed={tasks_completed}, plan_iterations={plan_iterations}, needs_replanning={needs_replanning}")
        
        # Use the PlannerAgent to determine the next steps with ReAct capabilities
        planner_agent_func = AgentRegistry.get_agent_function("planner")
        if not planner_agent_func:
            logger.error("Planner agent not found, using fallback logic")
            return self._fallback_planning(state)
        
        # Check if we need to trigger replanning based on observations
        if needs_replanning or self._should_replan_based_on_observations(state):
            logger.info("ReAct: Triggering replanning based on agent observations")
            # Reset the flag and allow planner to create new plan
            state = state.copy()
            state["needs_replanning"] = True
        
        # Execute the enhanced planner agent with ReAct support
        updated_state = planner_agent_func(state)
        return updated_state
    
    def _should_replan_based_on_observations(self, state: GraphState) -> bool:
        """Check if observations from agents suggest we should replan."""
        observations = state.get('observations', [])
        if not observations:
            return False
        
        # Get the most recent observation
        latest_obs = observations[-1]
        
        # Triggers for replanning based on agent observations
        triggers = [
            latest_obs.get('quality_too_low', False),
            latest_obs.get('context_insufficient', False),
            latest_obs.get('needs_web_fallback', False),
            latest_obs.get('contradictory_evidence', False)
        ]
        
        return any(triggers)

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

        # Check for parallel execution opportunity (corpus_retrieval + web_search)
        remaining_tasks = [task for task in tasks_to_run if task not in tasks_completed]
        
        if self._can_execute_in_parallel(remaining_tasks):
            logger.info("Detected parallel execution opportunity for corpus_retrieval and web_search")
            # Execute parallel tasks and mark as special state
            state['parallel_execution'] = True
            state['parallel_tasks'] = ['corpus_retrieval', 'web_search']
            
            # Add both agents to path for tracking
            retrieval_agent = self._get_agent_for_capability("document_search")
            web_agent = self._get_agent_for_capability("web_search")
            if retrieval_agent:
                state['agent_path'].append(f"{retrieval_agent}(parallel)")
            if web_agent:
                state['agent_path'].append(f"{web_agent}(parallel)")
            
            # Route to parallel execution node
            return "parallel_executor"

        # Find the next task that hasn't been completed (sequential execution).
        for task in remaining_tasks:
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
        
        # Convert conversation history to simple messages format for agents
        # Note: We'll store this separately from LangGraph messages to avoid conflicts
        conversation_messages = []
        if initial_state and "conversation_history" in initial_state:
            for entry in initial_state["conversation_history"]:
                # Ensure entry is a dict, not a ConversationEntry dataclass object
                if hasattr(entry, 'query') and hasattr(entry, 'answer'):  # It's a ConversationEntry dataclass
                    entry_query = entry.query
                    entry_answer = entry.answer
                    entry_timestamp = entry.timestamp
                    entry_confidence = entry.confidence
                    entry_agent_path = entry.agent_path
                elif isinstance(entry, dict):  # It's already a dict
                    entry_query = entry.get("query", "")
                    entry_answer = entry.get("answer", "")
                    entry_timestamp = entry.get("timestamp")
                    entry_confidence = entry.get("confidence", 0.0)
                    entry_agent_path = entry.get("agent_path", [])
                else:
                    # Skip unknown entry types
                    continue
                
                # Add user message as plain dict
                conversation_messages.append({
                    "role": "user",
                    "content": str(entry_query),
                    "timestamp": entry_timestamp
                })
                # Add assistant message as plain dict
                conversation_messages.append({
                    "role": "assistant", 
                    "content": str(entry_answer),
                    "timestamp": entry_timestamp,
                    "confidence": float(entry_confidence),
                    "agent_path": list(entry_agent_path) if entry_agent_path else []
                })
        
        # Initialize state, starting the agent_path with the entrypoint.
        # Use empty messages for LangGraph, and add conversation_messages separately
        state = GraphState(
            messages=[],  # Empty for LangGraph
            query=query,
            session_id=session_id,
            intent=None,
            required_capability=None,
            retrieved_context=[],
            step_count=0,
            agent_path=['routing'], # Start the path with the entry point agent
            tasks_to_run=[],
            tasks_completed=[],
            conversation_messages=conversation_messages,  # Add conversation history separately
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
    
    def _can_execute_in_parallel(self, remaining_tasks: List[str]) -> bool:
        """
        Check if the remaining tasks can be executed in parallel.
        
        Currently supports parallel execution of corpus_retrieval and web_search.
        
        Args:
            remaining_tasks: List of tasks that haven't been completed yet
            
        Returns:
            True if tasks can be executed in parallel, False otherwise
        """
        # Check if both corpus_retrieval and web_search are in remaining tasks
        has_corpus = 'corpus_retrieval' in remaining_tasks
        has_web = 'web_search' in remaining_tasks
        
        # Only execute in parallel if both tasks are present and we have the required agents
        if has_corpus and has_web:
            retrieval_agent = self._get_agent_for_capability("document_search")
            web_agent = self._get_agent_for_capability("web_search")
            return retrieval_agent is not None and web_agent is not None
        
        return False
    
    async def parallel_executor_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute corpus retrieval and web search in parallel for improved performance.
        
        Args:
            state: Current graph state
            
        Returns:
            Updated state with results from both parallel tasks
        """
        logger.info("Starting parallel execution of corpus_retrieval and web_search")
        
        # Get agent functions
        retrieval_agent_func = AgentRegistry.get_agent_function(
            self._get_agent_for_capability("document_search")
        )
        web_agent_func = AgentRegistry.get_agent_function(
            self._get_agent_for_capability("web_search")
        )
        
        if not retrieval_agent_func or not web_agent_func:
            logger.error("Missing agent functions for parallel execution")
            # Fall back to marking tasks as completed to avoid infinite loop
            tasks_completed = state.get('tasks_completed', [])
            return {
                "tasks_completed": tasks_completed + ['corpus_retrieval', 'web_search'],
                "error_info": {
                    "error_type": "ParallelExecutionError",
                    "error_message": "Missing agent functions for parallel execution"
                }
            }
        
        try:
            # Create separate state copies for each agent to avoid interference
            retrieval_state = state.copy()
            web_state = state.copy()
            
            # Create async tasks for parallel execution
            retrieval_task = asyncio.create_task(
                self._run_agent_async(retrieval_agent_func, retrieval_state)
            )
            web_task = asyncio.create_task(
                self._run_agent_async(web_agent_func, web_state)
            )
            
            # Wait for both tasks to complete
            retrieval_result, web_result = await asyncio.gather(
                retrieval_task, web_task, return_exceptions=True
            )
            
            # Merge results from both agents
            merged_state = self._merge_parallel_results(
                state, retrieval_result, web_result
            )
            
            # Mark both tasks as completed
            tasks_completed = merged_state.get('tasks_completed', [])
            if 'corpus_retrieval' not in tasks_completed:
                tasks_completed.append('corpus_retrieval')
            if 'web_search' not in tasks_completed:
                tasks_completed.append('web_search')
            
            merged_state['tasks_completed'] = tasks_completed
            merged_state['parallel_execution'] = False  # Reset flag
            
            logger.info("Parallel execution completed successfully")
            return merged_state
            
        except Exception as e:
            logger.error(f"Error during parallel execution: {str(e)}")
            # Mark tasks as completed to avoid infinite loop, but note the error
            tasks_completed = state.get('tasks_completed', [])
            return {
                "tasks_completed": tasks_completed + ['corpus_retrieval', 'web_search'],
                "error_info": {
                    "error_type": "ParallelExecutionError",
                    "error_message": str(e)
                }
            }
    
    async def _run_agent_async(self, agent_func: AgentFunction, state: GraphState) -> Dict[str, Any]:
        """
        Run an agent function asynchronously.
        
        Args:
            agent_func: Agent function to execute
            state: State to pass to the agent
            
        Returns:
            Updated state from the agent
        """
        # Check if agent function is async
        if asyncio.iscoroutinefunction(agent_func):
            return await agent_func(state)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, agent_func, state)
    
    def _merge_parallel_results(self, original_state: GraphState, 
                               retrieval_result: Any, web_result: Any) -> Dict[str, Any]:
        """
        Merge results from parallel execution of retrieval and web search agents.
        
        Args:
            original_state: Original state before parallel execution
            retrieval_result: Result from corpus retrieval agent
            web_result: Result from web search agent
            
        Returns:
            Merged state with combined results
        """
        merged_state = original_state.copy()
        
        # Handle retrieval results
        if isinstance(retrieval_result, Exception):
            logger.error(f"Corpus retrieval failed: {str(retrieval_result)}")
            merged_state['error_info'] = merged_state.get('error_info', {})
            merged_state['error_info']['corpus_retrieval_error'] = str(retrieval_result)
        elif isinstance(retrieval_result, dict):
            # Merge retrieval context and sources
            if 'retrieved_context' in retrieval_result:
                merged_state['retrieved_context'] = retrieval_result['retrieved_context']
            if 'document_sources' in retrieval_result:
                merged_state['document_sources'] = retrieval_result['document_sources']
            if 'best_retrieval_score' in retrieval_result:
                merged_state['best_retrieval_score'] = retrieval_result['best_retrieval_score']
        
        # Handle web search results
        if isinstance(web_result, Exception):
            logger.error(f"Web search failed: {str(web_result)}")
            merged_state['error_info'] = merged_state.get('error_info', {})
            merged_state['error_info']['web_search_error'] = str(web_result)
        elif isinstance(web_result, dict):
            # Merge web context and sources
            if 'web_context' in web_result:
                merged_state['web_context'] = web_result['web_context']
            if 'web_sources' in web_result:
                merged_state['web_sources'] = web_result['web_sources']
        
        # Combine observations from both agents
        observations = merged_state.get('observations', [])
        if isinstance(retrieval_result, dict) and 'observations' in retrieval_result:
            observations.extend(retrieval_result['observations'])
        if isinstance(web_result, dict) and 'observations' in web_result:
            observations.extend(web_result['observations'])
        merged_state['observations'] = observations
        
        logger.info("Parallel results merged successfully")
        return merged_state

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
