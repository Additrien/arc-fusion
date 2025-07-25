# Planner Agent Implementation

This document details the implementation of the PlannerAgent and the transition from rule-based to LLM-based planning in the Arc-Fusion multi-agent RAG system.

## Overview

The PlannerAgent represents a significant architectural evolution from the previous rule-based orchestrator to an LLM-based planning system that implements the foundations of a ReAct (Reason + Act) architecture. This transition enables dynamic, model-driven execution plans rather than static graphs.

## Key Changes

### 1. Introduction of PlannerAgent

The PlannerAgent is a new agent that uses an LLM (Gemini 2.5 Flash Lite) to dynamically plan which agents to execute and in what order based on:

- The user's query
- The classified intent from the RoutingAgent
- The current state of the system
- Available agents and their capabilities

### 2. Replacement of Rule-Based Orchestrator

The previous orchestrator used hard-coded logic to determine the execution sequence:

```python
# Previous rule-based logic
if intent == 'retrieve_corpus':
    tasks = ['corpus_retrieval']
elif intent == 'search_web':
    tasks = ['web_search']
elif intent == 'corpus_and_web_search':
    tasks = ['corpus_retrieval', 'web_search']
elif intent == 'clarify':
    tasks = ['clarification']
```

This has been replaced with LLM-based planning that can dynamically determine the optimal sequence of agents to execute.

### 3. Framework Updates

The LangGraph framework has been updated to:

1. Route from the RoutingAgent to the PlannerAgent
2. Use the PlannerAgent for dynamic task planning
3. Maintain the conditional routing logic but based on the LLM-generated plan
4. Handle the PlannerAgent specially in the graph construction to avoid node conflicts

## Implementation Details

### PlannerAgent Architecture

The PlannerAgent follows the same pattern as other agents in the system:

1. **Registration**: Uses the `@AgentRegistry.register` decorator
2. **Capabilities**: Declared as `task_planning` and `workflow_orchestration`
3. **LLM Integration**: Uses the primary LLM model (Gemini 2.5 Flash Lite) for planning
4. **State Management**: Updates the `tasks_to_run` and `tasks_completed` fields in the graph state

### Planning Process

The PlannerAgent follows this process:

1. **State Analysis**: Examines the current state including query, intent, and completed tasks
2. **Agent Discovery**: Retrieves information about all available agents and their capabilities
3. **LLM Prompting**: Constructs a prompt that provides the LLM with:
   - Available agents and their capabilities
   - Current state information
   - Planning rules and constraints
4. **Plan Generation**: The LLM generates a plan specifying which agents to execute in what order
5. **Plan Parsing**: The response is parsed to extract the execution plan
6. **State Update**: The graph state is updated with the planned tasks

### Fallback Mechanism

If the LLM-based planning fails, a fallback mechanism uses the previous rule-based logic to ensure system reliability.

## Benefits of LLM-Based Planning

### 1. Dynamic Execution Plans

Unlike the previous static rules, the LLM can generate execution plans that are tailored to each specific query and context, potentially leading to more efficient and effective processing.

### 2. Adaptability

The system can adapt to new agents and capabilities without requiring changes to hard-coded rules, making it more maintainable and extensible.

### 3. Complex Reasoning

The LLM can consider multiple factors simultaneously when planning, such as:
- Query complexity
- Available context
- Quality of previous results
- Potential need for clarification

### 4. Foundation for ReAct Architecture

This implementation provides the foundation for a full ReAct (Reason + Act) architecture where the system can iteratively plan, act, observe, and replan based on intermediate results.

## Future Enhancements

### 1. Iterative Planning

Future enhancements could enable the PlannerAgent to dynamically adjust the plan based on intermediate results, creating a true Reason-Act loop.

### 2. Plan Optimization

The system could learn from past executions to optimize future plans, potentially using reinforcement learning techniques.

### 3. Multi-Turn Planning

For complex queries requiring multiple steps, the PlannerAgent could generate and execute multi-turn plans with intermediate goals.

## Integration with Existing System

The PlannerAgent integrates seamlessly with the existing system:

1. **Backward Compatibility**: The system continues to function with existing agents
2. **Incremental Enhancement**: The transition doesn't break existing functionality
3. **Extensibility**: New agents automatically become available to the PlannerAgent

## Performance Considerations

### 1. LLM Latency

The addition of LLM-based planning introduces some latency, but this is mitigated by:

- Using a fast, cost-effective model (Gemini 2.5 Flash Lite)
- Caching mechanisms for similar queries
- Parallel execution of independent agents

### 2. Cost Management

LLM calls are managed through:

- Efficient prompting to minimize token usage
- Fallback mechanisms to avoid unnecessary LLM calls
- Rate limiting and retry logic

## Testing and Validation

The PlannerAgent has been validated through:

1. **Unit Testing**: Individual components are tested
2. **Integration Testing**: The agent works correctly within the full system
3. **Scenario Testing**: Various query types and edge cases are handled correctly
4. **Fallback Testing**: The system gracefully handles LLM failures

## Conclusion

The implementation of the PlannerAgent represents a significant advancement in the system's capabilities, moving from static, rule-based orchestration to dynamic, LLM-driven planning. This transition not only improves the system's adaptability and efficiency but also establishes a foundation for more sophisticated AI architectures like ReAct.
