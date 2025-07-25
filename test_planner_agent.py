#!/usr/bin/env python3
"""
Test script to verify PlannerAgent functionality.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_planner_agent():
    """Test that the PlannerAgent is correctly registered and the framework builds."""
    try:
        # Import the agent loader to trigger agent registration
        from agents.loader import load_all_agents
        from agents.registry import AgentRegistry
        from agents.framework import AgentFramework
        
        # Load all agents
        load_all_agents()
        
        # Check that the planner agent is registered
        agents = AgentRegistry.get_all_agents()
        print(f"Registered agents: {list(agents.keys())}")
        
        if 'planner' not in agents:
            print("ERROR: PlannerAgent not found in registered agents")
            return False
            
        print("SUCCESS: PlannerAgent is registered")
        
        # Check that the planner agent has the correct capabilities
        planner_config = agents['planner']
        capabilities = planner_config.get('capabilities', [])
        print(f"PlannerAgent capabilities: {capabilities}")
        
        if 'task_planning' not in capabilities:
            print("ERROR: PlannerAgent does not have 'task_planning' capability")
            return False
            
        print("SUCCESS: PlannerAgent has correct capabilities")
        
        # Test building the framework
        framework = AgentFramework()
        graph = framework.build_graph()
        
        if graph is None:
            print("ERROR: Failed to build graph")
            return False
            
        print("SUCCESS: Framework graph built successfully")
        
        # Get agent info
        agent_info = framework.get_agent_info()
        print(f"Agent info: {agent_info}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing PlannerAgent implementation...")
    success = test_planner_agent()
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)
