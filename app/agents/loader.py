"""
Agent Loader - Imports all agents to ensure they are registered.

This module imports all agent modules so their @AgentRegistry.register
decorators are executed, making the agents available to the framework.
"""

from ..utils.logger import get_logger

logger = get_logger('arc_fusion.agents.loader')

def load_all_agents():
    """
    Import all agent modules to trigger registration.
    
    This function ensures all agents are loaded and registered with
    the AgentRegistry before the framework tries to use them.
    """
    logger.info("Loading all agents...")
    
    try:
        # Import all agent modules - this triggers their @register decorators
        from . import routing_agent
        from . import corpus_retrieval_agent
        from . import web_search_agent
        from . import synthesis_agent
        from . import clarification_agent
        
        # Log successful registration
        from .registry import AgentRegistry
        agents = AgentRegistry.get_all_agents()
        capabilities = AgentRegistry.get_capabilities_summary()
        
        logger.info(f"Successfully loaded {len(agents)} agents with {len(capabilities)} capabilities")
        
        # Log each agent
        for agent_name, config in agents.items():
            agent_caps = config['capabilities']
            logger.info(f"  - {agent_name}: {agent_caps}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load agents: {str(e)}")
        raise

# Auto-load agents when this module is imported
load_all_agents() 