"""
Agent Registry Pattern - The core of our extensible agent architecture.

This module provides the foundation for registering agents and their capabilities,
enabling dynamic agent discovery and routing.
"""

from typing import Dict, List, Callable, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger('arc_fusion.agents.registry')


class AgentRegistry:
    """
    Central registry for managing agents and their capabilities.
    
    Uses a decorator pattern to allow agents to self-register with their
    capabilities, making the system easily extensible.
    """
    
    _agents: Dict[str, Dict[str, Any]] = {}
    _capabilities_map: Dict[str, List[str]] = {}
    
    @classmethod
    def register(cls, 
                 name: str, 
                 capabilities: List[str],
                 dependencies: Optional[List[str]] = None,
                 priority: int = 0) -> Callable:
        """
        Decorator to register an agent with its capabilities.
        
        Args:
            name: Unique agent name
            capabilities: List of capabilities this agent provides
            dependencies: List of agent names this agent depends on
            priority: Priority for capability resolution (higher = preferred)
        
        Example:
            @AgentRegistry.register("corpus_retrieval", ["document_search", "rag"])
            def corpus_retrieval_agent(state: GraphState) -> GraphState:
                return retrieval_service.process(state)
        """
        def decorator(agent_func: Callable) -> Callable:
            if name in cls._agents:
                logger.warning(f"Agent '{name}' already registered, overwriting")
            
            cls._agents[name] = {
                'function': agent_func,
                'capabilities': capabilities,
                'dependencies': dependencies or [],
                'priority': priority,
                'registered': True
            }
            
            # Update capabilities mapping
            for capability in capabilities:
                if capability not in cls._capabilities_map:
                    cls._capabilities_map[capability] = []
                cls._capabilities_map[capability].append(name)
                # Sort by priority (highest first)
                cls._capabilities_map[capability].sort(
                    key=lambda x: cls._agents[x]['priority'], 
                    reverse=True
                )
            
            logger.info(f"Registered agent '{name}' with capabilities: {capabilities}")
            return agent_func
        
        return decorator
    
    @classmethod
    def get_agent(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get agent configuration by name."""
        return cls._agents.get(name)
    
    @classmethod
    def get_agent_function(cls, name: str) -> Optional[Callable]:
        """Get agent function by name."""
        agent = cls.get_agent(name)
        return agent['function'] if agent else None
    
    @classmethod
    def get_all_agents(cls) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents."""
        return cls._agents.copy()
    
    @classmethod
    def find_agents_by_capability(cls, capability: str) -> List[str]:
        """Find all agents that provide a specific capability."""
        return cls._capabilities_map.get(capability, [])
    
    @classmethod
    def get_best_agent_for_capability(cls, capability: str) -> Optional[str]:
        """Get the highest priority agent for a capability."""
        agents = cls.find_agents_by_capability(capability)
        return agents[0] if agents else None
    
    @classmethod
    def get_agent_dependencies(cls, name: str) -> List[str]:
        """Get dependencies for an agent."""
        agent = cls.get_agent(name)
        return agent['dependencies'] if agent else []
    
    @classmethod
    def validate_dependencies(cls) -> Dict[str, List[str]]:
        """Validate all agent dependencies and return missing ones."""
        missing_deps = {}
        
        for agent_name, config in cls._agents.items():
            missing = []
            for dep in config['dependencies']:
                if dep not in cls._agents:
                    missing.append(dep)
            if missing:
                missing_deps[agent_name] = missing
        
        return missing_deps
    
    @classmethod
    def get_capabilities_summary(cls) -> Dict[str, List[str]]:
        """Get summary of all capabilities and their providers."""
        return cls._capabilities_map.copy()
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered agents (useful for testing)."""
        cls._agents.clear()
        cls._capabilities_map.clear()
        logger.info("Agent registry cleared") 