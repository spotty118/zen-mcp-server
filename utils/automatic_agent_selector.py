"""
Automatic Agent Selection System

This module implements intelligent agent selection based on task characteristics
and available CPU cores. It analyzes the task requirements and automatically
selects the most appropriate agents for optimal performance.

Key Features:
- Task-based agent role selection
- CPU core count optimization
- Dynamic team composition
- Load balancing across agents
- Performance optimization
"""

import logging
import os
import psutil
import platform
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from utils.agent_core import AgentRole, Agent
from utils.agent_communication import AgentCommunicationSystem

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Different types of tasks that can be processed"""
    CODE_REVIEW = "code_review"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    DEBUGGING = "debugging"
    PLANNING = "planning"
    CONSENSUS_BUILDING = "consensus_building"
    GENERAL_ANALYSIS = "general_analysis"
    PARALLEL_THINKING = "parallel_thinking"
    DEEP_THINKING = "deep_thinking"


class TaskComplexity(Enum):
    """Complexity levels for tasks"""
    SIMPLE = "simple"        # 1-2 agents
    MODERATE = "moderate"    # 2-4 agents
    COMPLEX = "complex"      # 3-6 agents
    VERY_COMPLEX = "very_complex"  # 4-8 agents


@dataclass
class TaskCharacteristics:
    """Characteristics of a task that influence agent selection"""
    task_type: TaskType
    complexity: TaskComplexity
    requires_security_focus: bool = False
    requires_performance_focus: bool = False
    requires_architecture_review: bool = False
    requires_consensus: bool = False
    requires_deep_analysis: bool = False
    file_count: int = 0
    estimated_processing_time: float = 60.0  # seconds
    priority: int = 5  # 1-10, 10 being highest


@dataclass
class SystemCapabilities:
    """Information about the system's capabilities"""
    total_cores: int
    available_cores: int
    cpu_architecture: str
    system_type: str  # linux, darwin, windows
    memory_gb: float
    is_high_performance: bool
    supports_parallel_processing: bool
    recommended_max_agents: int


class AutomaticAgentSelector:
    """
    Intelligent agent selection system that automatically chooses
    the best agents for a given task based on system capabilities
    """
    
    def __init__(self, communication_system: AgentCommunicationSystem):
        self.communication_system = communication_system
        self.system_capabilities = self._detect_system_capabilities()
        self.task_agent_mapping = self._initialize_task_mappings()
        
        logger.info(f"Agent selector initialized with {self.system_capabilities.total_cores} cores, "
                   f"recommending max {self.system_capabilities.recommended_max_agents} agents")
    
    def _detect_system_capabilities(self) -> SystemCapabilities:
        """Detect system capabilities for optimal agent allocation"""
        total_cores = os.cpu_count() or 4
        
        try:
            # Get available memory
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
        except Exception:
            memory_gb = 8.0  # Default assumption
        
        # Detect architecture and system
        architecture = platform.machine().lower()
        system_type = platform.system().lower()
        
        # Determine if this is a high-performance system
        is_high_performance = (
            total_cores >= 8 and 
            memory_gb >= 16 and
            ("x86_64" in architecture or "aarch64" in architecture or "arm64" in architecture)
        )
        
        # Calculate recommended max agents based on system capabilities
        if is_high_performance:
            recommended_max_agents = min(total_cores, 8)  # Cap at 8 for most efficient operation
        else:
            recommended_max_agents = min(total_cores // 2, 4)  # Conservative for lower-end systems
        
        # Ensure minimum of 2 agents for meaningful collaboration
        recommended_max_agents = max(recommended_max_agents, 2)
        
        return SystemCapabilities(
            total_cores=total_cores,
            available_cores=total_cores,  # Simplified - assume all cores available
            cpu_architecture=architecture,
            system_type=system_type,
            memory_gb=memory_gb,
            is_high_performance=is_high_performance,
            supports_parallel_processing=total_cores >= 4,
            recommended_max_agents=recommended_max_agents
        )
    
    def _initialize_task_mappings(self) -> Dict[TaskType, Dict[str, Any]]:
        """Initialize mappings between task types and optimal agent configurations"""
        return {
            TaskType.CODE_REVIEW: {
                "primary_roles": [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.ARCHITECTURE_REVIEWER],
                "secondary_roles": [AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER],
                "min_agents": 2,
                "optimal_agents": 3,
                "max_agents": 4,
                "requires_coordinator": True
            },
            TaskType.SECURITY_ANALYSIS: {
                "primary_roles": [AgentRole.SECURITY_ANALYST],
                "secondary_roles": [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.ARCHITECTURE_REVIEWER],
                "min_agents": 1,
                "optimal_agents": 2,
                "max_agents": 3,
                "requires_coordinator": False
            },
            TaskType.PERFORMANCE_OPTIMIZATION: {
                "primary_roles": [AgentRole.PERFORMANCE_OPTIMIZER],
                "secondary_roles": [AgentRole.ARCHITECTURE_REVIEWER, AgentRole.CODE_QUALITY_INSPECTOR],
                "min_agents": 1,
                "optimal_agents": 2,
                "max_agents": 3,
                "requires_coordinator": False
            },
            TaskType.ARCHITECTURE_DESIGN: {
                "primary_roles": [AgentRole.ARCHITECTURE_REVIEWER],
                "secondary_roles": [AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.SECURITY_ANALYST],
                "min_agents": 2,
                "optimal_agents": 3,
                "max_agents": 4,
                "requires_coordinator": True
            },
            TaskType.DEBUGGING: {
                "primary_roles": [AgentRole.DEBUG_SPECIALIST],
                "secondary_roles": [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.PERFORMANCE_OPTIMIZER],
                "min_agents": 1,
                "optimal_agents": 2,
                "max_agents": 3,
                "requires_coordinator": False
            },
            TaskType.PLANNING: {
                "primary_roles": [AgentRole.PLANNING_COORDINATOR],
                "secondary_roles": [AgentRole.ARCHITECTURE_REVIEWER, AgentRole.CONSENSUS_FACILITATOR],
                "min_agents": 1,
                "optimal_agents": 2,
                "max_agents": 3,
                "requires_coordinator": True
            },
            TaskType.CONSENSUS_BUILDING: {
                "primary_roles": [AgentRole.CONSENSUS_FACILITATOR],
                "secondary_roles": [AgentRole.PLANNING_COORDINATOR, AgentRole.ARCHITECTURE_REVIEWER],
                "min_agents": 2,
                "optimal_agents": 3,
                "max_agents": 5,
                "requires_coordinator": True
            },
            TaskType.PARALLEL_THINKING: {
                "primary_roles": [AgentRole.GENERALIST, AgentRole.ARCHITECTURE_REVIEWER],
                "secondary_roles": [AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.DEBUG_SPECIALIST],
                "min_agents": 2,
                "optimal_agents": 4,
                "max_agents": 6,
                "requires_coordinator": True
            },
            TaskType.GENERAL_ANALYSIS: {
                "primary_roles": [AgentRole.GENERALIST],
                "secondary_roles": [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.ARCHITECTURE_REVIEWER],
                "min_agents": 1,
                "optimal_agents": 2,
                "max_agents": 3,
                "requires_coordinator": False
            }
        }
    
    def select_agents_for_task(
        self,
        task_characteristics: TaskCharacteristics,
        available_agent_ids: Optional[List[str]] = None
    ) -> Tuple[List[str], Optional[str]]:
        """
        Select the optimal agents for a given task
        
        Args:
            task_characteristics: Characteristics of the task
            available_agent_ids: List of available agent IDs (if None, use all available)
            
        Returns:
            Tuple of (selected_agent_ids, coordinator_agent_id)
        """
        task_mapping = self.task_agent_mapping.get(task_characteristics.task_type)
        if not task_mapping:
            logger.warning(f"No mapping found for task type {task_characteristics.task_type}")
            return self._select_default_agents(available_agent_ids)
        
        # Determine optimal agent count based on task complexity and system capabilities
        target_agent_count = self._calculate_target_agent_count(task_characteristics, task_mapping)
        
        # Get available agents
        if available_agent_ids is None:
            available_agents = list(self.communication_system.agents.values())
        else:
            available_agents = [
                self.communication_system.agents[agent_id] 
                for agent_id in available_agent_ids 
                if agent_id in self.communication_system.agents
            ]
        
        if not available_agents:
            logger.warning("No available agents found")
            return [], None
        
        # Select agents based on roles and preferences
        selected_agents = self._select_optimal_agents(
            available_agents, 
            task_mapping, 
            task_characteristics,
            target_agent_count
        )
        
        # Select coordinator if needed
        coordinator = None
        if task_mapping.get("requires_coordinator", False) and selected_agents:
            coordinator = self._select_coordinator(selected_agents, task_characteristics)
        
        selected_agent_ids = [agent.agent_id for agent in selected_agents]
        coordinator_id = coordinator.agent_id if coordinator else None
        
        logger.info(f"Selected {len(selected_agent_ids)} agents for {task_characteristics.task_type.value}: "
                   f"{[agent.role.value for agent in selected_agents]}")
        
        return selected_agent_ids, coordinator_id
    
    def _calculate_target_agent_count(
        self, 
        task_characteristics: TaskCharacteristics, 
        task_mapping: Dict[str, Any]
    ) -> int:
        """Calculate the optimal number of agents for this task"""
        base_count = task_mapping.get("optimal_agents", 2)
        min_count = task_mapping.get("min_agents", 1)
        max_count = task_mapping.get("max_agents", 4)
        
        # Adjust based on task complexity
        complexity_multiplier = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.3,
            TaskComplexity.VERY_COMPLEX: 1.6
        }
        
        adjusted_count = int(base_count * complexity_multiplier[task_characteristics.complexity])
        
        # Apply system limitations
        max_system_agents = self.system_capabilities.recommended_max_agents
        
        # Final count within bounds
        target_count = max(min_count, min(adjusted_count, max_count, max_system_agents))
        
        return target_count
    
    def _select_optimal_agents(
        self,
        available_agents: List[Agent],
        task_mapping: Dict[str, Any],
        task_characteristics: TaskCharacteristics,
        target_count: int
    ) -> List[Agent]:
        """Select the most suitable agents for the task"""
        primary_roles = task_mapping.get("primary_roles", [])
        secondary_roles = task_mapping.get("secondary_roles", [])
        
        selected_agents = []
        
        # First, try to get agents with primary roles
        for role in primary_roles:
            agents_with_role = [agent for agent in available_agents if agent.role == role]
            if agents_with_role and len(selected_agents) < target_count:
                # Select the most recently active agent with this role
                best_agent = max(agents_with_role, key=lambda a: a.last_activity)
                if best_agent not in selected_agents:
                    selected_agents.append(best_agent)
        
        # Then add agents with secondary roles if needed
        for role in secondary_roles:
            if len(selected_agents) >= target_count:
                break
            
            agents_with_role = [agent for agent in available_agents if agent.role == role]
            for agent in agents_with_role:
                if agent not in selected_agents and len(selected_agents) < target_count:
                    selected_agents.append(agent)
        
        # Fill remaining slots with any available agents if needed
        remaining_agents = [agent for agent in available_agents if agent not in selected_agents]
        while len(selected_agents) < target_count and remaining_agents:
            # Prefer more active agents
            best_remaining = max(remaining_agents, key=lambda a: a.last_activity)
            selected_agents.append(best_remaining)
            remaining_agents.remove(best_remaining)
        
        return selected_agents
    
    def _select_coordinator(
        self, 
        selected_agents: List[Agent], 
        task_characteristics: TaskCharacteristics
    ) -> Optional[Agent]:
        """Select the best coordinator from the selected agents"""
        # Prefer specific roles for coordination
        coordinator_preferences = [
            AgentRole.PLANNING_COORDINATOR,
            AgentRole.CONSENSUS_FACILITATOR,
            AgentRole.ARCHITECTURE_REVIEWER,
            AgentRole.GENERALIST
        ]
        
        for preferred_role in coordinator_preferences:
            for agent in selected_agents:
                if agent.role == preferred_role:
                    return agent
        
        # Fallback to first agent if no preferred coordinator found
        return selected_agents[0] if selected_agents else None
    
    def _select_default_agents(self, available_agent_ids: Optional[List[str]]) -> Tuple[List[str], Optional[str]]:
        """Fallback agent selection when no specific mapping exists"""
        if available_agent_ids is None:
            available_agents = list(self.communication_system.agents.values())
        else:
            available_agents = [
                self.communication_system.agents[agent_id] 
                for agent_id in available_agent_ids 
                if agent_id in self.communication_system.agents
            ]
        
        # Select up to 2 agents by default
        max_default = min(2, self.system_capabilities.recommended_max_agents)
        selected_agents = available_agents[:max_default]
        
        selected_agent_ids = [agent.agent_id for agent in selected_agents]
        coordinator_id = selected_agent_ids[0] if selected_agent_ids else None
        
        return selected_agent_ids, coordinator_id
    
    def analyze_task_from_prompt(self, prompt: str, files: Optional[List[str]] = None) -> TaskCharacteristics:
        """
        Analyze a prompt and files to determine task characteristics
        
        Args:
            prompt: The user's prompt/request
            files: Optional list of files being processed
            
        Returns:
            TaskCharacteristics object
        """
        prompt_lower = prompt.lower()
        
        # Determine task type based on keywords
        task_type = TaskType.GENERAL_ANALYSIS  # Default
        
        if any(keyword in prompt_lower for keyword in ["review", "code review", "analyze code"]):
            task_type = TaskType.CODE_REVIEW
        elif any(keyword in prompt_lower for keyword in ["security", "vulnerability", "exploit", "secure"]):
            task_type = TaskType.SECURITY_ANALYSIS
        elif any(keyword in prompt_lower for keyword in ["performance", "optimize", "speed", "efficiency"]):
            task_type = TaskType.PERFORMANCE_OPTIMIZATION
        elif any(keyword in prompt_lower for keyword in ["architecture", "design", "structure", "pattern"]):
            task_type = TaskType.ARCHITECTURE_DESIGN
        elif any(keyword in prompt_lower for keyword in ["debug", "error", "bug", "fix", "troubleshoot"]):
            task_type = TaskType.DEBUGGING
        elif any(keyword in prompt_lower for keyword in ["plan", "planning", "roadmap", "strategy"]):
            task_type = TaskType.PLANNING
        elif any(keyword in prompt_lower for keyword in ["consensus", "agreement", "vote", "decide"]):
            task_type = TaskType.CONSENSUS_BUILDING
        elif any(keyword in prompt_lower for keyword in ["parallel", "concurrent", "multiple"]):
            task_type = TaskType.PARALLEL_THINKING
        
        # Determine complexity based on prompt length and file count
        file_count = len(files) if files else 0
        prompt_length = len(prompt)
        
        if file_count > 10 or prompt_length > 1000:
            complexity = TaskComplexity.VERY_COMPLEX
        elif file_count > 5 or prompt_length > 500:
            complexity = TaskComplexity.COMPLEX
        elif file_count > 2 or prompt_length > 200:
            complexity = TaskComplexity.MODERATE
        else:
            complexity = TaskComplexity.SIMPLE
        
        # Determine focus areas
        requires_security_focus = any(keyword in prompt_lower for keyword in ["security", "vulnerability", "secure"])
        requires_performance_focus = any(keyword in prompt_lower for keyword in ["performance", "optimize", "speed"])
        requires_architecture_review = any(keyword in prompt_lower for keyword in ["architecture", "design", "structure"])
        requires_consensus = any(keyword in prompt_lower for keyword in ["consensus", "agreement", "multiple"])
        requires_deep_analysis = any(keyword in prompt_lower for keyword in ["deep", "thorough", "comprehensive", "detailed"])
        
        return TaskCharacteristics(
            task_type=task_type,
            complexity=complexity,
            requires_security_focus=requires_security_focus,
            requires_performance_focus=requires_performance_focus,
            requires_architecture_review=requires_architecture_review,
            requires_consensus=requires_consensus,
            requires_deep_analysis=requires_deep_analysis,
            file_count=file_count,
            estimated_processing_time=60.0 + (file_count * 10.0),  # Rough estimate
            priority=5
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system capabilities"""
        return {
            "total_cores": self.system_capabilities.total_cores,
            "available_cores": self.system_capabilities.available_cores,
            "cpu_architecture": self.system_capabilities.cpu_architecture,
            "system_type": self.system_capabilities.system_type,
            "memory_gb": self.system_capabilities.memory_gb,
            "is_high_performance": self.system_capabilities.is_high_performance,
            "supports_parallel_processing": self.system_capabilities.supports_parallel_processing,
            "recommended_max_agents": self.system_capabilities.recommended_max_agents,
            "registered_agents": len(self.communication_system.agents),
            "active_agents": len([a for a in self.communication_system.agents.values() if a.status.value == "active"])
        }


# Global singleton instance
_agent_selector_instance = None


def get_automatic_agent_selector(communication_system: Optional[AgentCommunicationSystem] = None) -> AutomaticAgentSelector:
    """Get the global automatic agent selector instance (singleton pattern)"""
    global _agent_selector_instance
    if _agent_selector_instance is None:
        if communication_system is None:
            from utils.agent_communication import get_agent_communication_system
            communication_system = get_agent_communication_system()
        _agent_selector_instance = AutomaticAgentSelector(communication_system)
    return _agent_selector_instance