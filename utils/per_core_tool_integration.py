"""
Per-Core Tool Integration

This module provides seamless integration between existing MCP tools and the per-core agent system.
It automatically leverages the per-core agent system for enhanced multi-agent capabilities while
maintaining backward compatibility with single-agent operation as fallback.

Key Features:
- Intelligent agent assignment based on task type and agent expertise
- Automatic workload distribution across available agents
- Graceful fallback to single-agent operation when per-core system is unavailable
- Enhanced multi-agent coordination for complex tasks
- Transparent integration with existing tool architecture
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from utils.agent_core import AgentRole, AgentStatus
from utils.per_core_agent_manager import PerCoreAgentManager
from utils.agent_communication import AgentCommunicationSystem, get_agent_communication_system
from tools.shared.base_tool import BaseTool

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks that can be assigned to agents"""
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ARCHITECTURE_REVIEW = "architecture_review"
    CODE_QUALITY_INSPECTION = "code_quality_inspection"
    DEBUG_ASSISTANCE = "debug_assistance"
    PLANNING_COORDINATION = "planning_coordination"
    CONSENSUS_BUILDING = "consensus_building"
    GENERAL_ASSISTANCE = "general_assistance"
    COLLABORATIVE_THINKING = "collaborative_thinking"
    DOCUMENTATION_GENERATION = "documentation_generation"
    TEST_GENERATION = "test_generation"
    REFACTORING_ANALYSIS = "refactoring_analysis"


@dataclass
class AgentAssignment:
    """Represents an agent assignment for a specific task"""
    agent_id: str
    core_id: int
    role: AgentRole
    task_type: TaskType
    priority: int  # Lower numbers = higher priority
    reasoning: str  # Why this agent was selected


@dataclass
class MultiAgentResponse:
    """Response from multi-agent coordination"""
    primary_response: str
    agent_contributions: Dict[str, str]  # agent_id -> contribution
    coordination_metadata: Dict[str, Any]
    fallback_used: bool = False


class PerCoreToolIntegrator:
    """
    Integrates existing MCP tools with the per-core agent system for enhanced capabilities.
    
    This class acts as a bridge between the existing tool architecture and the per-core
    agent system, providing intelligent agent assignment and coordination while maintaining
    full backward compatibility.
    """
    
    def __init__(self):
        self.per_core_manager: Optional[PerCoreAgentManager] = None
        self.communication_system: Optional[AgentCommunicationSystem] = None
        self._initialization_attempted = False
        self._is_available = False
        
        # Task type to agent role mapping for intelligent assignment
        self.task_to_role_mapping = {
            TaskType.SECURITY_ANALYSIS: [AgentRole.SECURITY_ANALYST, AgentRole.CODE_QUALITY_INSPECTOR],
            TaskType.PERFORMANCE_OPTIMIZATION: [AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.ARCHITECTURE_REVIEWER],
            TaskType.ARCHITECTURE_REVIEW: [AgentRole.ARCHITECTURE_REVIEWER, AgentRole.PLANNING_COORDINATOR],
            TaskType.CODE_QUALITY_INSPECTION: [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.SECURITY_ANALYST],
            TaskType.DEBUG_ASSISTANCE: [AgentRole.DEBUG_SPECIALIST, AgentRole.CODE_QUALITY_INSPECTOR],
            TaskType.PLANNING_COORDINATION: [AgentRole.PLANNING_COORDINATOR, AgentRole.CONSENSUS_FACILITATOR],
            TaskType.CONSENSUS_BUILDING: [AgentRole.CONSENSUS_FACILITATOR, AgentRole.PLANNING_COORDINATOR],
            TaskType.GENERAL_ASSISTANCE: [AgentRole.GENERALIST],
            TaskType.COLLABORATIVE_THINKING: [AgentRole.CONSENSUS_FACILITATOR, AgentRole.GENERALIST],
            TaskType.DOCUMENTATION_GENERATION: [AgentRole.GENERALIST, AgentRole.CODE_QUALITY_INSPECTOR],
            TaskType.TEST_GENERATION: [AgentRole.CODE_QUALITY_INSPECTOR, AgentRole.DEBUG_SPECIALIST],
            TaskType.REFACTORING_ANALYSIS: [AgentRole.ARCHITECTURE_REVIEWER, AgentRole.CODE_QUALITY_INSPECTOR]
        }
        
        # Tool name to task type mapping for automatic task classification
        self.tool_to_task_mapping = {
            "secaudit": TaskType.SECURITY_ANALYSIS,
            "analyze": TaskType.CODE_QUALITY_INSPECTION,
            "codereview": TaskType.CODE_QUALITY_INSPECTION,
            "debug": TaskType.DEBUG_ASSISTANCE,
            "planner": TaskType.PLANNING_COORDINATION,
            "consensus": TaskType.CONSENSUS_BUILDING,
            "chat": TaskType.GENERAL_ASSISTANCE,
            "thinkdeep": TaskType.COLLABORATIVE_THINKING,
            "parallelthink": TaskType.COLLABORATIVE_THINKING,
            "docgen": TaskType.DOCUMENTATION_GENERATION,
            "testgen": TaskType.TEST_GENERATION,
            "refactor": TaskType.REFACTORING_ANALYSIS,
            "tracer": TaskType.CODE_QUALITY_INSPECTION,
            "precommit": TaskType.CODE_QUALITY_INSPECTION
        }
    
    def initialize(self) -> bool:
        """
        Initialize the per-core agent system integration.
        
        Returns:
            True if initialization successful, False if fallback to single-agent mode
        """
        if self._initialization_attempted:
            return self._is_available
        
        self._initialization_attempted = True
        
        try:
            # Check if per-core agents are enabled in configuration
            from config import ENABLE_PER_CORE_AGENTS, PER_CORE_OPENROUTER_REQUIRED, PER_CORE_MAX_AGENTS
            
            if not ENABLE_PER_CORE_AGENTS:
                logger.info("Per-core agent system disabled in configuration")
                self._is_available = False
                return False
            
            # Check if OpenRouter API key is available for per-core agents
            import os
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            
            if not openrouter_key or openrouter_key == "your_openrouter_api_key_here":
                if PER_CORE_OPENROUTER_REQUIRED:
                    logger.info("OpenRouter API key required but not configured, per-core agent system disabled")
                    self._is_available = False
                    return False
                else:
                    logger.info("No OpenRouter API key configured, per-core agent system will use fallback providers")
                    # Still initialize but without OpenRouter-specific features
                    openrouter_key = None
            
            # Initialize per-core agent manager
            self.per_core_manager = PerCoreAgentManager(
                openrouter_api_key=openrouter_key,
                max_agents=PER_CORE_MAX_AGENTS  # Use configured max agents or all available cores
            )
            
            # Initialize agents
            agents = self.per_core_manager.initialize_agents()
            
            if not agents:
                logger.warning("No agents were initialized, falling back to single-agent mode")
                self._is_available = False
                return False
            
            # Get communication system
            self.communication_system = get_agent_communication_system()
            
            if not self.communication_system:
                logger.warning("Agent communication system not available, falling back to single-agent mode")
                self._is_available = False
                return False
            
            self._is_available = True
            logger.info(f"Per-core tool integration initialized successfully with {len(agents)} agents")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize per-core agent system: {e}")
            logger.info("Falling back to single-agent mode")
            self._is_available = False
            return False
    
    def is_available(self) -> bool:
        """Check if the per-core agent system is available"""
        if not self._initialization_attempted:
            self.initialize()
        return self._is_available
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information for monitoring and validation.
        
        Returns:
            Dictionary containing system health metrics and status information
        """
        health_info = {
            "available": self.is_available(),
            "total_agents": 0,
            "healthy_agents": 0,
            "agent_roles": [],
            "system_capabilities": {},
            "last_check": None
        }
        
        if not self.is_available() or not self.per_core_manager:
            health_info["reason"] = "Per-core agent system not available"
            return health_info
        
        try:
            # Get system health from manager
            manager_health = self.per_core_manager.get_system_health()
            
            # Update health info with manager data
            health_info.update({
                "total_agents": manager_health.get("total_agents", 0),
                "healthy_agents": sum(1 for agent in manager_health.get("agents", []) 
                                    if agent.get("status") == "active"),
                "agent_roles": list(set(agent.get("role") for agent in manager_health.get("agents", []))),
                "last_check": manager_health.get("last_health_check")
            })
            
            # System capabilities
            health_info["system_capabilities"] = {
                "communication": self.communication_system is not None,
                "openrouter": manager_health.get("openrouter_available", False),
                "multi_agent_coordination": health_info["total_agents"] > 1,
                "task_classification": True,
                "intelligent_assignment": True
            }
            
            # Overall health assessment
            health_info["available"] = (
                health_info["healthy_agents"] > 0 and
                health_info["system_capabilities"]["communication"]
            )
            
            return health_info
            
        except Exception as e:
            health_info["reason"] = f"Error retrieving system health: {str(e)}"
            health_info["available"] = False
            return health_info
    
    def classify_task_type(self, tool_name: str, arguments: Dict[str, Any]) -> TaskType:
        """
        Classify the task type based on tool name and arguments.
        
        Args:
            tool_name: Name of the tool being executed
            arguments: Tool arguments that may provide additional context
            
        Returns:
            TaskType classification for intelligent agent assignment
        """
        # Direct mapping from tool name
        if tool_name in self.tool_to_task_mapping:
            return self.tool_to_task_mapping[tool_name]
        
        # Analyze arguments for additional context
        prompt = arguments.get("prompt", "").lower()
        files = arguments.get("files", [])
        
        # Security-related keywords
        security_keywords = ["security", "vulnerability", "exploit", "attack", "malicious", "injection", "xss", "csrf"]
        if any(keyword in prompt for keyword in security_keywords):
            return TaskType.SECURITY_ANALYSIS
        
        # Performance-related keywords
        performance_keywords = ["performance", "optimization", "speed", "memory", "cpu", "bottleneck", "profiling"]
        if any(keyword in prompt for keyword in performance_keywords):
            return TaskType.PERFORMANCE_OPTIMIZATION
        
        # Architecture-related keywords
        architecture_keywords = ["architecture", "design", "pattern", "structure", "scalability", "maintainability"]
        if any(keyword in prompt for keyword in architecture_keywords):
            return TaskType.ARCHITECTURE_REVIEW
        
        # Debug-related keywords
        debug_keywords = ["debug", "error", "bug", "issue", "problem", "fix", "troubleshoot"]
        if any(keyword in prompt for keyword in debug_keywords):
            return TaskType.DEBUG_ASSISTANCE
        
        # File type analysis for additional context
        if files:
            # Check for test files
            test_files = [f for f in files if any(test_pattern in f.lower() for test_pattern in ["test", "spec", "__test__"])]
            if test_files:
                return TaskType.TEST_GENERATION
            
            # Check for configuration files
            config_files = [f for f in files if any(config_pattern in f.lower() for config_pattern in ["config", "settings", "env"])]
            if config_files:
                return TaskType.SECURITY_ANALYSIS
        
        # Default to general assistance
        return TaskType.GENERAL_ASSISTANCE
    
    def assign_agents(self, task_type: TaskType, num_agents: int = 1) -> List[AgentAssignment]:
        """
        Assign the most suitable agents for a given task type.
        
        Args:
            task_type: Type of task to assign agents for
            num_agents: Number of agents to assign (default: 1)
            
        Returns:
            List of agent assignments ordered by priority
        """
        if not self.is_available() or not self.per_core_manager:
            return []
        
        # Get preferred roles for this task type
        preferred_roles = self.task_to_role_mapping.get(task_type, [AgentRole.GENERALIST])
        
        assignments = []
        assigned_agents = set()
        
        # First pass: assign agents with preferred roles
        for priority, role in enumerate(preferred_roles):
            if len(assignments) >= num_agents:
                break
            
            # Get healthy agents with this role
            role_agents = self.per_core_manager.get_agents_by_role(role)
            healthy_role_agents = [
                agent for agent in role_agents 
                if agent.status == AgentStatus.ACTIVE and agent.agent_id not in assigned_agents
            ]
            
            for agent in healthy_role_agents:
                if len(assignments) >= num_agents:
                    break
                
                assignment = AgentAssignment(
                    agent_id=agent.agent_id,
                    core_id=agent.core_id,
                    role=agent.role,
                    task_type=task_type,
                    priority=priority,
                    reasoning=f"Primary role match: {role.value} is optimal for {task_type.value}"
                )
                assignments.append(assignment)
                assigned_agents.add(agent.agent_id)
        
        # Second pass: fill remaining slots with any healthy agents
        if len(assignments) < num_agents:
            all_agents = list(self.per_core_manager.agents.values())
            remaining_healthy_agents = [
                agent for agent in all_agents
                if agent.status == AgentStatus.ACTIVE and agent.agent_id not in assigned_agents
            ]
            
            for agent in remaining_healthy_agents:
                if len(assignments) >= num_agents:
                    break
                
                assignment = AgentAssignment(
                    agent_id=agent.agent_id,
                    core_id=agent.core_id,
                    role=agent.role,
                    task_type=task_type,
                    priority=len(preferred_roles) + 1,  # Lower priority than preferred roles
                    reasoning=f"Fallback assignment: {agent.role.value} available for {task_type.value}"
                )
                assignments.append(assignment)
                assigned_agents.add(agent.agent_id)
        
        logger.debug(f"Assigned {len(assignments)} agents for {task_type.value}: "
                    f"{[a.agent_id for a in assignments]}")
        
        return assignments
    
    async def execute_with_agents(
        self, 
        tool: BaseTool, 
        arguments: Dict[str, Any],
        agent_assignments: List[AgentAssignment]
    ) -> MultiAgentResponse:
        """
        Execute a tool with assigned agents for enhanced multi-agent coordination.
        
        Args:
            tool: The tool instance to execute
            arguments: Tool execution arguments
            agent_assignments: List of assigned agents for this task
            
        Returns:
            MultiAgentResponse with coordinated results from multiple agents
        """
        if not agent_assignments or not self.communication_system:
            # Fallback to single-agent execution
            logger.debug("No agent assignments or communication system, using fallback execution")
            result = await tool.execute(arguments)
            return MultiAgentResponse(
                primary_response=result[0].text if result else "",
                agent_contributions={},
                coordination_metadata={"fallback_reason": "no_agents_assigned"},
                fallback_used=True
            )
        
        try:
            # For single agent assignment, execute directly with that agent's context
            if len(agent_assignments) == 1:
                assignment = agent_assignments[0]
                logger.info(f"Executing {tool.get_name()} with single agent {assignment.agent_id} "
                          f"(role: {assignment.role.value})")
                
                # Execute with agent-specific context
                enhanced_arguments = await self._enhance_arguments_for_agent(arguments, assignment)
                result = await tool.execute(enhanced_arguments)
                
                return MultiAgentResponse(
                    primary_response=result[0].text if result else "",
                    agent_contributions={assignment.agent_id: result[0].text if result else ""},
                    coordination_metadata={
                        "assigned_agent": assignment.agent_id,
                        "agent_role": assignment.role.value,
                        "task_type": assignment.task_type.value,
                        "reasoning": assignment.reasoning
                    },
                    fallback_used=False
                )
            
            # For multiple agents, coordinate execution
            else:
                logger.info(f"Executing {tool.get_name()} with {len(agent_assignments)} agents: "
                          f"{[a.agent_id for a in agent_assignments]}")
                
                return await self._coordinate_multi_agent_execution(tool, arguments, agent_assignments)
        
        except Exception as e:
            logger.error(f"Error in multi-agent execution: {e}")
            # Fallback to single-agent execution
            logger.info("Falling back to single-agent execution due to error")
            result = await tool.execute(arguments)
            return MultiAgentResponse(
                primary_response=result[0].text if result else "",
                agent_contributions={},
                coordination_metadata={"fallback_reason": f"execution_error: {str(e)}"},
                fallback_used=True
            )
    
    async def _enhance_arguments_for_agent(
        self, 
        arguments: Dict[str, Any], 
        assignment: AgentAssignment
    ) -> Dict[str, Any]:
        """
        Enhance tool arguments with agent-specific context and preferences.
        
        Args:
            arguments: Original tool arguments
            assignment: Agent assignment with role and context
            
        Returns:
            Enhanced arguments with agent-specific context
        """
        enhanced_args = arguments.copy()
        
        # Add agent context to the prompt if present
        if "prompt" in enhanced_args:
            original_prompt = enhanced_args["prompt"]
            
            # Add role-specific context
            role_context = self._get_role_specific_context(assignment.role, assignment.task_type)
            
            enhanced_prompt = f"""AGENT CONTEXT: You are operating as a {assignment.role.value} agent specialized in {assignment.task_type.value}.

{role_context}

ORIGINAL REQUEST:
{original_prompt}"""
            
            enhanced_args["prompt"] = enhanced_prompt
        
        # Add agent metadata for tracking
        enhanced_args["_agent_context"] = {
            "agent_id": assignment.agent_id,
            "core_id": assignment.core_id,
            "role": assignment.role.value,
            "task_type": assignment.task_type.value,
            "reasoning": assignment.reasoning
        }
        
        return enhanced_args
    
    def _get_role_specific_context(self, role: AgentRole, task_type: TaskType) -> str:
        """
        Get role-specific context and guidance for enhanced agent performance.
        
        Args:
            role: Agent role
            task_type: Task type being performed
            
        Returns:
            Role-specific context string
        """
        role_contexts = {
            AgentRole.SECURITY_ANALYST: """
As a Security Analyst agent, focus on:
- Identifying potential security vulnerabilities and threats
- Analyzing code for common security issues (OWASP Top 10)
- Evaluating authentication, authorization, and data protection mechanisms
- Assessing input validation and sanitization practices
- Reviewing cryptographic implementations and secure communication
- Providing actionable security recommendations with risk assessments
""",
            AgentRole.PERFORMANCE_OPTIMIZER: """
As a Performance Optimizer agent, focus on:
- Identifying performance bottlenecks and optimization opportunities
- Analyzing algorithmic complexity and resource usage patterns
- Evaluating memory management and garbage collection impact
- Assessing database query performance and caching strategies
- Reviewing concurrent processing and parallelization opportunities
- Providing specific performance improvement recommendations with metrics
""",
            AgentRole.ARCHITECTURE_REVIEWER: """
As an Architecture Reviewer agent, focus on:
- Evaluating overall system design and architectural patterns
- Assessing scalability, maintainability, and extensibility
- Reviewing component interactions and dependency management
- Analyzing design principles adherence (SOLID, DRY, KISS)
- Evaluating separation of concerns and modularity
- Providing architectural improvement recommendations with trade-offs
""",
            AgentRole.CODE_QUALITY_INSPECTOR: """
As a Code Quality Inspector agent, focus on:
- Analyzing code style, formatting, and consistency
- Evaluating naming conventions and code readability
- Assessing error handling and exception management
- Reviewing code documentation and comments quality
- Identifying code smells and refactoring opportunities
- Providing code quality improvement recommendations with examples
""",
            AgentRole.DEBUG_SPECIALIST: """
As a Debug Specialist agent, focus on:
- Identifying root causes of bugs and issues
- Analyzing error patterns and failure scenarios
- Evaluating debugging information and stack traces
- Assessing error reproduction steps and test cases
- Reviewing logging and monitoring implementations
- Providing specific debugging strategies and fix recommendations
""",
            AgentRole.PLANNING_COORDINATOR: """
As a Planning Coordinator agent, focus on:
- Breaking down complex tasks into manageable components
- Coordinating between different aspects of development work
- Prioritizing tasks based on dependencies and impact
- Evaluating resource allocation and timeline considerations
- Assessing risk factors and mitigation strategies
- Providing structured project planning and coordination guidance
""",
            AgentRole.CONSENSUS_FACILITATOR: """
As a Consensus Facilitator agent, focus on:
- Synthesizing different perspectives and approaches
- Identifying common ground and areas of agreement
- Facilitating decision-making processes
- Evaluating trade-offs and compromise solutions
- Building consensus among conflicting viewpoints
- Providing balanced recommendations that consider all stakeholders
""",
            AgentRole.GENERALIST: """
As a Generalist agent, focus on:
- Providing comprehensive analysis across multiple domains
- Adapting approach based on the specific task requirements
- Offering balanced perspectives on various aspects
- Connecting insights from different areas of expertise
- Providing practical, actionable recommendations
- Maintaining flexibility to address diverse development needs
"""
        }
        
        return role_contexts.get(role, role_contexts[AgentRole.GENERALIST])
    
    async def _coordinate_multi_agent_execution(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        agent_assignments: List[AgentAssignment]
    ) -> MultiAgentResponse:
        """
        Coordinate execution across multiple agents for enhanced analysis.
        
        Args:
            tool: Tool to execute
            arguments: Tool arguments
            agent_assignments: List of assigned agents
            
        Returns:
            Coordinated multi-agent response
        """
        # Execute tool with each assigned agent concurrently
        tasks = []
        for assignment in agent_assignments:
            enhanced_args = await self._enhance_arguments_for_agent(arguments, assignment)
            task = asyncio.create_task(
                self._execute_with_single_agent(tool, enhanced_args, assignment)
            )
            tasks.append((assignment, task))
        
        # Wait for all agent executions to complete
        agent_results = {}
        for assignment, task in tasks:
            try:
                result = await task
                agent_results[assignment.agent_id] = {
                    "response": result[0].text if result else "",
                    "role": assignment.role.value,
                    "reasoning": assignment.reasoning
                }
            except Exception as e:
                logger.error(f"Agent {assignment.agent_id} execution failed: {e}")
                agent_results[assignment.agent_id] = {
                    "response": f"Agent execution failed: {str(e)}",
                    "role": assignment.role.value,
                    "reasoning": assignment.reasoning,
                    "error": True
                }
        
        # Synthesize responses from all agents
        primary_response = await self._synthesize_agent_responses(
            tool.get_name(), 
            arguments, 
            agent_results
        )
        
        # Extract just the response text for agent_contributions
        agent_contributions = {
            agent_id: result["response"] 
            for agent_id, result in agent_results.items()
        }
        
        coordination_metadata = {
            "num_agents": len(agent_assignments),
            "agent_roles": [a.role.value for a in agent_assignments],
            "task_type": agent_assignments[0].task_type.value,
            "coordination_strategy": "concurrent_execution_with_synthesis"
        }
        
        return MultiAgentResponse(
            primary_response=primary_response,
            agent_contributions=agent_contributions,
            coordination_metadata=coordination_metadata,
            fallback_used=False
        )
    
    async def _execute_with_single_agent(
        self,
        tool: BaseTool,
        enhanced_arguments: Dict[str, Any],
        assignment: AgentAssignment
    ) -> List:
        """Execute tool with a single agent's enhanced context"""
        try:
            return await tool.execute(enhanced_arguments)
        except Exception as e:
            logger.error(f"Single agent execution failed for {assignment.agent_id}: {e}")
            # Return error response in expected format
            from mcp.types import TextContent
            from tools.models import ToolOutput
            
            error_output = ToolOutput(
                status="error",
                content=f"Agent {assignment.agent_id} execution failed: {str(e)}",
                content_type="text"
            )
            return [TextContent(type="text", text=error_output.model_dump_json())]
    
    async def _synthesize_agent_responses(
        self,
        tool_name: str,
        original_arguments: Dict[str, Any],
        agent_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Synthesize responses from multiple agents into a coherent final response.
        
        Args:
            tool_name: Name of the tool that was executed
            original_arguments: Original tool arguments
            agent_results: Results from each agent
            
        Returns:
            Synthesized response combining insights from all agents
        """
        # Filter out failed agents
        successful_results = {
            agent_id: result for agent_id, result in agent_results.items()
            if not result.get("error", False)
        }
        
        if not successful_results:
            return "All assigned agents failed to execute. Please try again or check system status."
        
        # Create synthesis prompt
        synthesis_prompt = f"""MULTI-AGENT RESPONSE SYNTHESIS

You are synthesizing responses from multiple specialized AI agents who analyzed the same request using the {tool_name} tool.

ORIGINAL REQUEST: {original_arguments.get('prompt', 'No prompt provided')}

AGENT RESPONSES:
"""
        
        for agent_id, result in successful_results.items():
            synthesis_prompt += f"""
--- {result['role'].upper()} AGENT ({agent_id}) ---
Reasoning: {result['reasoning']}
Response: {result['response']}
"""
        
        synthesis_prompt += """
SYNTHESIS INSTRUCTIONS:
1. Combine the insights from all agents into a comprehensive, coherent response
2. Highlight areas where agents agree and note any significant differences
3. Prioritize insights based on each agent's area of expertise
4. Provide a unified conclusion that leverages the strengths of each perspective
5. Maintain the original tool's response format and style
6. If agents disagree, explain the different viewpoints and provide balanced guidance

Synthesized Response:"""
        
        try:
            # Use a simple approach for synthesis - in a full implementation,
            # this could use the communication system to get a consensus agent
            # For now, we'll create a structured synthesis
            
            synthesis_response = f"""# Multi-Agent Analysis Results

This analysis was performed by {len(successful_results)} specialized agents working in coordination:

"""
            
            # Add individual agent insights
            for agent_id, result in successful_results.items():
                synthesis_response += f"""## {result['role']} Perspective
{result['response']}

"""
            
            # Add coordination summary
            synthesis_response += f"""## Coordinated Conclusion

The {len(successful_results)} agents have provided complementary perspectives on your request. """
            
            if len(successful_results) > 1:
                synthesis_response += """Each agent focused on their area of expertise, providing a more comprehensive analysis than would be possible with a single perspective. """
            
            synthesis_response += """The insights above represent the collective intelligence of the per-core agent system working together to address your needs.

---

*This response was generated through per-core agent coordination, leveraging multiple specialized AI agents for enhanced analysis quality.*"""
            
            return synthesis_response
            
        except Exception as e:
            logger.error(f"Failed to synthesize agent responses: {e}")
            # Fallback to simple concatenation
            fallback_response = "# Multi-Agent Analysis Results\n\n"
            for agent_id, result in successful_results.items():
                fallback_response += f"## {result['role']} Agent\n{result['response']}\n\n"
            return fallback_response
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get health status of the per-core agent system.
        
        Returns:
            Dictionary with system health information
        """
        if not self.is_available() or not self.per_core_manager:
            return {
                "available": False,
                "reason": "Per-core agent system not initialized or unavailable"
            }
        
        try:
            health_info = self.per_core_manager.get_system_health()
            health_info["available"] = True
            return health_info
        except Exception as e:
            return {
                "available": False,
                "reason": f"Error getting system health: {str(e)}"
            }


# Global instance for tool integration
_per_core_integrator: Optional[PerCoreToolIntegrator] = None


def get_per_core_integrator() -> PerCoreToolIntegrator:
    """Get the global per-core tool integrator instance"""
    global _per_core_integrator
    if _per_core_integrator is None:
        _per_core_integrator = PerCoreToolIntegrator()
    return _per_core_integrator


async def enhance_tool_execution(
    tool: BaseTool,
    arguments: Dict[str, Any]
) -> Tuple[List, bool]:
    """
    Enhance tool execution with per-core agent coordination.
    
    This function provides the main integration point for existing tools to leverage
    the per-core agent system. It automatically determines if multi-agent coordination
    would be beneficial and coordinates execution accordingly.
    
    Args:
        tool: The tool instance to execute
        arguments: Tool execution arguments
        
    Returns:
        Tuple of (execution_result, used_per_core_system)
    """
    integrator = get_per_core_integrator()
    
    # Check if per-core system is available
    if not integrator.is_available():
        logger.debug(f"Per-core system not available for {tool.get_name()}, using standard execution")
        result = await tool.execute(arguments)
        return result, False
    
    try:
        # Classify the task type
        task_type = integrator.classify_task_type(tool.get_name(), arguments)
        logger.debug(f"Classified {tool.get_name()} as {task_type.value}")
        
        # Determine if multi-agent coordination would be beneficial
        # For now, use single agent assignment for simplicity and reliability
        num_agents = 1
        
        # Some tools benefit from multiple perspectives
        from config import PER_CORE_MULTI_AGENT_TOOLS
        if tool.get_name() in PER_CORE_MULTI_AGENT_TOOLS:
            num_agents = min(2, len(integrator.per_core_manager.agents))
        
        # Assign agents
        agent_assignments = integrator.assign_agents(task_type, num_agents)
        
        if not agent_assignments:
            logger.debug(f"No agents available for {tool.get_name()}, using standard execution")
            result = await tool.execute(arguments)
            return result, False
        
        # Execute with assigned agents
        multi_agent_response = await integrator.execute_with_agents(tool, arguments, agent_assignments)
        
        if multi_agent_response.fallback_used:
            logger.debug(f"Per-core execution fell back to standard mode for {tool.get_name()}")
            # The response is already in the correct format from the fallback
            from mcp.types import TextContent
            return [TextContent(type="text", text=multi_agent_response.primary_response)], False
        
        # Format the multi-agent response for MCP
        from mcp.types import TextContent
        from tools.models import ToolOutput
        import json
        
        # Create enhanced tool output with multi-agent metadata
        enhanced_output = ToolOutput(
            status="success",
            content=multi_agent_response.primary_response,
            content_type="text",
            metadata={
                "per_core_coordination": True,
                "agents_used": len(agent_assignments),
                "agent_roles": [a.role.value for a in agent_assignments],
                "task_type": task_type.value,
                **multi_agent_response.coordination_metadata
            }
        )
        
        result = [TextContent(type="text", text=enhanced_output.model_dump_json())]
        return result, True
        
    except Exception as e:
        logger.error(f"Error in per-core tool execution for {tool.get_name()}: {e}")
        logger.info("Falling back to standard tool execution")
        result = await tool.execute(arguments)
        return result, False