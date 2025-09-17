"""
Per-Core Agent Manager

This module implements the core infrastructure for per-CPU-core agent management.
Each available CPU core acts as an autonomous AI agent with its own OpenRouter
connection for independent thinking sessions.

Key Features:
- CPU core detection and agent lifecycle management
- Agent role assignment strategy based on available cores
- OpenRouter-specific configuration for each agent
- Agent health monitoring and failure recovery
- Graceful system shutdown and cleanup
"""

import logging
import multiprocessing
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from utils.agent_core import Agent, AgentRole, AgentStatus
from utils.agent_communication import AgentCommunicationSystem, get_agent_communication_system
from utils.agent_api_client import AgentAPIClient
from providers.base import ProviderType
from utils.per_core_agent_monitoring import get_per_core_agent_monitor, initialize_monitoring
from utils.agent_alerting import get_alerting_system

logger = logging.getLogger(__name__)


@dataclass
class PerCoreAgentStatus:
    """Tracks the status and health of each per-core agent"""
    agent_id: str
    core_id: int
    role: AgentRole
    status: AgentStatus
    openrouter_connected: bool
    active_thinking_sessions: int
    total_api_calls: int
    success_rate: float
    last_activity: float
    memory_usage_mb: float
    
    def is_healthy(self) -> bool:
        """Check if the agent is in a healthy state"""
        return (
            self.status in [AgentStatus.ACTIVE, AgentStatus.WAITING, AgentStatus.THINKING] and
            self.openrouter_connected and
            self.success_rate >= 0.5 and  # At least 50% success rate
            (time.time() - self.last_activity) < 300  # Active within last 5 minutes
        )
    
    def needs_attention(self) -> bool:
        """Check if the agent needs attention or intervention"""
        return (
            not self.openrouter_connected or
            self.success_rate < 0.3 or  # Less than 30% success rate
            (time.time() - self.last_activity) > 600  # Inactive for more than 10 minutes
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "core_id": self.core_id,
            "role": self.role.value,
            "status": self.status.value,
            "openrouter_connected": self.openrouter_connected,
            "active_thinking_sessions": self.active_thinking_sessions,
            "total_api_calls": self.total_api_calls,
            "success_rate": self.success_rate,
            "last_activity": self.last_activity,
            "memory_usage_mb": self.memory_usage_mb,
            "is_healthy": self.is_healthy(),
            "needs_attention": self.needs_attention()
        }


class PerCoreAgentManager:
    """
    Central coordinator that manages agent lifecycle and per-core agent coordination.
    Creates one agent per available CPU core with OpenRouter connections.
    """
    
    def __init__(self, openrouter_api_key: Optional[str] = None, max_agents: Optional[int] = None, 
                 config_manager: Optional['PerCoreAgentConfigManager'] = None):
        """
        Initialize the per-core agent manager
        
        Args:
            openrouter_api_key: OpenRouter API key for agent connections (deprecated, use config_manager)
            max_agents: Maximum number of agents to create (deprecated, use config_manager)
            config_manager: Configuration manager instance for secure configuration handling
        """
        # Initialize configuration management
        if config_manager:
            self.config_manager = config_manager
            self.system_config = config_manager.get_config()
        else:
            # Fallback to legacy parameters or create default config manager
            from utils.per_core_agent_config import get_per_core_agent_config_manager
            self.config_manager = get_per_core_agent_config_manager()
            self.system_config = self.config_manager.get_config()
            
            # Override with legacy parameters if provided
            if openrouter_api_key:
                self.system_config.openrouter_api_key = openrouter_api_key
            if max_agents:
                self.system_config.max_agents = max_agents
        
        # Extract configuration values
        self.openrouter_api_key = self.system_config.openrouter_api_key
        self.max_agents = self.system_config.max_agents
        
        # Core detection
        self.detected_cores = self._detect_cpu_cores()
        self.effective_max_agents = min(
            max_agents or self.detected_cores,
            self.detected_cores
        )
        
        # Agent management
        self.agents: Dict[int, Agent] = {}  # core_id -> Agent
        self.agent_statuses: Dict[str, PerCoreAgentStatus] = {}  # agent_id -> status
        self.communication_system: Optional[AgentCommunicationSystem] = None
        
        # Thread safety
        self._agents_lock = threading.RLock()
        self._status_lock = threading.RLock()
        
        # Health monitoring
        self._shutdown = False
        self._health_monitor_thread: Optional[threading.Thread] = None
        
        # Monitoring and alerting integration
        self._monitor = None
        self._alerting_system = None
        
        # Emergency and degradation mode tracking
        self._emergency_mode = False
        self._emergency_local_mode = False
        self._emergency_mode_activated_at = 0.0
        self._emergency_mode_reason = ""
        self._redistribution_history: List[Dict[str, Any]] = []
        
        # Configuration change callback
        self.config_manager.add_change_callback(self._on_config_changed)
        
        # Check for recovery state from previous shutdown
        self._check_recovery_state()
        
        logger.info(f"PerCoreAgentManager initialized: {self.detected_cores} cores detected, "
                   f"will create {self.effective_max_agents} agents, config validation: {len(self.system_config.validate()) == 0}")
    
    def _detect_cpu_cores(self) -> int:
        """
        Detect the number of available CPU cores using multiple methods
        
        Returns:
            Number of detected CPU cores
        """
        try:
            # Try os.cpu_count() first (logical cores)
            os_cores = os.cpu_count()
            
            # Try multiprocessing.cpu_count() as backup
            mp_cores = multiprocessing.cpu_count()
            
            # Use the more conservative estimate
            detected = min(os_cores or 4, mp_cores or 4)
            
            logger.info(f"CPU core detection: os.cpu_count()={os_cores}, "
                       f"multiprocessing.cpu_count()={mp_cores}, using {detected}")
            
            return max(detected, 1)  # Ensure at least 1 core
            
        except Exception as e:
            logger.warning(f"Error detecting CPU cores: {e}, defaulting to 4")
            return 4
    
    def _assign_agent_roles(self, num_agents: int) -> List[AgentRole]:
        """
        Assign roles to agents based on available cores
        
        Args:
            num_agents: Number of agents to assign roles for
            
        Returns:
            List of AgentRole assignments
        """
        # Priority order for role assignment based on importance and specialization
        role_priority = [
            AgentRole.SECURITY_ANALYST,      # Core 0 - Critical security analysis
            AgentRole.PERFORMANCE_OPTIMIZER, # Core 1 - Performance monitoring
            AgentRole.ARCHITECTURE_REVIEWER, # Core 2 - System design review
            AgentRole.CODE_QUALITY_INSPECTOR,# Core 3 - Code quality checks
            AgentRole.DEBUG_SPECIALIST,      # Core 4 - Debugging assistance
            AgentRole.PLANNING_COORDINATOR,  # Core 5 - Task coordination
            AgentRole.CONSENSUS_FACILITATOR, # Core 6 - Decision facilitation
            AgentRole.GENERALIST            # Core 7+ - General assistance
        ]
        
        assigned_roles = []
        
        for i in range(num_agents):
            if i < len(role_priority):
                assigned_roles.append(role_priority[i])
            else:
                # For additional cores beyond our specialized roles, assign Generalist
                assigned_roles.append(AgentRole.GENERALIST)
        
        logger.debug(f"Assigned roles for {num_agents} agents: {[role.value for role in assigned_roles]}")
        return assigned_roles
    
    def initialize_agents(self) -> List[Agent]:
        """
        Initialize agents for each CPU core with appropriate role assignments
        
        Returns:
            List of created Agent instances
        """
        with self._agents_lock:
            if self.agents:
                logger.warning("Agents already initialized, skipping initialization")
                return list(self.agents.values())
            
            # Get communication system
            self.communication_system = get_agent_communication_system()
            
            # Assign roles based on number of agents
            assigned_roles = self._assign_agent_roles(self.effective_max_agents)
            
            created_agents = []
            
            # Import error handling and logging
            from utils.per_core_error_handling import get_per_core_error_handler
            from utils.per_core_logging import log_system_event, log_agent_activity
            
            error_handler = get_per_core_error_handler()
            
            # Log system initialization event
            log_system_event(
                event_type="agent_initialization_start",
                event_details={
                    "total_cores": self.detected_cores,
                    "effective_max_agents": self.effective_max_agents,
                    "assigned_roles": [role.value for role in assigned_roles],
                    "openrouter_configured": bool(self.openrouter_api_key)
                },
                severity="INFO"
            )
            
            initialization_errors = []
            
            for core_id in range(self.effective_max_agents):
                try:
                    role = assigned_roles[core_id]
                    agent_id = f"core_{core_id}_{role.value}"
                    
                    # Log agent initialization start
                    log_agent_activity(
                        agent_id=agent_id,
                        activity_type="agent_initialization_start",
                        activity_details={
                            "core_id": core_id,
                            "role": role.value,
                            "openrouter_api_key_available": bool(self.openrouter_api_key)
                        }
                    )
                    
                    # Create agent through communication system
                    agent = self.communication_system.register_agent(
                        core_id=core_id,
                        role=role,
                        agent_id=agent_id
                    )
                    
                    # Store agent by core ID
                    self.agents[core_id] = agent
                    created_agents.append(agent)
                    
                    # Initialize agent status tracking
                    self._initialize_agent_status(agent)
                    
                    # Configure OpenRouter for this agent if API key provided
                    if self.openrouter_api_key:
                        try:
                            self._configure_agent_openrouter(agent)
                            
                            # Verify OpenRouter configuration was successful
                            with self._status_lock:
                                if (agent.agent_id in self.agent_statuses and 
                                    not self.agent_statuses[agent.agent_id].openrouter_connected):
                                    logger.warning(f"Agent {agent.agent_id} initialized but OpenRouter connection failed")
                                    
                                    # Log OpenRouter configuration failure
                                    log_agent_activity(
                                        agent_id=agent.agent_id,
                                        activity_type="openrouter_configuration_failed",
                                        activity_details={
                                            "core_id": core_id,
                                            "role": role.value,
                                            "api_key_provided": bool(self.openrouter_api_key)
                                        },
                                        success=False,
                                        error_message="OpenRouter connection verification failed"
                                    )
                                else:
                                    # Log successful OpenRouter configuration
                                    log_agent_activity(
                                        agent_id=agent.agent_id,
                                        activity_type="openrouter_configuration_success",
                                        activity_details={
                                            "core_id": core_id,
                                            "role": role.value,
                                            "preferred_models": getattr(self._create_openrouter_config_for_role(role), 'preferred_models', [])
                                        },
                                        success=True
                                    )
                        except Exception as openrouter_error:
                            logger.error(f"Failed to configure OpenRouter for agent {agent.agent_id}: {openrouter_error}")
                            
                            # Handle OpenRouter configuration error
                            # Note: Error handler is async but we're in sync context
                            # Log error for now, async error handling can be added later
                            logger.error(f"OpenRouter configuration error for agent {agent.agent_id}: {openrouter_error}")
                            # TODO: Consider making initialize_agents async to properly handle errors
                            
                            initialization_errors.append({
                                "core_id": core_id,
                                "agent_id": agent.agent_id,
                                "error_type": "openrouter_configuration",
                                "error": str(openrouter_error)
                            })
                    else:
                        logger.info(f"No OpenRouter API key provided, agent {agent.agent_id} will use default providers")
                    
                    # Log successful agent initialization
                    log_agent_activity(
                        agent_id=agent.agent_id,
                        activity_type="agent_initialization_success",
                        activity_details={
                            "core_id": core_id,
                            "role": role.value,
                            "openrouter_configured": bool(self.openrouter_api_key and 
                                                        agent.agent_id in self.agent_statuses and 
                                                        self.agent_statuses[agent.agent_id].openrouter_connected)
                        },
                        success=True
                    )
                    
                    logger.info(f"Initialized agent {agent.agent_id} on core {core_id} with role {role.value}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize agent for core {core_id}: {e}")
                    
                    # Handle agent initialization error
                    # Note: Error handler is async but we're in sync context
                    # Log error for now, async error handling can be added later
                    logger.error(f"Agent initialization error for core {core_id}: {e}")
                    # TODO: Consider making initialize_agents async to properly handle errors
                    
                    initialization_errors.append({
                        "core_id": core_id,
                        "agent_id": f"core_{core_id}_{role.value}",
                        "error_type": "agent_initialization",
                        "error": str(e)
                    })
                    continue
            
            # Initialize monitoring and alerting
            try:
                self._initialize_monitoring_and_alerting()
            except Exception as e:
                logger.error(f"Failed to initialize monitoring and alerting: {e}")
                initialization_errors.append({
                    "component": "monitoring_and_alerting",
                    "error_type": "monitoring_initialization",
                    "error": str(e)
                })
            
            # Start health monitoring
            try:
                self._start_health_monitoring()
            except Exception as e:
                logger.error(f"Failed to start health monitoring: {e}")
                initialization_errors.append({
                    "component": "health_monitoring",
                    "error_type": "health_monitoring_start",
                    "error": str(e)
                })
            
            # Log final initialization results
            if initialization_errors:
                log_system_event(
                    event_type="agent_initialization_completed_with_errors",
                    event_details={
                        "successful_agents": len(created_agents),
                        "total_attempted": self.effective_max_agents,
                        "errors": initialization_errors,
                        "success_rate": len(created_agents) / self.effective_max_agents if self.effective_max_agents > 0 else 0
                    },
                    severity="WARNING",
                    affected_agents=[agent.agent_id for agent in created_agents]
                )
                
                logger.warning(f"Initialized {len(created_agents)} agents across {self.effective_max_agents} cores "
                             f"with {len(initialization_errors)} errors")
            else:
                log_system_event(
                    event_type="agent_initialization_completed_successfully",
                    event_details={
                        "successful_agents": len(created_agents),
                        "total_cores": self.detected_cores,
                        "effective_max_agents": self.effective_max_agents,
                        "openrouter_configured": bool(self.openrouter_api_key)
                    },
                    severity="INFO",
                    affected_agents=[agent.agent_id for agent in created_agents]
                )
                
                logger.info(f"Successfully initialized {len(created_agents)} agents across {self.effective_max_agents} cores")
            
            return created_agents
    
    def _initialize_agent_status(self, agent: Agent) -> None:
        """Initialize status tracking for an agent"""
        with self._status_lock:
            status = PerCoreAgentStatus(
                agent_id=agent.agent_id,
                core_id=agent.core_id,
                role=agent.role,
                status=agent.status,
                openrouter_connected=False,  # Will be updated when OpenRouter is configured
                active_thinking_sessions=0,
                total_api_calls=0,
                success_rate=1.0,  # Start optimistic
                last_activity=agent.last_activity,
                memory_usage_mb=0.0  # Will be updated by health monitoring
            )
            self.agent_statuses[agent.agent_id] = status
    
    def _configure_agent_openrouter(self, agent: Agent) -> None:
        """Configure OpenRouter connection for an agent"""
        try:
            # Get the agent's API client from communication system
            if self.communication_system:
                api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                if api_client:
                    # Validate API key first
                    if not self._validate_openrouter_api_key(self.openrouter_api_key):
                        logger.error(f"Invalid OpenRouter API key for agent {agent.agent_id}")
                        return
                    
                    # Create role-specific OpenRouter configuration
                    openrouter_config = self._create_openrouter_config_for_role(agent.role)
                    
                    # Configure for OpenRouter-only operation
                    api_client.configure_openrouter_only(self.openrouter_api_key, openrouter_config)
                    
                    # Perform connection health check
                    connection_healthy = self._check_openrouter_connection_health(agent.agent_id, api_client)
                    
                    # Update status to reflect OpenRouter connection
                    with self._status_lock:
                        if agent.agent_id in self.agent_statuses:
                            self.agent_statuses[agent.agent_id].openrouter_connected = connection_healthy
                    
                    if connection_healthy:
                        logger.info(f"Successfully configured OpenRouter for agent {agent.agent_id} with role {agent.role.value}")
                    else:
                        logger.warning(f"OpenRouter configured for agent {agent.agent_id} but connection health check failed")
                else:
                    logger.warning(f"No API client found for agent {agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to configure OpenRouter for agent {agent.agent_id}: {e}")
            # Update status to reflect failed connection
            with self._status_lock:
                if agent.agent_id in self.agent_statuses:
                    self.agent_statuses[agent.agent_id].openrouter_connected = False
    
    def shutdown_agents(self) -> None:
        """
        Gracefully shutdown all agents and cleanup resources with enhanced state persistence
        and proper OpenRouter connection cleanup
        """
        logger.info("Initiating enhanced graceful shutdown of per-core agents")
        shutdown_start_time = time.time()
        
        # Phase 1: Signal shutdown and stop health monitoring
        logger.info("Phase 1: Stopping health monitoring and signaling shutdown")
        self._shutdown = True
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5)
            logger.debug("Health monitoring thread stopped")
        
        # Phase 2: Persist system state for recovery
        logger.info("Phase 2: Persisting system state for recovery on restart")
        try:
            self._persist_system_state_for_recovery()
        except Exception as e:
            logger.error(f"Error persisting system state for recovery: {e}")
        
        # Phase 3: Graceful agent shutdown with enhanced state persistence
        logger.info("Phase 3: Graceful agent shutdown with state persistence")
        with self._agents_lock:
            shutdown_errors = []
            
            for agent in self.agents.values():
                try:
                    logger.debug(f"Shutting down agent {agent.agent_id}")
                    
                    # Create comprehensive final context snapshot
                    snapshot_id = agent.create_context_snapshot()
                    if snapshot_id:
                        logger.debug(f"Created final snapshot {snapshot_id} for agent {agent.agent_id}")
                    
                    # Persist detailed shutdown insight with system state
                    shutdown_insight = {
                        "shutdown_timestamp": time.time(),
                        "agent_role": agent.role.value,
                        "core_id": agent.core_id,
                        "last_activity": agent.last_activity,
                        "status_before_shutdown": agent.status.value,
                        "total_thoughts": len(agent.thoughts),
                        "system_shutdown": True,
                        "recovery_data": {
                            "openrouter_configured": hasattr(self, 'openrouter_api_key') and self.openrouter_api_key is not None,
                            "agent_configuration": {
                                "role": agent.role.value,
                                "core_id": agent.core_id,
                                "agent_id": agent.agent_id
                            }
                        }
                    }
                    
                    agent.persist_insight(
                        content=f"Enhanced graceful shutdown initiated at {time.time()}. "
                               f"Agent {agent.agent_id} ({agent.role.value}) on core {agent.core_id} "
                               f"shutting down with full state persistence for recovery. "
                               f"Recovery data: {shutdown_insight}",
                        importance=0.95,
                        tags={"shutdown", "graceful", "system", "recovery", "enhanced"}
                    )
                    
                    # Update status to offline with timestamp
                    agent.update_status(AgentStatus.OFFLINE)
                    logger.debug(f"Set agent {agent.agent_id} to offline status with enhanced memory persistence")
                    
                except Exception as e:
                    error_msg = f"Error during graceful shutdown of agent {agent.agent_id}: {e}"
                    logger.error(error_msg)
                    shutdown_errors.append(error_msg)
            
            # Phase 4: Cleanup OpenRouter connections and API clients
            logger.info("Phase 4: Cleaning up OpenRouter connections and API clients")
            try:
                self._cleanup_openrouter_connections()
            except Exception as e:
                error_msg = f"Error cleaning up OpenRouter connections: {e}"
                logger.error(error_msg)
                shutdown_errors.append(error_msg)
            
            # Phase 5: Unregister agents from communication system
            logger.info("Phase 5: Unregistering agents from communication system")
            if self.communication_system:
                for agent in self.agents.values():
                    try:
                        self.communication_system.unregister_agent(agent.agent_id)
                        logger.debug(f"Unregistered agent {agent.agent_id}")
                    except Exception as e:
                        error_msg = f"Error unregistering agent {agent.agent_id}: {e}"
                        logger.error(error_msg)
                        shutdown_errors.append(error_msg)
                
                # Shutdown the communication system itself
                try:
                    self.communication_system.shutdown()
                    logger.debug("Communication system shut down")
                except Exception as e:
                    error_msg = f"Error shutting down communication system: {e}"
                    logger.error(error_msg)
                    shutdown_errors.append(error_msg)
            
            # Clear agent collections
            self.agents.clear()
        
        # Phase 6: Clear agent status tracking
        logger.info("Phase 6: Clearing agent status tracking")
        with self._status_lock:
            self.agent_statuses.clear()
        
        # Phase 7: Shutdown persistent memory with enhanced cleanup
        logger.info("Phase 7: Shutting down persistent memory systems")
        try:
            from utils.agent_persistent_memory import shutdown_all_agent_memories
            shutdown_all_agent_memories()
            logger.info("All agent persistent memory instances shut down")
        except Exception as e:
            error_msg = f"Error shutting down agent persistent memories: {e}"
            logger.error(error_msg)
            shutdown_errors.append(error_msg)
        
        # Phase 8: Shutdown monitoring and alerting
        logger.info("Phase 8: Shutting down monitoring and alerting systems")
        try:
            self._shutdown_monitoring_and_alerting()
        except Exception as e:
            error_msg = f"Error shutting down monitoring and alerting: {e}"
            logger.error(error_msg)
            shutdown_errors.append(error_msg)
        
        # Phase 9: Final cleanup and state persistence
        logger.info("Phase 9: Final cleanup and shutdown state persistence")
        try:
            self._persist_final_shutdown_state(shutdown_start_time, shutdown_errors)
        except Exception as e:
            logger.error(f"Error persisting final shutdown state: {e}")
        
        # Calculate and log shutdown duration
        shutdown_duration = time.time() - shutdown_start_time
        
        if shutdown_errors:
            logger.warning(f"Per-core agent shutdown completed with {len(shutdown_errors)} errors in {shutdown_duration:.2f}s")
            for error in shutdown_errors:
                logger.warning(f"Shutdown error: {error}")
        else:
            logger.info(f"Per-core agent shutdown completed successfully in {shutdown_duration:.2f}s with enhanced state persistence")
    
    def _persist_system_state_for_recovery(self) -> None:
        """
        Persist system state for recovery on restart
        """
        try:
            import json
            import os
            
            # Create recovery state directory
            recovery_dir = os.path.join(os.path.expanduser("~"), ".zen_mcp", "recovery")
            os.makedirs(recovery_dir, exist_ok=True)
            
            # Collect system state
            system_state = {
                "shutdown_timestamp": time.time(),
                "detected_cores": self.detected_cores,
                "effective_max_agents": self.effective_max_agents,
                "openrouter_configured": bool(self.openrouter_api_key),
                "agents": {},
                "agent_statuses": {},
                "system_config": self.system_config.to_dict() if self.system_config else {},
                "emergency_mode": self._emergency_mode,
                "emergency_mode_reason": self._emergency_mode_reason,
                "redistribution_history": self._redistribution_history[-10:]  # Keep last 10 entries
            }
            
            # Collect agent states
            with self._agents_lock:
                for core_id, agent in self.agents.items():
                    system_state["agents"][str(core_id)] = {
                        "agent_id": agent.agent_id,
                        "role": agent.role.value,
                        "core_id": agent.core_id,
                        "status": agent.status.value,
                        "last_activity": agent.last_activity,
                        "total_thoughts": len(agent.thoughts),
                        "created_at": getattr(agent, 'created_at', time.time())
                    }
            
            # Collect agent status information
            with self._status_lock:
                for agent_id, status in self.agent_statuses.items():
                    system_state["agent_statuses"][agent_id] = status.to_dict()
            
            # Save system state to file
            recovery_file = os.path.join(recovery_dir, "per_core_agent_system_state.json")
            with open(recovery_file, 'w') as f:
                json.dump(system_state, f, indent=2)
            
            logger.info(f"System state persisted for recovery: {recovery_file}")
            
            # Also create a timestamp file for recovery detection
            timestamp_file = os.path.join(recovery_dir, "last_shutdown_timestamp.txt")
            with open(timestamp_file, 'w') as f:
                f.write(str(time.time()))
            
        except Exception as e:
            logger.error(f"Failed to persist system state for recovery: {e}")
            raise
    
    def _cleanup_openrouter_connections(self) -> None:
        """
        Cleanup OpenRouter connections and API client resources
        """
        logger.debug("Starting OpenRouter connection cleanup")
        
        cleanup_count = 0
        cleanup_errors = []
        
        # Cleanup API clients for each agent
        if self.communication_system:
            for agent in self.agents.values():
                try:
                    api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                    if api_client:
                        # Cancel any active API calls
                        active_calls = getattr(api_client, 'active_calls', {})
                        if active_calls:
                            logger.debug(f"Cancelling {len(active_calls)} active API calls for agent {agent.agent_id}")
                            for call_id, api_call in list(active_calls.items()):
                                try:
                                    api_call.status = "cancelled"
                                    api_call.error = "System shutdown"
                                    del active_calls[call_id]
                                except Exception as e:
                                    logger.warning(f"Error cancelling API call {call_id}: {e}")
                        
                        # Reset OpenRouter-specific state
                        if hasattr(api_client, 'openrouter_config') and api_client.openrouter_config:
                            logger.debug(f"Cleaning up OpenRouter configuration for agent {agent.agent_id}")
                            
                            # Clear rate limiting state
                            if hasattr(api_client, 'openrouter_call_times'):
                                api_client.openrouter_call_times.clear()
                            
                            # Reset circuit breaker state
                            if hasattr(api_client, 'openrouter_circuit_open'):
                                api_client.openrouter_circuit_open = False
                                api_client.openrouter_failure_count = 0
                            
                            # Clear OpenRouter configuration (but don't delete the object)
                            # This prevents any pending operations from using stale config
                            api_client.openrouter_only = False
                            
                            cleanup_count += 1
                        
                        # Use the new cleanup_connections method
                        if hasattr(api_client, 'cleanup_connections'):
                            api_client.cleanup_connections()
                        elif hasattr(api_client, 'reset_statistics'):
                            # Fallback to reset_statistics if cleanup_connections not available
                            api_client.reset_statistics()
                        
                except Exception as e:
                    error_msg = f"Error cleaning up API client for agent {agent.agent_id}: {e}"
                    logger.warning(error_msg)
                    cleanup_errors.append(error_msg)
        
        # Clear any global OpenRouter state if it exists
        try:
            # Reset any provider-level OpenRouter state
            from providers.registry import ModelProviderRegistry
            from providers.base import ProviderType
            
            registry = ModelProviderRegistry()
            openrouter_provider = registry.get_provider(ProviderType.OPENROUTER)
            
            if openrouter_provider and hasattr(openrouter_provider, 'reset_connection_state'):
                openrouter_provider.reset_connection_state()
                logger.debug("Reset OpenRouter provider connection state")
                
        except Exception as e:
            error_msg = f"Error resetting OpenRouter provider state: {e}"
            logger.warning(error_msg)
            cleanup_errors.append(error_msg)
        
        if cleanup_errors:
            logger.warning(f"OpenRouter cleanup completed with {len(cleanup_errors)} errors. "
                          f"Successfully cleaned up {cleanup_count} agent connections.")
            for error in cleanup_errors:
                logger.debug(f"OpenRouter cleanup error: {error}")
        else:
            logger.info(f"OpenRouter connection cleanup completed successfully. "
                       f"Cleaned up {cleanup_count} agent connections.")
    
    def _persist_final_shutdown_state(self, shutdown_start_time: float, shutdown_errors: List[str]) -> None:
        """
        Persist final shutdown state and statistics
        
        Args:
            shutdown_start_time: When the shutdown process started
            shutdown_errors: List of errors encountered during shutdown
        """
        try:
            import json
            import os
            
            # Create shutdown logs directory
            shutdown_logs_dir = os.path.join(os.path.expanduser("~"), ".zen_mcp", "shutdown_logs")
            os.makedirs(shutdown_logs_dir, exist_ok=True)
            
            shutdown_duration = time.time() - shutdown_start_time
            
            # Create detailed shutdown report
            shutdown_report = {
                "shutdown_start_time": shutdown_start_time,
                "shutdown_end_time": time.time(),
                "shutdown_duration_seconds": shutdown_duration,
                "shutdown_successful": len(shutdown_errors) == 0,
                "error_count": len(shutdown_errors),
                "errors": shutdown_errors,
                "system_info": {
                    "detected_cores": self.detected_cores,
                    "effective_max_agents": self.effective_max_agents,
                    "openrouter_configured": bool(self.openrouter_api_key),
                    "emergency_mode_active": self._emergency_mode,
                    "config_manager_active": self.config_manager is not None
                },
                "shutdown_phases": [
                    "Stop health monitoring",
                    "Persist system state for recovery",
                    "Graceful agent shutdown with state persistence",
                    "Cleanup OpenRouter connections and API clients",
                    "Unregister agents from communication system",
                    "Clear agent status tracking",
                    "Shutdown persistent memory systems",
                    "Shutdown monitoring and alerting systems",
                    "Final cleanup and shutdown state persistence"
                ],
                "recovery_info": {
                    "can_recover": True,
                    "recovery_state_persisted": True,
                    "next_startup_recommendations": [
                        "Check recovery state files in ~/.zen_mcp/recovery/",
                        "Validate OpenRouter API key if configured",
                        "Review shutdown errors if any occurred",
                        "Monitor agent initialization on restart"
                    ]
                }
            }
            
            # Save shutdown report with timestamp
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(shutdown_start_time))
            shutdown_report_file = os.path.join(shutdown_logs_dir, f"shutdown_report_{timestamp_str}.json")
            
            with open(shutdown_report_file, 'w') as f:
                json.dump(shutdown_report, f, indent=2)
            
            # Also update the latest shutdown report
            latest_report_file = os.path.join(shutdown_logs_dir, "latest_shutdown_report.json")
            with open(latest_report_file, 'w') as f:
                json.dump(shutdown_report, f, indent=2)
            
            logger.info(f"Final shutdown state persisted: {shutdown_report_file}")
            
            # Clean up old shutdown reports (keep last 10)
            try:
                import glob
                report_files = glob.glob(os.path.join(shutdown_logs_dir, "shutdown_report_*.json"))
                if len(report_files) > 10:
                    # Sort by modification time and remove oldest
                    report_files.sort(key=lambda x: os.path.getmtime(x))
                    for old_file in report_files[:-10]:
                        os.remove(old_file)
                        logger.debug(f"Removed old shutdown report: {old_file}")
            except Exception as e:
                logger.debug(f"Error cleaning up old shutdown reports: {e}")
            
        except Exception as e:
            logger.error(f"Failed to persist final shutdown state: {e}")
            raise
    
    def _check_recovery_state(self) -> None:
        """
        Check for recovery state from previous shutdown and log recovery information
        """
        try:
            import json
            import os
            
            recovery_dir = os.path.join(os.path.expanduser("~"), ".zen_mcp", "recovery")
            recovery_file = os.path.join(recovery_dir, "per_core_agent_system_state.json")
            timestamp_file = os.path.join(recovery_dir, "last_shutdown_timestamp.txt")
            
            if os.path.exists(recovery_file) and os.path.exists(timestamp_file):
                # Read recovery state
                with open(recovery_file, 'r') as f:
                    recovery_state = json.load(f)
                
                with open(timestamp_file, 'r') as f:
                    last_shutdown_timestamp = float(f.read().strip())
                
                # Calculate time since last shutdown
                time_since_shutdown = time.time() - last_shutdown_timestamp
                
                logger.info(f"Recovery state detected from previous shutdown {time_since_shutdown:.1f} seconds ago")
                logger.info(f"Previous system had {recovery_state.get('effective_max_agents', 'unknown')} agents "
                           f"across {recovery_state.get('detected_cores', 'unknown')} cores")
                
                # Check if OpenRouter was configured
                if recovery_state.get('openrouter_configured', False):
                    logger.info("Previous system had OpenRouter configured")
                    if not self.openrouter_api_key:
                        logger.warning("Previous system had OpenRouter configured but current system does not - "
                                     "agents may not initialize with the same capabilities")
                
                # Check for emergency mode
                if recovery_state.get('emergency_mode', False):
                    logger.warning(f"Previous system was in emergency mode: {recovery_state.get('emergency_mode_reason', 'unknown reason')}")
                
                # Log agent recovery information
                previous_agents = recovery_state.get('agents', {})
                if previous_agents:
                    logger.info(f"Previous system had {len(previous_agents)} agents:")
                    for core_id, agent_info in previous_agents.items():
                        logger.debug(f"  Core {core_id}: {agent_info.get('role', 'unknown')} "
                                   f"(last active: {time.time() - agent_info.get('last_activity', 0):.1f}s ago)")
                
                # Check shutdown logs for any errors
                shutdown_logs_dir = os.path.join(os.path.expanduser("~"), ".zen_mcp", "shutdown_logs")
                latest_report_file = os.path.join(shutdown_logs_dir, "latest_shutdown_report.json")
                
                if os.path.exists(latest_report_file):
                    try:
                        with open(latest_report_file, 'r') as f:
                            shutdown_report = json.load(f)
                        
                        if not shutdown_report.get('shutdown_successful', True):
                            error_count = shutdown_report.get('error_count', 0)
                            logger.warning(f"Previous shutdown had {error_count} errors - check shutdown logs for details")
                        else:
                            shutdown_duration = shutdown_report.get('shutdown_duration_seconds', 0)
                            logger.info(f"Previous shutdown completed successfully in {shutdown_duration:.2f}s")
                    except Exception as e:
                        logger.debug(f"Could not read shutdown report: {e}")
                
                # Clean up recovery state files after successful read
                try:
                    os.remove(recovery_file)
                    os.remove(timestamp_file)
                    logger.debug("Recovery state files cleaned up")
                except Exception as e:
                    logger.debug(f"Could not clean up recovery state files: {e}")
                
            else:
                logger.debug("No recovery state found - this appears to be a fresh startup")
                
        except Exception as e:
            logger.debug(f"Error checking recovery state: {e}")
            # Don't fail initialization due to recovery state issues
    
    def get_recovery_recommendations(self) -> List[str]:
        """
        Get recommendations for system recovery based on previous shutdown state
        
        Returns:
            List of recovery recommendations
        """
        recommendations = []
        
        try:
            import json
            import os
            
            # Check shutdown logs for recommendations
            shutdown_logs_dir = os.path.join(os.path.expanduser("~"), ".zen_mcp", "shutdown_logs")
            latest_report_file = os.path.join(shutdown_logs_dir, "latest_shutdown_report.json")
            
            if os.path.exists(latest_report_file):
                with open(latest_report_file, 'r') as f:
                    shutdown_report = json.load(f)
                
                # Add recommendations from shutdown report
                recovery_info = shutdown_report.get('recovery_info', {})
                if 'next_startup_recommendations' in recovery_info:
                    recommendations.extend(recovery_info['next_startup_recommendations'])
                
                # Add specific recommendations based on shutdown errors
                if shutdown_report.get('error_count', 0) > 0:
                    recommendations.append("Review shutdown errors in the shutdown report")
                    recommendations.append("Consider checking system resources and OpenRouter connectivity")
                
                # Add recommendations based on system info
                system_info = shutdown_report.get('system_info', {})
                if system_info.get('emergency_mode_active', False):
                    recommendations.append("Previous system was in emergency mode - investigate underlying issues")
                
                if not system_info.get('openrouter_configured', False) and self.openrouter_api_key:
                    recommendations.append("OpenRouter is now configured but wasn't in previous session - agents will have enhanced capabilities")
                
        except Exception as e:
            logger.debug(f"Error getting recovery recommendations: {e}")
            recommendations.append("Check system logs for any startup issues")
        
        return recommendations
    
    def _on_config_changed(self, old_config: Optional['PerCoreAgentSystemConfig'], 
                          new_config: 'PerCoreAgentSystemConfig') -> None:
        """
        Handle configuration changes with hot-reload capabilities
        
        Args:
            old_config: Previous configuration (may be None)
            new_config: New configuration
        """
        logger.info("Per-core agent configuration changed, applying hot-reload...")
        
        try:
            # Update system configuration
            self.system_config = new_config
            self.openrouter_api_key = new_config.openrouter_api_key
            self.max_agents = new_config.max_agents
            
            # Check if system should be enabled/disabled
            if old_config and old_config.enabled != new_config.enabled:
                if new_config.enabled and not self.agents:
                    logger.info("Per-core agents enabled in configuration, initializing agents...")
                    self.initialize_agents()
                elif not new_config.enabled and self.agents:
                    logger.info("Per-core agents disabled in configuration, shutting down agents...")
                    self.shutdown_agents()
            
            # Update health check interval if changed
            if old_config and old_config.health_check_interval != new_config.health_check_interval:
                logger.info(f"Health check interval changed from {old_config.health_check_interval}s to {new_config.health_check_interval}s")
                # Health monitoring thread will pick up the new interval on next cycle
            
            # Update agent timeouts if changed
            if old_config and old_config.agent_timeout != new_config.agent_timeout:
                logger.info(f"Agent timeout changed from {old_config.agent_timeout}s to {new_config.agent_timeout}s")
                # Update timeout for existing agents
                with self._agents_lock:
                    for agent in self.agents.values():
                        if hasattr(agent, 'timeout'):
                            agent.timeout = new_config.agent_timeout
            
            # Update OpenRouter configurations for existing agents if API key changed
            if (old_config and old_config.openrouter_api_key != new_config.openrouter_api_key and 
                new_config.openrouter_api_key):
                logger.info("OpenRouter API key changed, updating agent configurations...")
                with self._agents_lock:
                    for agent in self.agents.values():
                        try:
                            self._configure_agent_openrouter(agent)
                        except Exception as e:
                            logger.error(f"Failed to update OpenRouter config for agent {agent.agent_id}: {e}")
            
            # Update role-specific configurations if they changed
            if old_config and old_config.role_configs != new_config.role_configs:
                logger.info("Role configurations changed, updating agent OpenRouter settings...")
                with self._agents_lock:
                    for agent in self.agents.values():
                        try:
                            # Get updated role configuration
                            role_config = self.config_manager.get_openrouter_config_for_role(agent.role)
                            if role_config:
                                # Update agent's API client configuration
                                if self.communication_system:
                                    api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                                    if api_client and hasattr(api_client, 'update_openrouter_config'):
                                        api_client.update_openrouter_config(role_config)
                                        logger.debug(f"Updated OpenRouter config for agent {agent.agent_id}")
                        except Exception as e:
                            logger.error(f"Failed to update role config for agent {agent.agent_id}: {e}")
            
            # Validate new configuration
            config_errors = new_config.validate()
            if config_errors:
                logger.warning(f"New configuration has validation errors: {'; '.join(config_errors)}")
                # Send alert about configuration issues
                if self._alerting_system:
                    self._alerting_system.send_alert(
                        "configuration_validation_failed",
                        f"Per-core agent configuration validation failed: {'; '.join(config_errors)}",
                        severity="warning"
                    )
            else:
                logger.info("Configuration hot-reload completed successfully")
                
        except Exception as e:
            logger.error(f"Error during configuration hot-reload: {e}")
            # Send alert about hot-reload failure
            if self._alerting_system:
                self._alerting_system.send_alert(
                    "config_hot_reload_failed",
                    f"Per-core agent configuration hot-reload failed: {e}",
                    severity="error"
                )
    
    def _validate_openrouter_api_key(self, api_key: str) -> bool:
        """
        Validate OpenRouter API key format and basic structure
        
        Args:
            api_key: The OpenRouter API key to validate
            
        Returns:
            True if the API key appears valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            logger.error("OpenRouter API key is empty or not a string")
            return False
        
        # Basic format validation for OpenRouter API keys
        # OpenRouter keys typically start with 'sk-or-' and are followed by base64-like characters
        if not api_key.startswith('sk-or-'):
            logger.warning("OpenRouter API key does not start with expected prefix 'sk-or-'")
            # Don't fail validation as format might change, just warn
        
        # Check minimum length (OpenRouter keys are typically 50+ characters)
        if len(api_key) < 20:
            logger.error(f"OpenRouter API key too short: {len(api_key)} characters")
            return False
        
        # Check for placeholder values
        placeholder_values = [
            'your_api_key_here',
            'your_openrouter_key',
            'sk-or-placeholder',
            'test_key',
            'dummy_key'
        ]
        
        if api_key.lower() in [p.lower() for p in placeholder_values]:
            logger.error(f"OpenRouter API key appears to be a placeholder: {api_key}")
            return False
        
        logger.debug("OpenRouter API key passed basic validation")
        return True
    
    def _create_openrouter_config_for_role(self, role: AgentRole) -> 'OpenRouterConfig':
        """
        Create role-specific OpenRouter configuration using the configuration manager
        
        Args:
            role: The agent role to create configuration for
            
        Returns:
            OpenRouterConfig instance with role-specific settings
        """
        from utils.agent_api_client import OpenRouterConfig
        
        # Get role-specific configuration from config manager
        role_config = self.config_manager.get_openrouter_config_for_role(role)
        
        if role_config:
            # Convert from PerCoreAgentConfig to OpenRouterConfig
            config = OpenRouterConfig(
                api_key=role_config.api_key,
                preferred_models=role_config.preferred_models,
                rate_limit_per_minute=role_config.rate_limit_per_minute,
                max_concurrent_calls=role_config.max_concurrent_calls,
                thinking_mode_default=role_config.thinking_mode_default,
                temperature_range=role_config.temperature_range,
                fallback_enabled=role_config.fallback_enabled,
                circuit_breaker_threshold=role_config.circuit_breaker_threshold,
                circuit_breaker_timeout=role_config.circuit_breaker_timeout
            )
            
            logger.debug(f"Created OpenRouter config for {role.value} from config manager: "
                        f"{len(config.preferred_models)} models, rate_limit={config.rate_limit_per_minute}")
            
            return config
        
        else:
            # Fallback to default configuration if no role-specific config available
            logger.warning(f"No OpenRouter configuration found for role {role.value}, using fallback")
            
            config = OpenRouterConfig(
                api_key=self.openrouter_api_key or "",
                preferred_models=["google/gemini-2.5-flash", "openai/o3-mini"],
                rate_limit_per_minute=60,
                max_concurrent_calls=3,
                thinking_mode_default="standard",
                temperature_range=(0.4, 0.7),
                fallback_enabled=True,
                circuit_breaker_threshold=5,
                circuit_breaker_timeout=300.0
            )
            
            return config
    
    def _check_openrouter_connection_health(self, agent_id: str, api_client: 'AgentAPIClient') -> bool:
        """
        Perform a health check on the OpenRouter connection for an agent
        
        Args:
            agent_id: The agent ID to check
            api_client: The agent's API client
            
        Returns:
            True if the connection is healthy, False otherwise
        """
        try:
            # Check if OpenRouter configuration is present
            if not api_client.openrouter_config:
                logger.warning(f"No OpenRouter config found for agent {agent_id}")
                return False
            
            # Check if API key is configured
            if not api_client.openrouter_config.api_key:
                logger.warning(f"No OpenRouter API key configured for agent {agent_id}")
                return False
            
            # Check circuit breaker status
            if hasattr(api_client, '_is_openrouter_circuit_open') and api_client._is_openrouter_circuit_open():
                logger.warning(f"OpenRouter circuit breaker is open for agent {agent_id}")
                return False
            
            # Verify preferred models are configured
            if not api_client.openrouter_config.preferred_models:
                logger.warning(f"No preferred models configured for agent {agent_id}")
                return False
            
            # Check if OpenRouter provider is available in registry
            try:
                from providers.registry import ModelProviderRegistry
                from providers.base import ProviderType
                
                registry = ModelProviderRegistry()
                openrouter_provider = registry.get_provider(ProviderType.OPENROUTER)
                
                if not openrouter_provider:
                    logger.warning(f"OpenRouter provider not available in registry for agent {agent_id}")
                    return False
                
                # Check if at least one preferred model is available
                available_models = registry.get_available_model_names(ProviderType.OPENROUTER)
                preferred_available = any(
                    model in available_models 
                    for model in api_client.openrouter_config.preferred_models
                )
                
                if not preferred_available:
                    logger.warning(f"None of the preferred models are available for agent {agent_id}")
                    return False
                
            except Exception as e:
                logger.warning(f"Could not verify OpenRouter provider availability for agent {agent_id}: {e}")
                # Don't fail health check for provider registry issues
            
            logger.debug(f"OpenRouter connection health check passed for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"OpenRouter connection health check failed for agent {agent_id}: {e}")
            return False
    
    def get_agent_by_core(self, core_id: int) -> Optional[Agent]:
        """
        Get the agent assigned to a specific CPU core
        
        Args:
            core_id: The CPU core ID
            
        Returns:
            Agent instance or None if not found
        """
        with self._agents_lock:
            return self.agents.get(core_id)
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """
        Get all agents with a specific role
        
        Args:
            role: The agent role to search for
            
        Returns:
            List of agents with the specified role
        """
        with self._agents_lock:
            return [agent for agent in self.agents.values() if agent.role == role]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by its ID
        
        Args:
            agent_id: The agent ID to search for
            
        Returns:
            Agent instance if found, None otherwise
        """
        with self._agents_lock:
            for agent in self.agents.values():
                if agent.agent_id == agent_id:
                    return agent
            return None
    
    async def restart_agent(self, agent_id: str) -> bool:
        """
        Public method to restart an agent (used by error recovery system)
        
        Args:
            agent_id: ID of the agent to restart
            
        Returns:
            True if restart was successful, False otherwise
        """
        from utils.per_core_logging import log_agent_activity, log_system_event
        
        # Find the agent to get its core_id and role
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            logger.error(f"Cannot restart agent {agent_id}: agent not found")
            return False
        
        core_id = agent.core_id
        role = agent.role
        
        # Log restart attempt
        log_agent_activity(
            agent_id=agent_id,
            activity_type="agent_restart_attempt",
            activity_details={
                "core_id": core_id,
                "role": role.value,
                "reason": "error_recovery"
            }
        )
        
        log_system_event(
            event_type="agent_restart_initiated",
            event_details={
                "agent_id": agent_id,
                "core_id": core_id,
                "role": role.value,
                "initiated_by": "error_recovery_system"
            },
            severity="INFO",
            affected_agents=[agent_id]
        )
        
        # Attempt restart
        success = self._restart_agent(agent_id, core_id, role)
        
        if success:
            # Log successful restart
            log_agent_activity(
                agent_id=f"core_{core_id}_{role.value}",  # New agent ID
                activity_type="agent_restart_success",
                activity_details={
                    "original_agent_id": agent_id,
                    "core_id": core_id,
                    "role": role.value,
                    "recovery_method": "full_restart"
                },
                success=True
            )
            
            log_system_event(
                event_type="agent_restart_completed",
                event_details={
                    "original_agent_id": agent_id,
                    "new_agent_id": f"core_{core_id}_{role.value}",
                    "core_id": core_id,
                    "role": role.value,
                    "success": True
                },
                severity="INFO",
                affected_agents=[f"core_{core_id}_{role.value}"]
            )
        else:
            # Log failed restart
            log_agent_activity(
                agent_id=agent_id,
                activity_type="agent_restart_failure",
                activity_details={
                    "core_id": core_id,
                    "role": role.value,
                    "recovery_method": "full_restart"
                },
                success=False,
                error_message="Agent restart failed"
            )
            
            log_system_event(
                event_type="agent_restart_failed",
                event_details={
                    "agent_id": agent_id,
                    "core_id": core_id,
                    "role": role.value,
                    "success": False
                },
                severity="ERROR",
                affected_agents=[agent_id]
            )
        
        return success
    
    def redistribute_workload(self, failed_agent_id: str) -> Dict[str, Any]:
        """
        Redistribute workload when an agent fails or becomes unresponsive
        
        Args:
            failed_agent_id: ID of the failed agent
            
        Returns:
            Dictionary with redistribution results and strategy applied
        """
        logger.warning(f"Redistributing workload from failed agent {failed_agent_id}")
        
        redistribution_result = {
            "failed_agent_id": failed_agent_id,
            "strategy_applied": None,
            "healthy_agents_available": 0,
            "workload_redistributed": False,
            "fallback_mode_activated": False,
            "notifications_sent": 0,
            "error": None
        }
        
        try:
            with self._agents_lock:
                # Find the failed agent
                failed_agent = None
                failed_core_id = None
                
                for core_id, agent in self.agents.items():
                    if agent.agent_id == failed_agent_id:
                        failed_agent = agent
                        failed_core_id = core_id
                        break
                
                if not failed_agent:
                    error_msg = f"Failed agent {failed_agent_id} not found in agent registry"
                    logger.error(error_msg)
                    redistribution_result["error"] = error_msg
                    return redistribution_result
                
                # Mark agent as offline
                failed_agent.update_status(AgentStatus.OFFLINE)
                
                # Update status tracking
                with self._status_lock:
                    if failed_agent_id in self.agent_statuses:
                        self.agent_statuses[failed_agent_id].status = AgentStatus.OFFLINE
                        self.agent_statuses[failed_agent_id].openrouter_connected = False
                
                # Get healthy agents for workload redistribution
                healthy_agents = [
                    agent for agent in self.agents.values()
                    if agent.agent_id != failed_agent_id and agent.status == AgentStatus.ACTIVE
                ]
                
                # Get agents with same role as failed agent for specialized workload
                same_role_agents = [
                    agent for agent in healthy_agents
                    if agent.role == failed_agent.role
                ]
                
                redistribution_result["healthy_agents_available"] = len(healthy_agents)
                
                # Apply redistribution strategy based on available agents
                if same_role_agents:
                    # Strategy 1: Redistribute to agents with same role
                    redistribution_result["strategy_applied"] = "same_role_redistribution"
                    target_agents = same_role_agents
                    logger.info(f"Redistributing {failed_agent.role.value} workload from {failed_agent_id} "
                              f"to {len(same_role_agents)} agents with same role")
                    
                elif healthy_agents:
                    # Strategy 2: Redistribute to any healthy agents
                    redistribution_result["strategy_applied"] = "general_redistribution"
                    target_agents = healthy_agents
                    logger.info(f"Redistributing workload from {failed_agent_id} "
                              f"to {len(healthy_agents)} healthy agents (mixed roles)")
                    
                else:
                    # Strategy 3: No healthy agents - activate emergency fallback
                    redistribution_result["strategy_applied"] = "emergency_fallback"
                    redistribution_result["fallback_mode_activated"] = True
                    logger.critical("No healthy agents available! Activating emergency fallback mode")
                    self._activate_emergency_fallback_mode(failed_agent)
                    return redistribution_result
                
                # Execute workload redistribution
                redistribution_result["workload_redistributed"] = True
                
                # Notify target agents about workload redistribution
                if self.communication_system:
                    for target_agent in target_agents:
                        try:
                            # Send detailed redistribution notification
                            message_content = {
                                "type": "workload_redistribution",
                                "failed_agent_id": failed_agent_id,
                                "failed_agent_role": failed_agent.role.value,
                                "redistribution_strategy": redistribution_result["strategy_applied"],
                                "your_additional_load": self._calculate_additional_load(
                                    failed_agent, target_agent, len(target_agents)
                                ),
                                "priority_tasks": self._get_priority_tasks_for_role(failed_agent.role)
                            }
                            
                            self.communication_system.send_message(
                                from_agent="system",
                                to_agent=target_agent.agent_id,
                                message_type="workload_redistribution",
                                content=message_content,
                                priority=9  # High priority for redistribution
                            )
                            redistribution_result["notifications_sent"] += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to notify agent {target_agent.agent_id} about redistribution: {e}")
                
                # Update system-wide load balancing metrics
                self._update_load_balancing_metrics(failed_agent_id, target_agents)
                
                logger.info(f"Successfully redistributed workload from {failed_agent_id} using strategy: "
                          f"{redistribution_result['strategy_applied']}")
                
        except Exception as e:
            error_msg = f"Error during workload redistribution for {failed_agent_id}: {e}"
            logger.error(error_msg)
            redistribution_result["error"] = error_msg
        
        return redistribution_result
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive system health information
        
        Returns:
            Dictionary containing system health metrics
        """
        with self._agents_lock, self._status_lock:
            total_agents = len(self.agents)
            healthy_agents = sum(1 for status in self.agent_statuses.values() if status.is_healthy())
            agents_needing_attention = sum(1 for status in self.agent_statuses.values() if status.needs_attention())
            
            # Calculate overall system health score (0.0 to 1.0)
            if total_agents > 0:
                health_score = healthy_agents / total_agents
            else:
                health_score = 0.0
            
            # Aggregate statistics
            total_api_calls = sum(status.total_api_calls for status in self.agent_statuses.values())
            avg_success_rate = (
                sum(status.success_rate for status in self.agent_statuses.values()) / total_agents
                if total_agents > 0 else 0.0
            )
            
            return {
                "detected_cores": self.detected_cores,
                "max_agents": self.effective_max_agents,
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "agents_needing_attention": agents_needing_attention,
                "health_score": health_score,
                "total_api_calls": total_api_calls,
                "avg_success_rate": avg_success_rate,
                "openrouter_configured": self.openrouter_api_key is not None,
                "agent_statuses": {
                    agent_id: status.to_dict() 
                    for agent_id, status in self.agent_statuses.items()
                },
                "role_distribution": self._get_role_distribution()
            }
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of agent roles"""
        distribution = {}
        with self._agents_lock:
            for agent in self.agents.values():
                role_name = agent.role.value
                distribution[role_name] = distribution.get(role_name, 0) + 1
        return distribution
    
    def _start_health_monitoring(self) -> None:
        """Start background health monitoring thread"""
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            return
        
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name="PerCoreAgentHealthMonitor"
        )
        self._health_monitor_thread.start()
        logger.info("Started per-core agent health monitoring")
    
    def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while not self._shutdown:
            try:
                self._update_agent_health_status()
                time.sleep(30)  # Check health every 30 seconds
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(60)  # Longer pause on error
    
    def get_openrouter_system_health(self) -> Dict[str, Any]:
        """
        Get comprehensive OpenRouter system health information
        
        Returns:
            Dictionary containing OpenRouter-specific health metrics
        """
        with self._agents_lock, self._status_lock:
            openrouter_stats = {
                "api_key_configured": self.openrouter_api_key is not None,
                "api_key_valid": False,
                "total_agents_with_openrouter": 0,
                "healthy_openrouter_connections": 0,
                "failed_openrouter_connections": 0,
                "agents_with_circuit_breaker_open": 0,
                "total_openrouter_api_calls": 0,
                "total_thinking_sessions": 0,
                "avg_openrouter_success_rate": 0.0,
                "agent_openrouter_details": {}
            }
            
            # Validate API key if configured
            if self.openrouter_api_key:
                openrouter_stats["api_key_valid"] = self._validate_openrouter_api_key(self.openrouter_api_key)
            
            # Collect per-agent OpenRouter statistics
            total_success_rates = []
            
            for agent_id, agent in self.agents.items():
                if self.communication_system:
                    api_client = self.communication_system.get_agent_api_client(agent_id)
                    if api_client and api_client.openrouter_only:
                        openrouter_stats["total_agents_with_openrouter"] += 1
                        
                        # Get agent's OpenRouter usage stats
                        agent_stats = api_client.get_openrouter_usage_stats()
                        openrouter_stats["agent_openrouter_details"][agent_id] = agent_stats
                        
                        # Aggregate statistics
                        openrouter_stats["total_openrouter_api_calls"] += agent_stats.get("total_openrouter_calls", 0)
                        openrouter_stats["total_thinking_sessions"] += agent_stats.get("total_thinking_sessions", 0)
                        
                        if agent_stats.get("openrouter_success_rate", 0) > 0:
                            total_success_rates.append(agent_stats["openrouter_success_rate"])
                        
                        # Check connection health
                        if agent_id in self.agent_statuses:
                            if self.agent_statuses[agent_id].openrouter_connected:
                                openrouter_stats["healthy_openrouter_connections"] += 1
                            else:
                                openrouter_stats["failed_openrouter_connections"] += 1
                        
                        # Check circuit breaker status
                        if agent_stats.get("circuit_breaker_open", False):
                            openrouter_stats["agents_with_circuit_breaker_open"] += 1
            
            # Calculate average success rate
            if total_success_rates:
                openrouter_stats["avg_openrouter_success_rate"] = sum(total_success_rates) / len(total_success_rates)
            
            # Add overall health assessment
            total_openrouter_agents = openrouter_stats["total_agents_with_openrouter"]
            if total_openrouter_agents > 0:
                health_ratio = openrouter_stats["healthy_openrouter_connections"] / total_openrouter_agents
                openrouter_stats["overall_health_score"] = health_ratio
                
                if health_ratio >= 0.8:
                    openrouter_stats["health_status"] = "healthy"
                elif health_ratio >= 0.5:
                    openrouter_stats["health_status"] = "degraded"
                else:
                    openrouter_stats["health_status"] = "unhealthy"
            else:
                openrouter_stats["overall_health_score"] = 0.0
                openrouter_stats["health_status"] = "not_configured"
            
            return openrouter_stats
    
    def validate_and_test_openrouter_connection(self) -> Dict[str, Any]:
        """
        Validate OpenRouter API key and test connection across all agents
        
        Returns:
            Dictionary with validation and test results
        """
        results = {
            "api_key_validation": {
                "valid": False,
                "error": None
            },
            "agent_connection_tests": {},
            "overall_success": False,
            "summary": {
                "total_agents_tested": 0,
                "successful_connections": 0,
                "failed_connections": 0
            }
        }
        
        # Validate API key
        if not self.openrouter_api_key:
            results["api_key_validation"]["error"] = "No OpenRouter API key configured"
            return results
        
        try:
            results["api_key_validation"]["valid"] = self._validate_openrouter_api_key(self.openrouter_api_key)
            if not results["api_key_validation"]["valid"]:
                results["api_key_validation"]["error"] = "API key failed validation"
                return results
        except Exception as e:
            results["api_key_validation"]["error"] = f"API key validation error: {e}"
            return results
        
        # Test connections for each agent
        with self._agents_lock:
            for agent_id, agent in self.agents.items():
                if self.communication_system:
                    api_client = self.communication_system.get_agent_api_client(agent_id)
                    if api_client:
                        results["summary"]["total_agents_tested"] += 1
                        
                        try:
                            # Test connection health
                            connection_healthy = self._check_openrouter_connection_health(agent_id, api_client)
                            
                            results["agent_connection_tests"][agent_id] = {
                                "agent_role": agent.role.value,
                                "connection_healthy": connection_healthy,
                                "openrouter_configured": api_client.openrouter_only,
                                "preferred_models": (
                                    api_client.openrouter_config.preferred_models 
                                    if api_client.openrouter_config else []
                                ),
                                "error": None
                            }
                            
                            if connection_healthy:
                                results["summary"]["successful_connections"] += 1
                            else:
                                results["summary"]["failed_connections"] += 1
                                
                        except Exception as e:
                            results["agent_connection_tests"][agent_id] = {
                                "agent_role": agent.role.value,
                                "connection_healthy": False,
                                "openrouter_configured": False,
                                "preferred_models": [],
                                "error": str(e)
                            }
                            results["summary"]["failed_connections"] += 1
        
        # Determine overall success
        results["overall_success"] = (
            results["api_key_validation"]["valid"] and
            results["summary"]["successful_connections"] > 0 and
            results["summary"]["failed_connections"] == 0
        )
        
        return results
    
    def _update_agent_health_status(self) -> None:
        """Update health status for all agents and trigger recovery if needed"""
        with self._agents_lock, self._status_lock:
            for agent in self.agents.values():
                if agent.agent_id not in self.agent_statuses:
                    continue
                
                status = self.agent_statuses[agent.agent_id]
                
                # Update basic status
                status.status = agent.status
                status.last_activity = agent.last_activity
                
                # Update API call statistics if communication system available
                if self.communication_system:
                    api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                    if api_client:
                        call_stats = api_client.get_call_statistics()
                        status.total_api_calls = call_stats.get("total_calls", 0)
                        status.success_rate = call_stats.get("success_rate", 0.0)
                        status.active_thinking_sessions = call_stats.get("active_calls", 0)
                        
                        # Update OpenRouter connection status
                        status.openrouter_connected = self._check_openrouter_connection_health(
                            agent.agent_id, api_client
                        )
                
                # Check if agent needs attention and attempt recovery
                if status.needs_attention():
                    logger.warning(f"Agent {agent.agent_id} needs attention: "
                                 f"connected={status.openrouter_connected}, "
                                 f"success_rate={status.success_rate:.2f}, "
                                 f"last_activity={time.time() - status.last_activity:.0f}s ago")
                    
                    # Attempt automatic recovery for unhealthy agents
                    if not status.is_healthy():
                        recovery_success = self._attempt_agent_recovery(agent.agent_id)
                        if not recovery_success:
                            # If recovery failed, redistribute workload
                            self.redistribute_workload(agent.agent_id)
                
                # Check for OpenRouter API availability and handle degradation
                self._check_and_handle_openrouter_degradation()
    
    def _attempt_agent_recovery(self, agent_id: str) -> bool:
        """
        Attempt to recover a failed or unhealthy agent
        
        Args:
            agent_id: ID of the agent to recover
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting recovery for agent {agent_id}")
        
        try:
            # Find the agent and its core
            agent = None
            core_id = None
            
            with self._agents_lock:
                for cid, a in self.agents.items():
                    if a.agent_id == agent_id:
                        agent = a
                        core_id = cid
                        break
            
            if not agent or core_id is None:
                logger.error(f"Agent {agent_id} not found for recovery")
                return False
            
            # Get current status
            with self._status_lock:
                if agent_id not in self.agent_statuses:
                    logger.error(f"No status found for agent {agent_id}")
                    return False
                
                status = self.agent_statuses[agent_id]
            
            # Determine recovery strategy based on the type of failure
            recovery_success = False
            
            # Strategy 1: OpenRouter connection issues
            if not status.openrouter_connected and self.openrouter_api_key:
                logger.info(f"Attempting OpenRouter connection recovery for agent {agent_id}")
                recovery_success = self._recover_openrouter_connection(agent_id, agent)
            
            # Strategy 2: Agent responsiveness issues
            elif status.status == AgentStatus.OFFLINE:
                logger.info(f"Attempting agent restart for agent {agent_id}")
                recovery_success = self._restart_agent(agent_id, core_id, agent.role)
            
            # Strategy 3: Low success rate - reset API client
            elif status.success_rate < 0.3:
                logger.info(f"Attempting API client reset for agent {agent_id}")
                recovery_success = self._reset_agent_api_client(agent_id, agent)
            
            # Strategy 4: General health issues - full agent reset
            else:
                logger.info(f"Attempting full agent reset for agent {agent_id}")
                recovery_success = self._restart_agent(agent_id, core_id, agent.role)
            
            if recovery_success:
                logger.info(f"Successfully recovered agent {agent_id}")
                
                # Update status to reflect recovery
                with self._status_lock:
                    if agent_id in self.agent_statuses:
                        self.agent_statuses[agent_id].last_activity = time.time()
                
                # Notify other agents about successful recovery
                if self.communication_system:
                    for other_agent in self.agents.values():
                        if other_agent.agent_id != agent_id:
                            self.communication_system.send_message(
                                from_agent="system",
                                to_agent=other_agent.agent_id,
                                message_type="info",
                                content=f"Agent {agent_id} has been successfully recovered.",
                                priority=5
                            )
            else:
                logger.warning(f"Failed to recover agent {agent_id}, redistributing workload")
                self.redistribute_workload(agent_id)
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Error during agent recovery for {agent_id}: {e}")
            return False
    
    def _recover_openrouter_connection(self, agent_id: str, agent: Agent) -> bool:
        """
        Attempt to recover OpenRouter connection for an agent
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            logger.debug(f"Recovering OpenRouter connection for agent {agent_id}")
            
            # Reconfigure OpenRouter connection
            self._configure_agent_openrouter(agent)
            
            # Verify the connection is now healthy
            if self.communication_system:
                api_client = self.communication_system.get_agent_api_client(agent_id)
                if api_client:
                    connection_healthy = self._check_openrouter_connection_health(agent_id, api_client)
                    
                    # Update status
                    with self._status_lock:
                        if agent_id in self.agent_statuses:
                            self.agent_statuses[agent_id].openrouter_connected = connection_healthy
                    
                    return connection_healthy
            
            return False
            
        except Exception as e:
            logger.error(f"Error recovering OpenRouter connection for agent {agent_id}: {e}")
            return False
    
    def _restart_agent(self, agent_id: str, core_id: int, role: AgentRole) -> bool:
        """
        Restart an agent by recreating it on the same core
        
        Args:
            agent_id: ID of the agent to restart
            core_id: Core ID where the agent should be restarted
            role: Role of the agent
            
        Returns:
            True if restart was successful, False otherwise
        """
        try:
            logger.debug(f"Restarting agent {agent_id} on core {core_id}")
            
            # Remove the old agent
            with self._agents_lock:
                if core_id in self.agents:
                    old_agent = self.agents[core_id]
                    
                    # Unregister from communication system
                    if self.communication_system:
                        self.communication_system.unregister_agent(old_agent.agent_id)
                    
                    # Remove from agents dict
                    del self.agents[core_id]
            
            # Remove old status
            with self._status_lock:
                if agent_id in self.agent_statuses:
                    del self.agent_statuses[agent_id]
            
            # Create new agent with same role and core
            if self.communication_system:
                new_agent = self.communication_system.register_agent(
                    core_id=core_id,
                    role=role,
                    agent_id=f"core_{core_id}_{role.value}"
                )
                
                # Store new agent
                with self._agents_lock:
                    self.agents[core_id] = new_agent
                
                # Initialize status tracking
                self._initialize_agent_status(new_agent)
                
                # Configure OpenRouter if available
                if self.openrouter_api_key:
                    self._configure_agent_openrouter(new_agent)
                
                logger.info(f"Successfully restarted agent {new_agent.agent_id} on core {core_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error restarting agent {agent_id} on core {core_id}: {e}")
            return False
    
    def _reset_agent_api_client(self, agent_id: str, agent: Agent) -> bool:
        """
        Reset the API client for an agent to clear any problematic state
        
        Args:
            agent_id: ID of the agent
            agent: Agent instance
            
        Returns:
            True if reset was successful, False otherwise
        """
        try:
            logger.debug(f"Resetting API client for agent {agent_id}")
            
            if self.communication_system:
                # Get current API client
                api_client = self.communication_system.get_agent_api_client(agent_id)
                if api_client:
                    # Reset API client state
                    api_client.reset_statistics()
                    
                    # Reconfigure OpenRouter if needed
                    if self.openrouter_api_key and api_client.openrouter_only:
                        openrouter_config = self._create_openrouter_config_for_role(agent.role)
                        api_client.configure_openrouter_only(self.openrouter_api_key, openrouter_config)
                    
                    # Update agent status
                    with self._status_lock:
                        if agent_id in self.agent_statuses:
                            self.agent_statuses[agent_id].success_rate = 1.0  # Reset to optimistic
                            self.agent_statuses[agent_id].total_api_calls = 0
                            self.agent_statuses[agent_id].last_activity = time.time()
                    
                    logger.info(f"Successfully reset API client for agent {agent_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resetting API client for agent {agent_id}: {e}")
            return False
    
    def force_agent_recovery(self, agent_id: str) -> Dict[str, Any]:
        """
        Force recovery attempt for a specific agent (manual trigger)
        
        Args:
            agent_id: ID of the agent to recover
            
        Returns:
            Dictionary with recovery results
        """
        logger.info(f"Manual recovery triggered for agent {agent_id}")
        
        result = {
            "agent_id": agent_id,
            "recovery_attempted": False,
            "recovery_successful": False,
            "error": None,
            "recovery_strategy": None
        }
        
        try:
            # Check if agent exists
            agent_exists = False
            with self._agents_lock:
                for agent in self.agents.values():
                    if agent.agent_id == agent_id:
                        agent_exists = True
                        break
            
            if not agent_exists:
                result["error"] = f"Agent {agent_id} not found"
                return result
            
            # Check current status
            with self._status_lock:
                if agent_id not in self.agent_statuses:
                    result["error"] = f"No status found for agent {agent_id}"
                    return result
                
                status = self.agent_statuses[agent_id]
                
                # Determine recovery strategy
                if not status.openrouter_connected:
                    result["recovery_strategy"] = "openrouter_connection_recovery"
                elif status.status == AgentStatus.OFFLINE:
                    result["recovery_strategy"] = "agent_restart"
                elif status.success_rate < 0.3:
                    result["recovery_strategy"] = "api_client_reset"
                else:
                    result["recovery_strategy"] = "full_agent_reset"
            
            # Attempt recovery
            result["recovery_attempted"] = True
            result["recovery_successful"] = self._attempt_agent_recovery(agent_id)
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error in manual recovery for agent {agent_id}: {e}")
        
        return result
    
    def _activate_emergency_fallback_mode(self, failed_agent: Agent) -> None:
        """
        Activate emergency fallback mode when no healthy agents are available
        
        Args:
            failed_agent: The failed agent that triggered emergency mode
        """
        logger.critical(f"Activating emergency fallback mode due to failure of {failed_agent.agent_id}")
        
        try:
            # Store emergency state
            self._emergency_mode = True
            self._emergency_mode_activated_at = time.time()
            self._emergency_mode_reason = f"No healthy agents available after {failed_agent.agent_id} failure"
            
            # Attempt to create a minimal single-agent fallback
            if self.communication_system:
                try:
                    # Try to create a generalist agent as emergency backup
                    from utils.agent_core import AgentRole
                    
                    emergency_agent = self.communication_system.register_agent(
                        core_id=0,  # Use core 0 for emergency agent
                        role=AgentRole.GENERALIST,
                        agent_id=f"emergency_agent_{int(time.time())}"
                    )
                    
                    # Store emergency agent
                    with self._agents_lock:
                        self.agents[0] = emergency_agent
                    
                    # Initialize status tracking
                    self._initialize_agent_status(emergency_agent)
                    
                    # Configure with basic settings (no OpenRouter to avoid further failures)
                    emergency_agent.update_status(AgentStatus.ACTIVE)
                    
                    logger.info(f"Emergency agent {emergency_agent.agent_id} created successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to create emergency agent: {e}")
                    # System is in critical state - log for manual intervention
                    logger.critical("SYSTEM CRITICAL: No agents available and emergency agent creation failed!")
            
        except Exception as e:
            logger.error(f"Error activating emergency fallback mode: {e}")
    
    def _calculate_additional_load(self, failed_agent: Agent, target_agent: Agent, total_targets: int) -> Dict[str, Any]:
        """
        Calculate the additional load a target agent will receive from redistribution
        
        Args:
            failed_agent: The failed agent whose load is being redistributed
            target_agent: The agent receiving additional load
            total_targets: Total number of agents receiving redistributed load
            
        Returns:
            Dictionary with load calculation details
        """
        try:
            # Get failed agent's recent activity metrics
            failed_agent_load = 0
            if failed_agent.agent_id in self.agent_statuses:
                status = self.agent_statuses[failed_agent.agent_id]
                failed_agent_load = status.active_thinking_sessions + (status.total_api_calls / 100)
            
            # Calculate proportional load distribution
            additional_load_factor = failed_agent_load / total_targets if total_targets > 0 else 0
            
            # Role compatibility factor (same role agents get more load)
            role_compatibility = 1.0 if target_agent.role == failed_agent.role else 0.7
            
            # Target agent's current capacity
            target_capacity = 1.0  # Default capacity
            if target_agent.agent_id in self.agent_statuses:
                target_status = self.agent_statuses[target_agent.agent_id]
                # Reduce capacity based on current load
                current_load = target_status.active_thinking_sessions / 10.0  # Normalize
                target_capacity = max(0.1, 1.0 - current_load)
            
            adjusted_additional_load = additional_load_factor * role_compatibility * target_capacity
            
            return {
                "base_additional_load": additional_load_factor,
                "role_compatibility_factor": role_compatibility,
                "target_capacity_factor": target_capacity,
                "adjusted_additional_load": adjusted_additional_load,
                "load_increase_percentage": min(100, adjusted_additional_load * 100)
            }
            
        except Exception as e:
            logger.error(f"Error calculating additional load: {e}")
            return {
                "base_additional_load": 0.1,  # Minimal fallback
                "role_compatibility_factor": 1.0,
                "target_capacity_factor": 1.0,
                "adjusted_additional_load": 0.1,
                "load_increase_percentage": 10
            }
    
    def _get_priority_tasks_for_role(self, role: 'AgentRole') -> List[str]:
        """
        Get priority tasks that should be handled first for a specific role
        
        Args:
            role: The agent role to get priority tasks for
            
        Returns:
            List of priority task descriptions
        """
        from utils.agent_core import AgentRole
        
        role_priority_tasks = {
            AgentRole.SECURITY_ANALYST: [
                "Security vulnerability analysis",
                "Code security review",
                "Authentication and authorization checks",
                "Input validation and sanitization review"
            ],
            AgentRole.PERFORMANCE_OPTIMIZER: [
                "Performance bottleneck identification",
                "Resource usage optimization",
                "Algorithm efficiency analysis",
                "Memory leak detection"
            ],
            AgentRole.ARCHITECTURE_REVIEWER: [
                "System architecture validation",
                "Design pattern compliance",
                "Scalability assessment",
                "Component integration review"
            ],
            AgentRole.CODE_QUALITY_INSPECTOR: [
                "Code style and standards compliance",
                "Code complexity analysis",
                "Maintainability assessment",
                "Documentation quality review"
            ],
            AgentRole.DEBUG_SPECIALIST: [
                "Error reproduction and analysis",
                "Stack trace investigation",
                "Runtime behavior analysis",
                "Bug root cause identification"
            ],
            AgentRole.PLANNING_COORDINATOR: [
                "Task prioritization and scheduling",
                "Resource allocation planning",
                "Timeline estimation",
                "Dependency management"
            ],
            AgentRole.CONSENSUS_FACILITATOR: [
                "Multi-agent decision coordination",
                "Conflict resolution",
                "Opinion synthesis",
                "Consensus building"
            ],
            AgentRole.GENERALIST: [
                "General code analysis",
                "Basic troubleshooting",
                "Documentation assistance",
                "Simple task coordination"
            ]
        }
        
        return role_priority_tasks.get(role, role_priority_tasks[AgentRole.GENERALIST])
    
    def _update_load_balancing_metrics(self, failed_agent_id: str, target_agents: List[Agent]) -> None:
        """
        Update system-wide load balancing metrics after workload redistribution
        
        Args:
            failed_agent_id: ID of the failed agent
            target_agents: List of agents receiving redistributed workload
        """
        try:
            # Update redistribution history
            if not hasattr(self, '_redistribution_history'):
                self._redistribution_history = []
            
            redistribution_event = {
                "timestamp": time.time(),
                "failed_agent_id": failed_agent_id,
                "target_agent_ids": [agent.agent_id for agent in target_agents],
                "target_agent_roles": [agent.role.value for agent in target_agents],
                "redistribution_count": len(target_agents)
            }
            
            self._redistribution_history.append(redistribution_event)
            
            # Keep only recent history (last 100 events)
            if len(self._redistribution_history) > 100:
                self._redistribution_history = self._redistribution_history[-100:]
            
            # Update load balancing statistics
            with self._status_lock:
                for target_agent in target_agents:
                    if target_agent.agent_id in self.agent_statuses:
                        # Increment expected load for target agents
                        status = self.agent_statuses[target_agent.agent_id]
                        # This would be used by the load balancer to make decisions
                        if not hasattr(status, 'redistributed_load_factor'):
                            status.redistributed_load_factor = 1.0
                        status.redistributed_load_factor += 0.1  # Small increment per redistribution
            
            logger.debug(f"Updated load balancing metrics for redistribution from {failed_agent_id}")
            
        except Exception as e:
            logger.error(f"Error updating load balancing metrics: {e}")

    def handle_openrouter_api_unavailable(self) -> Dict[str, Any]:
        """
        Handle graceful degradation when OpenRouter API is unavailable
        
        Returns:
            Dictionary with degradation strategy results
        """
        logger.warning("OpenRouter API unavailable - implementing graceful degradation strategies")
        
        degradation_result = {
            "strategy_applied": None,
            "agents_affected": 0,
            "agents_switched_to_fallback": 0,
            "fallback_providers_available": [],
            "system_health_impact": "minimal",
            "recovery_actions": []
        }
        
        try:
            # Check which fallback providers are available
            from providers.registry import ModelProviderRegistry
            from providers.base import ProviderType
            
            registry = ModelProviderRegistry()
            available_providers = registry.get_available_providers()
            
            # Prioritize fallback providers
            fallback_priority = [
                ProviderType.OPENAI,
                ProviderType.GOOGLE,
                ProviderType.XAI,
                ProviderType.DIAL,
                ProviderType.CUSTOM
            ]
            
            available_fallbacks = [
                provider for provider in fallback_priority
                if provider in available_providers and provider != ProviderType.OPENROUTER
            ]
            
            degradation_result["fallback_providers_available"] = [p.value for p in available_fallbacks]
            
            if not available_fallbacks:
                # Critical situation - no fallback providers available
                degradation_result["strategy_applied"] = "emergency_local_mode"
                degradation_result["system_health_impact"] = "severe"
                logger.critical("No fallback providers available! System entering emergency local mode")
                self._activate_emergency_local_mode()
                return degradation_result
            
            # Strategy 1: Switch OpenRouter-only agents to fallback providers
            with self._agents_lock:
                openrouter_agents = []
                
                for agent in self.agents.values():
                    if self.communication_system:
                        api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                        if api_client and api_client.openrouter_only:
                            openrouter_agents.append(agent)
                
                degradation_result["agents_affected"] = len(openrouter_agents)
                
                if openrouter_agents:
                    degradation_result["strategy_applied"] = "fallback_provider_switch"
                    
                    for agent in openrouter_agents:
                        try:
                            # Switch agent to use fallback providers
                            success = self._switch_agent_to_fallback_providers(agent, available_fallbacks)
                            if success:
                                degradation_result["agents_switched_to_fallback"] += 1
                                
                                # Notify agent about the switch
                                if self.communication_system:
                                    self.communication_system.send_message(
                                        from_agent="system",
                                        to_agent=agent.agent_id,
                                        message_type="provider_switch",
                                        content={
                                            "reason": "openrouter_unavailable",
                                            "new_providers": [p.value for p in available_fallbacks],
                                            "temporary": True,
                                            "auto_revert": True
                                        },
                                        priority=7
                                    )
                                
                        except Exception as e:
                            logger.error(f"Failed to switch agent {agent.agent_id} to fallback: {e}")
                
                # Determine system health impact
                if degradation_result["agents_switched_to_fallback"] == degradation_result["agents_affected"]:
                    degradation_result["system_health_impact"] = "minimal"
                elif degradation_result["agents_switched_to_fallback"] > degradation_result["agents_affected"] * 0.7:
                    degradation_result["system_health_impact"] = "moderate"
                else:
                    degradation_result["system_health_impact"] = "significant"
                
                # Add recovery actions
                degradation_result["recovery_actions"] = [
                    "Monitor OpenRouter API status for recovery",
                    "Automatically revert to OpenRouter when available",
                    "Track fallback provider performance",
                    "Alert administrators if degradation persists"
                ]
                
                logger.info(f"Graceful degradation completed: {degradation_result['agents_switched_to_fallback']}"
                          f"/{degradation_result['agents_affected']} agents switched to fallback providers")
        
        except Exception as e:
            error_msg = f"Error during graceful degradation: {e}"
            logger.error(error_msg)
            degradation_result["error"] = error_msg
            degradation_result["strategy_applied"] = "error_fallback"
        
        return degradation_result
    
    def _switch_agent_to_fallback_providers(self, agent: Agent, fallback_providers: List[ProviderType]) -> bool:
        """
        Switch an agent from OpenRouter-only mode to fallback providers
        
        Args:
            agent: The agent to switch
            fallback_providers: List of available fallback providers
            
        Returns:
            True if switch was successful, False otherwise
        """
        try:
            if not self.communication_system:
                return False
            
            api_client = self.communication_system.get_agent_api_client(agent.agent_id)
            if not api_client:
                return False
            
            # Store original OpenRouter configuration for later restoration
            if not hasattr(api_client, '_original_openrouter_config'):
                api_client._original_openrouter_config = api_client.openrouter_config
                api_client._original_openrouter_only = api_client.openrouter_only
            
            # Disable OpenRouter-only mode
            api_client.openrouter_only = False
            
            # Update preferred providers to use fallbacks
            api_client.preferred_providers = fallback_providers
            
            # Reset circuit breaker state since we're switching providers
            api_client.openrouter_failure_count = 0
            api_client.openrouter_circuit_open = False
            
            # Update agent status
            with self._status_lock:
                if agent.agent_id in self.agent_statuses:
                    self.agent_statuses[agent.agent_id].openrouter_connected = False
                    # Mark as degraded mode
                    if not hasattr(self.agent_statuses[agent.agent_id], 'degraded_mode'):
                        self.agent_statuses[agent.agent_id].degraded_mode = True
                        self.agent_statuses[agent.agent_id].degraded_since = time.time()
            
            logger.info(f"Successfully switched agent {agent.agent_id} to fallback providers: "
                       f"{[p.value for p in fallback_providers]}")
            return True
            
        except Exception as e:
            logger.error(f"Error switching agent {agent.agent_id} to fallback providers: {e}")
            return False
    
    def _activate_emergency_local_mode(self) -> None:
        """
        Activate emergency local mode when no API providers are available
        """
        logger.critical("Activating emergency local mode - no API providers available")
        
        try:
            # Mark system as in emergency mode
            self._emergency_mode = True
            self._emergency_local_mode = True
            self._emergency_mode_activated_at = time.time()
            self._emergency_mode_reason = "No API providers available"
            
            # Disable all agents' API capabilities
            with self._agents_lock:
                for agent in self.agents.values():
                    try:
                        agent.update_status(AgentStatus.DEGRADED)
                        
                        # Update status to reflect emergency mode
                        with self._status_lock:
                            if agent.agent_id in self.agent_statuses:
                                status = self.agent_statuses[agent.agent_id]
                                status.openrouter_connected = False
                                status.emergency_mode = True
                                status.emergency_mode_since = time.time()
                        
                    except Exception as e:
                        logger.error(f"Error setting emergency mode for agent {agent.agent_id}: {e}")
            
            # Log critical system state
            logger.critical("SYSTEM IN EMERGENCY LOCAL MODE: All AI API capabilities disabled")
            logger.critical("Manual intervention required to restore API provider connectivity")
            
        except Exception as e:
            logger.error(f"Error activating emergency local mode: {e}")
    
    def check_and_recover_from_degradation(self) -> Dict[str, Any]:
        """
        Check if system can recover from degraded state and attempt recovery
        
        Returns:
            Dictionary with recovery attempt results
        """
        recovery_result = {
            "recovery_attempted": False,
            "recovery_successful": False,
            "agents_recovered": 0,
            "agents_still_degraded": 0,
            "openrouter_available": False,
            "error": None
        }
        
        try:
            # Check if OpenRouter is available again
            if self.openrouter_api_key:
                recovery_result["openrouter_available"] = self._test_openrouter_availability()
            
            if not recovery_result["openrouter_available"]:
                logger.debug("OpenRouter still unavailable, skipping recovery attempt")
                return recovery_result
            
            recovery_result["recovery_attempted"] = True
            logger.info("OpenRouter available again - attempting recovery from degraded state")
            
            # Recover agents that were switched to fallback providers
            with self._agents_lock:
                degraded_agents = []
                
                for agent in self.agents.values():
                    if self.communication_system:
                        api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                        if (api_client and 
                            hasattr(api_client, '_original_openrouter_config') and
                            api_client._original_openrouter_config):
                            degraded_agents.append(agent)
                
                for agent in degraded_agents:
                    try:
                        success = self._recover_agent_from_degradation(agent)
                        if success:
                            recovery_result["agents_recovered"] += 1
                        else:
                            recovery_result["agents_still_degraded"] += 1
                            
                    except Exception as e:
                        logger.error(f"Error recovering agent {agent.agent_id} from degradation: {e}")
                        recovery_result["agents_still_degraded"] += 1
            
            # Clear emergency mode if all agents recovered
            if recovery_result["agents_still_degraded"] == 0:
                self._emergency_mode = False
                self._emergency_local_mode = False
                recovery_result["recovery_successful"] = True
                logger.info("Successfully recovered all agents from degraded state")
            else:
                logger.warning(f"Partial recovery: {recovery_result['agents_recovered']} recovered, "
                             f"{recovery_result['agents_still_degraded']} still degraded")
        
        except Exception as e:
            error_msg = f"Error during degradation recovery: {e}"
            logger.error(error_msg)
            recovery_result["error"] = error_msg
        
        return recovery_result
    
    def _test_openrouter_availability(self) -> bool:
        """
        Test if OpenRouter API is available
        
        Returns:
            True if OpenRouter is available, False otherwise
        """
        try:
            from providers.registry import ModelProviderRegistry
            from providers.base import ProviderType
            
            registry = ModelProviderRegistry()
            openrouter_provider = registry.get_provider(ProviderType.OPENROUTER)
            
            if not openrouter_provider:
                return False
            
            # Test with a simple model list request (lightweight test)
            available_models = registry.get_available_model_names(ProviderType.OPENROUTER)
            return len(available_models) > 0
            
        except Exception as e:
            logger.debug(f"OpenRouter availability test failed: {e}")
            return False
    
    def _recover_agent_from_degradation(self, agent: Agent) -> bool:
        """
        Recover an agent from degraded state back to OpenRouter
        
        Args:
            agent: The agent to recover
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            if not self.communication_system:
                return False
            
            api_client = self.communication_system.get_agent_api_client(agent.agent_id)
            if not api_client or not hasattr(api_client, '_original_openrouter_config'):
                return False
            
            # Restore original OpenRouter configuration
            api_client.openrouter_config = api_client._original_openrouter_config
            api_client.openrouter_only = api_client._original_openrouter_only
            
            # Reset preferred providers to OpenRouter
            api_client.preferred_providers = [ProviderType.OPENROUTER]
            
            # Reset circuit breaker state
            api_client.openrouter_failure_count = 0
            api_client.openrouter_circuit_open = False
            
            # Test the restored connection
            connection_healthy = self._check_openrouter_connection_health(agent.agent_id, api_client)
            
            if connection_healthy:
                # Update agent status
                with self._status_lock:
                    if agent.agent_id in self.agent_statuses:
                        status = self.agent_statuses[agent.agent_id]
                        status.openrouter_connected = True
                        if hasattr(status, 'degraded_mode'):
                            status.degraded_mode = False
                            delattr(status, 'degraded_since')
                
                # Clean up temporary degradation state
                delattr(api_client, '_original_openrouter_config')
                delattr(api_client, '_original_openrouter_only')
                
                # Notify agent about recovery
                if self.communication_system:
                    self.communication_system.send_message(
                        from_agent="system",
                        to_agent=agent.agent_id,
                        message_type="provider_recovery",
                        content={
                            "message": "OpenRouter connection restored",
                            "providers_restored": ["openrouter"],
                            "degradation_duration": time.time() - getattr(
                                self.agent_statuses.get(agent.agent_id, {}), 'degraded_since', time.time()
                            )
                        },
                        priority=6
                    )
                
                logger.info(f"Successfully recovered agent {agent.agent_id} from degraded state")
                return True
            else:
                logger.warning(f"OpenRouter connection test failed for agent {agent.agent_id} during recovery")
                return False
                
        except Exception as e:
            logger.error(f"Error recovering agent {agent.agent_id} from degradation: {e}")
            return False

    def _check_and_handle_openrouter_degradation(self) -> None:
        """
        Check for OpenRouter API degradation and handle it proactively
        """
        try:
            if not self.openrouter_api_key:
                return
            
            # Count agents with circuit breakers open
            circuit_breaker_count = 0
            total_openrouter_agents = 0
            
            with self._agents_lock:
                for agent in self.agents.values():
                    if self.communication_system:
                        api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                        if api_client and api_client.openrouter_only:
                            total_openrouter_agents += 1
                            if api_client._is_openrouter_circuit_open():
                                circuit_breaker_count += 1
            
            # If more than 50% of OpenRouter agents have circuit breakers open, 
            # consider system-wide degradation
            if total_openrouter_agents > 0:
                degradation_ratio = circuit_breaker_count / total_openrouter_agents
                
                if degradation_ratio >= 0.5 and not getattr(self, '_system_degradation_handled', False):
                    logger.warning(f"System-wide OpenRouter degradation detected: "
                                 f"{circuit_breaker_count}/{total_openrouter_agents} agents affected")
                    
                    # Handle system-wide degradation
                    degradation_result = self.handle_openrouter_api_unavailable()
                    self._system_degradation_handled = True
                    self._system_degradation_time = time.time()
                    
                    logger.info(f"System degradation handled: {degradation_result['strategy_applied']}")
                
                elif degradation_ratio < 0.2 and getattr(self, '_system_degradation_handled', False):
                    # Recovery detected - attempt to restore normal operation
                    logger.info("OpenRouter degradation appears to be resolving, checking for recovery")
                    recovery_result = self.check_and_recover_from_degradation()
                    
                    if recovery_result.get('recovery_successful', False):
                        self._system_degradation_handled = False
                        if hasattr(self, '_system_degradation_time'):
                            delattr(self, '_system_degradation_time')
                        logger.info("Successfully recovered from system-wide OpenRouter degradation")
        
        except Exception as e:
            logger.error(f"Error checking OpenRouter degradation: {e}")

    def get_workload_redistribution_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of workload redistribution events
        
        Returns:
            List of redistribution events with timestamps and details
        """
        return self._redistribution_history.copy()
    
    def get_emergency_mode_status(self) -> Dict[str, Any]:
        """
        Get current emergency mode status
        
        Returns:
            Dictionary with emergency mode information
        """
        return {
            "emergency_mode_active": self._emergency_mode,
            "emergency_local_mode_active": getattr(self, '_emergency_local_mode', False),
            "activated_at": self._emergency_mode_activated_at,
            "reason": self._emergency_mode_reason,
            "duration_seconds": time.time() - self._emergency_mode_activated_at if self._emergency_mode else 0,
            "system_degradation_handled": getattr(self, '_system_degradation_handled', False),
            "system_degradation_time": getattr(self, '_system_degradation_time', 0)
        }

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about agent recovery attempts
        
        Returns:
            Dictionary with recovery statistics
        """
        # This would be enhanced with persistent recovery tracking
        # For now, return basic information from current session
        
        with self._agents_lock, self._status_lock:
            stats = {
                "total_agents": len(self.agents),
                "healthy_agents": sum(1 for status in self.agent_statuses.values() if status.is_healthy()),
                "agents_needing_attention": sum(1 for status in self.agent_statuses.values() if status.needs_attention()),
                "agents_with_openrouter_issues": sum(
                    1 for status in self.agent_statuses.values() 
                    if not status.openrouter_connected and self.openrouter_api_key
                ),
                "agents_with_low_success_rate": sum(
                    1 for status in self.agent_statuses.values() 
                    if status.success_rate < 0.5
                ),
                "offline_agents": sum(
                    1 for status in self.agent_statuses.values() 
                    if status.status == AgentStatus.OFFLINE
                ),
                "recovery_recommendations": []
            }
            
            # Generate recovery recommendations
            for agent_id, status in self.agent_statuses.items():
                if status.needs_attention():
                    recommendation = {
                        "agent_id": agent_id,
                        "issue": [],
                        "recommended_action": []
                    }
                    
                    if not status.openrouter_connected and self.openrouter_api_key:
                        recommendation["issue"].append("OpenRouter connection failed")
                        recommendation["recommended_action"].append("Reconfigure OpenRouter connection")
                    
                    if status.success_rate < 0.3:
                        recommendation["issue"].append(f"Low success rate: {status.success_rate:.2f}")
                        recommendation["recommended_action"].append("Reset API client")
                    
                    if status.status == AgentStatus.OFFLINE:
                        recommendation["issue"].append("Agent offline")
                        recommendation["recommended_action"].append("Restart agent")
                    
                    if (time.time() - status.last_activity) > 600:
                        recommendation["issue"].append("Agent inactive for >10 minutes")
                        recommendation["recommended_action"].append("Check agent responsiveness")
                    
                    if recommendation["issue"]:
                        stats["recovery_recommendations"].append(recommendation)
            
            return stats


    def create_individual_thinking_session(
        self,
        agent_id: str,
        thinking_prompt: str,
        thinking_mode: str = "high",
        timeout_seconds: float = 300.0,
        priority: int = 5
    ) -> Optional["AgentThinkingSession"]:
        """
        Create an individual thinking session for a specific agent
        
        Args:
            agent_id: ID of the agent to create the session for
            thinking_prompt: The prompt for the thinking session
            thinking_mode: Thinking mode intensity ("low", "standard", "high")
            timeout_seconds: Session timeout in seconds
            priority: Session priority (1-10)
            
        Returns:
            AgentThinkingSession instance if successful, None otherwise
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        # Find the agent
        agent = None
        with self._agents_lock:
            for a in self.agents.values():
                if a.agent_id == agent_id:
                    agent = a
                    break
        
        if not agent:
            logger.error(f"Agent {agent_id} not found for thinking session creation")
            return None
        
        # Get the thinking session coordinator
        coordinator = get_thinking_session_coordinator()
        
        # Create the thinking session
        session = coordinator.create_individual_session(
            agent=agent,
            thinking_prompt=thinking_prompt,
            model_used="auto",
            thinking_mode=thinking_mode,
            timeout_seconds=timeout_seconds,
            priority=priority
        )
        
        logger.info(f"Created individual thinking session {session.session_id} for agent {agent_id}")
        return session
    
    def create_synchronized_thinking_session(
        self,
        thinking_prompt: str,
        participating_agent_ids: Optional[List[str]] = None,
        session_type: str = "synchronized",
        timeout_seconds: float = 300.0,
        priority: int = 5,
        require_all_agents: bool = True,
        aggregation_strategy: str = "consensus"
    ) -> Optional[Dict[str, "AgentThinkingSession"]]:
        """
        Create a synchronized thinking session across multiple agents
        
        Args:
            thinking_prompt: The prompt for the thinking session
            participating_agent_ids: List of agent IDs to participate (None = all agents)
            session_type: Type of synchronized session
            timeout_seconds: Session timeout in seconds
            priority: Session priority (1-10)
            require_all_agents: Whether all agents must participate
            aggregation_strategy: Strategy for aggregating results
            
        Returns:
            Dictionary mapping agent_id to AgentThinkingSession, or None if failed
        """
        from utils.agent_thinking_session import (
            get_thinking_session_coordinator,
            SynchronizedThinkingRequest,
            ThinkingSessionType
        )
        
        # Determine participating agents
        if participating_agent_ids is None:
            with self._agents_lock:
                participating_agent_ids = [agent.agent_id for agent in self.agents.values()]
        
        if not participating_agent_ids:
            logger.error("No participating agents specified for synchronized thinking session")
            return None
        
        # Validate that all specified agents exist
        with self._agents_lock:
            available_agent_ids = {agent.agent_id for agent in self.agents.values()}
            invalid_agents = set(participating_agent_ids) - available_agent_ids
            
            if invalid_agents:
                logger.error(f"Invalid agent IDs for synchronized session: {invalid_agents}")
                return None
        
        # Map session type string to enum
        session_type_map = {
            "synchronized": ThinkingSessionType.SYNCHRONIZED,
            "collaborative": ThinkingSessionType.COLLABORATIVE,
            "consensus": ThinkingSessionType.CONSENSUS
        }
        
        session_type_enum = session_type_map.get(session_type, ThinkingSessionType.SYNCHRONIZED)
        
        # Create synchronized thinking request
        request = SynchronizedThinkingRequest(
            request_id=str(uuid.uuid4()),
            thinking_prompt=thinking_prompt,
            participating_agent_ids=participating_agent_ids,
            session_type=session_type_enum,
            timeout_seconds=timeout_seconds,
            priority=priority,
            require_all_agents=require_all_agents,
            aggregation_strategy=aggregation_strategy
        )
        
        # Get the thinking session coordinator
        coordinator = get_thinking_session_coordinator()
        
        # Create synchronized sessions
        sessions = coordinator.create_synchronized_session(request)
        
        logger.info(f"Created synchronized thinking session {request.request_id} with {len(sessions)} agents")
        return sessions
    
    def execute_agent_thinking_session(
        self,
        agent_id: str,
        session_id: str
    ) -> bool:
        """
        Execute a thinking session for a specific agent
        
        Args:
            agent_id: ID of the agent to execute the session
            session_id: ID of the thinking session to execute
            
        Returns:
            True if execution was started successfully, False otherwise
        """
        # Find the agent
        agent = None
        with self._agents_lock:
            for a in self.agents.values():
                if a.agent_id == agent_id:
                    agent = a
                    break
        
        if not agent:
            logger.error(f"Agent {agent_id} not found for thinking session execution")
            return False
        
        # Get the agent's API client
        if not self.communication_system:
            logger.error("Communication system not available for thinking session execution")
            return False
        
        api_client = self.communication_system.get_agent_api_client(agent_id)
        if not api_client:
            logger.error(f"API client not found for agent {agent_id}")
            return False
        
        # Get the thinking session
        from utils.agent_thinking_session import get_thinking_session_coordinator
        coordinator = get_thinking_session_coordinator()
        session = coordinator.get_session(session_id)
        
        if not session:
            logger.error(f"Thinking session {session_id} not found")
            return False
        
        if session.agent_id != agent_id:
            logger.error(f"Session {session_id} does not belong to agent {agent_id}")
            return False
        
        # Execute the session asynchronously
        import asyncio
        try:
            # Create a task to execute the thinking session
            asyncio.create_task(api_client.create_and_execute_thinking_session(
                thinking_prompt=session.thinking_prompt,
                thinking_mode=session.thinking_mode,
                timeout_seconds=session.timeout_seconds,
                priority=session.priority
            ))
            
            logger.info(f"Started execution of thinking session {session_id} for agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting thinking session execution for agent {agent_id}: {e}")
            return False
    
    def get_thinking_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a thinking session
        
        Args:
            session_id: ID of the thinking session
            
        Returns:
            Dictionary with session status information, or None if not found
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        session = coordinator.get_session(session_id)
        
        if not session:
            return None
        
        return session.to_dict()
    
    def get_agent_thinking_sessions(
        self,
        agent_id: str,
        include_completed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all thinking sessions for a specific agent
        
        Args:
            agent_id: ID of the agent
            include_completed: Whether to include completed sessions
            
        Returns:
            List of session dictionaries
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        sessions = coordinator.get_agent_sessions(agent_id, include_completed)
        
        return [session.to_dict() for session in sessions]
    
    def get_synchronized_thinking_results(
        self,
        parent_session_id: str,
        aggregation_strategy: str = "consensus"
    ) -> Optional[Dict[str, Any]]:
        """
        Get aggregated results from a synchronized thinking session
        
        Args:
            parent_session_id: ID of the parent synchronized session
            aggregation_strategy: Strategy for aggregating results
            
        Returns:
            Aggregated results dictionary or None if not ready
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        return coordinator.aggregate_synchronized_results(parent_session_id, aggregation_strategy)
    
    def get_thinking_session_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive thinking session statistics across all agents
        
        Returns:
            Dictionary with thinking session metrics
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        system_stats = coordinator.get_session_statistics()
        
        # Add per-agent thinking session statistics
        agent_stats = {}
        with self._agents_lock:
            for agent in self.agents.values():
                if self.communication_system:
                    api_client = self.communication_system.get_agent_api_client(agent.agent_id)
                    if api_client:
                        agent_stats[agent.agent_id] = api_client.get_thinking_session_statistics()
        
        return {
            "system_statistics": system_stats,
            "agent_statistics": agent_stats,
            "total_agents_with_thinking_capability": len(agent_stats),
            "agents_with_active_sessions": len([
                stats for stats in agent_stats.values()
                if stats.get("active_sessions", 0) > 0
            ])
        }
    
    def cancel_thinking_session(self, session_id: str, reason: str = "Cancelled by system") -> bool:
        """
        Cancel a thinking session
        
        Args:
            session_id: ID of the session to cancel
            reason: Reason for cancellation
            
        Returns:
            True if session was cancelled successfully, False otherwise
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        return coordinator.cancel_session(session_id, reason)
    
    def cleanup_old_thinking_sessions(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up old completed thinking sessions
        
        Args:
            max_age_seconds: Maximum age of sessions to keep (default: 1 hour)
            
        Returns:
            Number of sessions cleaned up
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        return coordinator.cleanup_completed_sessions(max_age_seconds)
    
    def _initialize_monitoring_and_alerting(self) -> None:
        """Initialize monitoring and alerting systems"""
        try:
            # Initialize monitoring system
            from utils.per_core_agent_monitoring import MonitoringThresholds
            
            # Create custom thresholds based on system configuration
            thresholds = MonitoringThresholds(
                agent_success_rate_warning=0.8,  # 80% success rate warning
                agent_success_rate_critical=0.6,  # 60% success rate critical
                agent_response_time_warning_ms=3000,  # 3 seconds warning
                agent_response_time_critical_ms=8000,  # 8 seconds critical
                agent_inactivity_warning_seconds=180,  # 3 minutes warning
                agent_inactivity_critical_seconds=300,  # 5 minutes critical
                openrouter_cost_warning_usd=5.0,  # $5/hour warning
                openrouter_cost_critical_usd=15.0,  # $15/hour critical
            )
            
            self._monitor = initialize_monitoring(thresholds, collection_interval_seconds=30)
            
            # Register all agents with the monitor
            with self._agents_lock:
                for agent in self.agents.values():
                    self._monitor.register_agent(agent.agent_id, agent.role.value, agent.core_id)
            
            # Initialize alerting system
            self._alerting_system = get_alerting_system()
            
            # Configure console alerting by default
            self._alerting_system.channels['console'] = type('AlertChannelConfig', (), {
                'channel': type('AlertChannel', (), {'CONSOLE': 'console'})(),
                'enabled': True,
                'min_severity': type('AlertSeverity', (), {'WARNING': 'warning'})(),
                'rate_limit_minutes': 1,
                'config': {}
            })()
            
            # Set up alert callback to integrate with monitoring
            def alert_callback(alert):
                logger.warning(f"Alert generated: {alert.title} - {alert.description}")
                # Could integrate with external monitoring systems here
            
            self._monitor.add_alert_callback(alert_callback)
            
            # Start alerting service
            self._alerting_system.start_delivery_service()
            
            logger.info("Monitoring and alerting systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring and alerting: {e}")
            # Continue without monitoring rather than failing completely
    
    def _shutdown_monitoring_and_alerting(self) -> None:
        """Shutdown monitoring and alerting systems"""
        try:
            if self._monitor:
                self._monitor.stop_monitoring()
                self._monitor = None
            
            if self._alerting_system:
                self._alerting_system.stop_delivery_service()
                self._alerting_system = None
            
            logger.info("Monitoring and alerting systems shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down monitoring and alerting: {e}")
    
    def get_comprehensive_monitoring_data(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data including metrics, alerts, and dashboard data
        
        Returns:
            Complete monitoring data structure
        """
        if not self._monitor:
            return {"error": "Monitoring system not initialized"}
        
        try:
            from utils.monitoring_dashboard import MonitoringDashboard
            
            dashboard = MonitoringDashboard(self._monitor)
            
            return {
                "system_metrics": self._monitor.get_system_metrics(),
                "agent_metrics": self._monitor.get_all_agent_metrics(),
                "active_alerts": self._monitor.get_active_alerts(),
                "alert_history": self._monitor.get_alert_history(50),
                "dashboard_data": self._monitor.get_dashboard_data(),
                "system_overview_dashboard": dashboard.create_system_overview_dashboard().to_dict(),
                "agent_performance_dashboard": dashboard.create_agent_performance_dashboard().to_dict(),
                "openrouter_dashboard": dashboard.create_openrouter_dashboard().to_dict(),
                "alerts_dashboard": dashboard.create_alerts_dashboard().to_dict(),
                "alerting_statistics": self._alerting_system.get_delivery_statistics() if self._alerting_system else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive monitoring data: {e}")
            return {"error": str(e)}
    
    def update_agent_monitoring_metrics(self, agent_id: str, api_call_success: bool, 
                                      response_time_ms: float, openrouter_call: bool = False,
                                      cost_usd: float = 0.0, tokens_used: int = 0) -> None:
        """
        Update monitoring metrics for an agent API call
        
        Args:
            agent_id: Agent identifier
            api_call_success: Whether the API call was successful
            response_time_ms: Response time in milliseconds
            openrouter_call: Whether this was an OpenRouter call
            cost_usd: Cost of the call in USD
            tokens_used: Number of tokens used
        """
        if self._monitor:
            self._monitor.record_api_call(
                agent_id=agent_id,
                success=api_call_success,
                response_time_ms=response_time_ms,
                openrouter_call=openrouter_call,
                cost_usd=cost_usd,
                tokens_used=tokens_used
            )
    
    def update_agent_thinking_session_metrics(self, agent_id: str, success: bool, 
                                            duration_ms: float) -> None:
        """
        Update monitoring metrics for an agent thinking session
        
        Args:
            agent_id: Agent identifier
            success: Whether the thinking session was successful
            duration_ms: Duration in milliseconds
        """
        if self._monitor:
            self._monitor.record_thinking_session(agent_id, success, duration_ms)
    
    def configure_email_alerts(self, smtp_host: str, smtp_port: int, username: str,
                             password: str, from_email: str, to_emails: List[str]) -> bool:
        """
        Configure email alerting
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
            
        Returns:
            True if configuration was successful
        """
        if not self._alerting_system:
            logger.error("Alerting system not initialized")
            return False
        
        try:
            from utils.per_core_agent_monitoring import AlertSeverity
            
            self._alerting_system.configure_email_channel(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                username=username,
                password=password,
                from_email=from_email,
                to_emails=to_emails,
                min_severity=AlertSeverity.WARNING
            )
            
            logger.info(f"Email alerting configured for {len(to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure email alerts: {e}")
            return False
    
    def configure_webhook_alerts(self, name: str, webhook_url: str, 
                               headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Configure webhook alerting
        
        Args:
            name: Webhook configuration name
            webhook_url: Webhook URL
            headers: Optional HTTP headers
            
        Returns:
            True if configuration was successful
        """
        if not self._alerting_system:
            logger.error("Alerting system not initialized")
            return False
        
        try:
            from utils.per_core_agent_monitoring import AlertSeverity
            
            self._alerting_system.configure_webhook_channel(
                name=name,
                webhook_url=webhook_url,
                headers=headers,
                min_severity=AlertSeverity.WARNING
            )
            
            logger.info(f"Webhook alerting configured: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure webhook alerts: {e}")
            return False
    
    def get_monitoring_dashboard_data(self, dashboard_type: str = "system_overview") -> Dict[str, Any]:
        """
        Get dashboard data for monitoring interfaces
        
        Args:
            dashboard_type: Type of dashboard (system_overview, agent_performance, openrouter, alerts)
            
        Returns:
            Dashboard data structure
        """
        if not self._monitor:
            return {"error": "Monitoring system not initialized"}
        
        try:
            from utils.monitoring_dashboard import MonitoringDashboard
            
            dashboard = MonitoringDashboard(self._monitor)
            
            if dashboard_type == "system_overview":
                return dashboard.create_system_overview_dashboard().to_dict()
            elif dashboard_type == "agent_performance":
                return dashboard.create_agent_performance_dashboard().to_dict()
            elif dashboard_type == "openrouter":
                return dashboard.create_openrouter_dashboard().to_dict()
            elif dashboard_type == "alerts":
                return dashboard.create_alerts_dashboard().to_dict()
            else:
                return {"error": f"Unknown dashboard type: {dashboard_type}"}
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}


# Global singleton instance
_per_core_agent_manager_instance = None
_per_core_agent_manager_lock = threading.Lock()


def get_per_core_agent_manager(
    openrouter_api_key: Optional[str] = None,
    max_agents: Optional[int] = None
) -> PerCoreAgentManager:
    """
    Get the global per-core agent manager instance (singleton pattern)
    
    Args:
        openrouter_api_key: OpenRouter API key (only used on first creation)
        max_agents: Maximum number of agents (only used on first creation)
        
    Returns:
        PerCoreAgentManager instance
    """
    global _per_core_agent_manager_instance
    if _per_core_agent_manager_instance is None:
        with _per_core_agent_manager_lock:
            if _per_core_agent_manager_instance is None:
                _per_core_agent_manager_instance = PerCoreAgentManager(
                    openrouter_api_key=openrouter_api_key,
                    max_agents=max_agents
                )
        return _per_core_agent_manager_instance
    
    def recover_agent_from_failure(self, failed_agent_id: str) -> bool:
        """
        Attempt to recover an agent from failure using persistent memory
        
        Args:
            failed_agent_id: ID of the failed agent
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Attempting to recover agent {failed_agent_id} from failure")
        
        with self._agents_lock:
            # Find the failed agent
            failed_agent = None
            failed_core_id = None
            
            for core_id, agent in self.agents.items():
                if agent.agent_id == failed_agent_id:
                    failed_agent = agent
                    failed_core_id = core_id
                    break
            
            if not failed_agent:
                logger.error(f"Failed agent {failed_agent_id} not found in agent registry")
                return False
            
            try:
                # Attempt context recovery using persistent memory
                recovery_success = failed_agent.recover_from_failure()
                
                if recovery_success:
                    # Update agent status to active
                    failed_agent.update_status(AgentStatus.ACTIVE)
                    
                    # Update status tracking
                    with self._status_lock:
                        if failed_agent_id in self.agent_statuses:
                            self.agent_statuses[failed_agent_id].status = AgentStatus.ACTIVE
                            self.agent_statuses[failed_agent_id].last_activity = time.time()
                    
                    # Reconfigure OpenRouter if needed
                    if self.openrouter_api_key:
                        self._configure_agent_openrouter(failed_agent)
                    
                    # Notify other agents about successful recovery
                    if self.communication_system:
                        self.communication_system.send_message(
                            from_agent="system",
                            to_agent="ALL",
                            message_type="alert",
                            content=f"Agent {failed_agent_id} has successfully recovered from failure",
                            priority=7
                        )
                    
                    logger.info(f"Successfully recovered agent {failed_agent_id} from failure")
                    return True
                else:
                    logger.warning(f"Failed to recover agent {failed_agent_id} using persistent memory")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during agent recovery for {failed_agent_id}: {e}")
                return False
    
    def create_system_wide_context_snapshot(self) -> Dict[str, str]:
        """
        Create context snapshots for all active agents
        
        Returns:
            Dictionary mapping agent IDs to snapshot IDs
        """
        snapshots = {}
        
        with self._agents_lock:
            for agent in self.agents.values():
                try:
                    snapshot_id = agent.create_context_snapshot()
                    if snapshot_id:
                        snapshots[agent.agent_id] = snapshot_id
                        logger.debug(f"Created system snapshot {snapshot_id} for agent {agent.agent_id}")
                except Exception as e:
                    logger.error(f"Failed to create snapshot for agent {agent.agent_id}: {e}")
        
        logger.info(f"Created system-wide context snapshots for {len(snapshots)} agents")
        return snapshots
    
    def _initialize_monitoring_and_alerting(self) -> None:
        """Initialize monitoring and alerting systems"""
        try:
            # Initialize monitoring system
            from utils.per_core_agent_monitoring import MonitoringThresholds
            
            # Create custom thresholds based on system configuration
            thresholds = MonitoringThresholds(
                agent_success_rate_warning=0.8,  # 80% success rate warning
                agent_success_rate_critical=0.6,  # 60% success rate critical
                agent_response_time_warning_ms=3000,  # 3 seconds warning
                agent_response_time_critical_ms=8000,  # 8 seconds critical
                agent_inactivity_warning_seconds=180,  # 3 minutes warning
                agent_inactivity_critical_seconds=300,  # 5 minutes critical
                openrouter_cost_warning_usd=5.0,  # $5/hour warning
                openrouter_cost_critical_usd=15.0,  # $15/hour critical
            )
            
            self._monitor = initialize_monitoring(thresholds, collection_interval_seconds=30)
            
            # Register all agents with the monitor
            with self._agents_lock:
                for agent in self.agents.values():
                    self._monitor.register_agent(agent.agent_id, agent.role.value, agent.core_id)
            
            # Initialize alerting system
            self._alerting_system = get_alerting_system()
            
            # Configure console alerting by default
            self._alerting_system.channels['console'] = type('AlertChannelConfig', (), {
                'channel': type('AlertChannel', (), {'CONSOLE': 'console'})(),
                'enabled': True,
                'min_severity': type('AlertSeverity', (), {'WARNING': 'warning'})(),
                'rate_limit_minutes': 1,
                'config': {}
            })()
            
            # Set up alert callback to integrate with monitoring
            def alert_callback(alert):
                logger.warning(f"Alert generated: {alert.title} - {alert.description}")
                # Could integrate with external monitoring systems here
            
            self._monitor.add_alert_callback(alert_callback)
            
            # Start alerting service
            self._alerting_system.start_delivery_service()
            
            logger.info("Monitoring and alerting systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring and alerting: {e}")
            # Continue without monitoring rather than failing completely
    
    def _shutdown_monitoring_and_alerting(self) -> None:
        """Shutdown monitoring and alerting systems"""
        try:
            if self._monitor:
                self._monitor.stop_monitoring()
                self._monitor = None
            
            if self._alerting_system:
                self._alerting_system.stop_delivery_service()
                self._alerting_system = None
            
            logger.info("Monitoring and alerting systems shut down")
            
        except Exception as e:
            logger.error(f"Error shutting down monitoring and alerting: {e}")
    
    def get_comprehensive_monitoring_data(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring data including metrics, alerts, and dashboard data
        
        Returns:
            Complete monitoring data structure
        """
        if not self._monitor:
            return {"error": "Monitoring system not initialized"}
        
        try:
            from utils.monitoring_dashboard import MonitoringDashboard
            
            dashboard = MonitoringDashboard(self._monitor)
            
            return {
                "system_metrics": self._monitor.get_system_metrics(),
                "agent_metrics": self._monitor.get_all_agent_metrics(),
                "active_alerts": self._monitor.get_active_alerts(),
                "alert_history": self._monitor.get_alert_history(50),
                "dashboard_data": self._monitor.get_dashboard_data(),
                "system_overview_dashboard": dashboard.create_system_overview_dashboard().to_dict(),
                "agent_performance_dashboard": dashboard.create_agent_performance_dashboard().to_dict(),
                "openrouter_dashboard": dashboard.create_openrouter_dashboard().to_dict(),
                "alerts_dashboard": dashboard.create_alerts_dashboard().to_dict(),
                "alerting_statistics": self._alerting_system.get_delivery_statistics() if self._alerting_system else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive monitoring data: {e}")
            return {"error": str(e)}
    
    def update_agent_monitoring_metrics(self, agent_id: str, api_call_success: bool, 
                                      response_time_ms: float, openrouter_call: bool = False,
                                      cost_usd: float = 0.0, tokens_used: int = 0) -> None:
        """
        Update monitoring metrics for an agent API call
        
        Args:
            agent_id: Agent identifier
            api_call_success: Whether the API call was successful
            response_time_ms: Response time in milliseconds
            openrouter_call: Whether this was an OpenRouter call
            cost_usd: Cost of the call in USD
            tokens_used: Number of tokens used
        """
        if self._monitor:
            self._monitor.record_api_call(
                agent_id=agent_id,
                success=api_call_success,
                response_time_ms=response_time_ms,
                openrouter_call=openrouter_call,
                cost_usd=cost_usd,
                tokens_used=tokens_used
            )
    
    def update_agent_thinking_session_metrics(self, agent_id: str, success: bool, 
                                            duration_ms: float) -> None:
        """
        Update monitoring metrics for an agent thinking session
        
        Args:
            agent_id: Agent identifier
            success: Whether the thinking session was successful
            duration_ms: Duration in milliseconds
        """
        if self._monitor:
            self._monitor.record_thinking_session(agent_id, success, duration_ms)
    
    def configure_email_alerts(self, smtp_host: str, smtp_port: int, username: str,
                             password: str, from_email: str, to_emails: List[str]) -> bool:
        """
        Configure email alerting
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
            
        Returns:
            True if configuration was successful
        """
        if not self._alerting_system:
            logger.error("Alerting system not initialized")
            return False
        
        try:
            from utils.per_core_agent_monitoring import AlertSeverity
            
            self._alerting_system.configure_email_channel(
                smtp_host=smtp_host,
                smtp_port=smtp_port,
                username=username,
                password=password,
                from_email=from_email,
                to_emails=to_emails,
                min_severity=AlertSeverity.WARNING
            )
            
            logger.info(f"Email alerting configured for {len(to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure email alerts: {e}")
            return False
    
    def configure_webhook_alerts(self, name: str, webhook_url: str, 
                               headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Configure webhook alerting
        
        Args:
            name: Webhook configuration name
            webhook_url: Webhook URL
            headers: Optional HTTP headers
            
        Returns:
            True if configuration was successful
        """
        if not self._alerting_system:
            logger.error("Alerting system not initialized")
            return False
        
        try:
            from utils.per_core_agent_monitoring import AlertSeverity
            
            self._alerting_system.configure_webhook_channel(
                name=name,
                webhook_url=webhook_url,
                headers=headers,
                min_severity=AlertSeverity.WARNING
            )
            
            logger.info(f"Webhook alerting configured: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure webhook alerts: {e}")
            return False
    
    def get_monitoring_dashboard_data(self, dashboard_type: str = "system_overview") -> Dict[str, Any]:
        """
        Get dashboard data for monitoring interfaces
        
        Args:
            dashboard_type: Type of dashboard (system_overview, agent_performance, openrouter, alerts)
            
        Returns:
            Dashboard data structure
        """
        if not self._monitor:
            return {"error": "Monitoring system not initialized"}
        
        try:
            from utils.monitoring_dashboard import MonitoringDashboard
            
            dashboard = MonitoringDashboard(self._monitor)
            
            if dashboard_type == "system_overview":
                return dashboard.create_system_overview_dashboard().to_dict()
            elif dashboard_type == "agent_performance":
                return dashboard.create_agent_performance_dashboard().to_dict()
            elif dashboard_type == "openrouter":
                return dashboard.create_openrouter_dashboard().to_dict()
            elif dashboard_type == "alerts":
                return dashboard.create_alerts_dashboard().to_dict()
            else:
                return {"error": f"Unknown dashboard type: {dashboard_type}"}
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def get_agent_memory_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics for all agents
        
        Returns:
            Dictionary with memory statistics for each agent
        """
        memory_stats = {}
        
        with self._agents_lock:
            for agent in self.agents.values():
                try:
                    persistent_memory = agent.get_persistent_memory()
                    if persistent_memory:
                        agent_stats = persistent_memory.get_memory_statistics()
                        memory_stats[agent.agent_id] = agent_stats
                except Exception as e:
                    logger.debug(f"Could not get memory statistics for agent {agent.agent_id}: {e}")
                    memory_stats[agent.agent_id] = {"error": str(e)}
        
        return memory_stats


# Global singleton instance
_per_core_agent_manager_instance = None
_per_core_agent_manager_lock = threading.Lock()


def get_per_core_agent_manager() -> Optional[PerCoreAgentManager]:
    """Get the global per-core agent manager instance (singleton pattern)"""
    global _per_core_agent_manager_instance
    return _per_core_agent_manager_instance


def initialize_per_core_agent_manager(openrouter_api_key: Optional[str] = None, max_agents: Optional[int] = None) -> PerCoreAgentManager:
    """Initialize the global per-core agent manager instance"""
    global _per_core_agent_manager_instance
    
    with _per_core_agent_manager_lock:
        if _per_core_agent_manager_instance is None:
            _per_core_agent_manager_instance = PerCoreAgentManager(
                openrouter_api_key=openrouter_api_key,
                max_agents=max_agents
            )
        
        return _per_core_agent_manager_instance


def shutdown_per_core_agent_manager() -> None:
    """Shutdown the global per-core agent manager instance"""
    global _per_core_agent_manager_instance
    
    with _per_core_agent_manager_lock:
        if _per_core_agent_manager_instance:
            _per_core_agent_manager_instance.shutdown_agents()
            _per_core_agent_manager_instance = None