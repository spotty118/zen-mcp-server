"""
Agent Thinking Session Management

This module implements thinking session coordination and management for per-core agents.
It provides data models and coordination capabilities for synchronized thinking sessions
between multiple agents with timeout handling and result aggregation.

Key Features:
- AgentThinkingSession data model for tracking individual thinking sessions
- Synchronized thinking session capabilities between multiple agents
- Session timeout handling and result aggregation
- Thread-safe session management and coordination
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union

from utils.agent_core import Agent, AgentRole, AgentStatus
from utils.agent_api_client import AgentAPICall

logger = logging.getLogger(__name__)


class ThinkingSessionStatus(Enum):
    """Status of a thinking session"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ThinkingSessionType(Enum):
    """Type of thinking session"""
    INDIVIDUAL = "individual"  # Single agent thinking
    SYNCHRONIZED = "synchronized"  # Multiple agents thinking together
    COLLABORATIVE = "collaborative"  # Agents building on each other's thoughts
    CONSENSUS = "consensus"  # Agents working toward agreement


@dataclass
class AgentThinkingSession:
    """
    Represents a thinking session executed by an agent through OpenRouter
    
    This data model tracks individual thinking sessions with comprehensive
    metadata for monitoring, debugging, and result aggregation.
    """
    session_id: str
    agent_id: str
    core_id: int
    thinking_prompt: str
    model_used: str
    thinking_mode: str
    started_at: float
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    openrouter_usage: Dict[str, int] = field(default_factory=dict)
    
    # Session metadata
    status: ThinkingSessionStatus = ThinkingSessionStatus.PENDING
    session_type: ThinkingSessionType = ThinkingSessionType.INDIVIDUAL
    timeout_seconds: float = 300.0  # 5 minutes default timeout
    priority: int = 5  # 1-10 priority scale
    
    # Coordination metadata
    parent_session_id: Optional[str] = None  # For synchronized sessions
    child_session_ids: List[str] = field(default_factory=list)
    related_agent_ids: List[str] = field(default_factory=list)
    
    # Performance metadata
    api_call_id: Optional[str] = None
    retry_count: int = 0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    
    def get_duration(self) -> float:
        """Get the duration of the thinking session in seconds"""
        if self.completed_at is None:
            return time.time() - self.started_at
        return self.completed_at - self.started_at
    
    def is_completed(self) -> bool:
        """Check if the thinking session is completed"""
        return self.status in [
            ThinkingSessionStatus.COMPLETED,
            ThinkingSessionStatus.FAILED,
            ThinkingSessionStatus.TIMEOUT,
            ThinkingSessionStatus.CANCELLED
        ]
    
    def is_active(self) -> bool:
        """Check if the thinking session is currently active"""
        return self.status == ThinkingSessionStatus.ACTIVE
    
    def is_timed_out(self) -> bool:
        """Check if the thinking session has timed out"""
        if self.is_completed():
            return self.status == ThinkingSessionStatus.TIMEOUT
        
        # Check if session has exceeded timeout
        return (time.time() - self.started_at) > self.timeout_seconds
    
    def mark_completed(self, result: str, tokens_used: int = 0, cost_estimate: float = 0.0) -> None:
        """Mark the session as completed with results"""
        self.completed_at = time.time()
        self.result = result
        self.status = ThinkingSessionStatus.COMPLETED
        self.tokens_used = tokens_used
        self.cost_estimate = cost_estimate
    
    def mark_failed(self, error: str) -> None:
        """Mark the session as failed with error details"""
        self.completed_at = time.time()
        self.error = error
        self.status = ThinkingSessionStatus.FAILED
    
    def mark_timeout(self) -> None:
        """Mark the session as timed out"""
        self.completed_at = time.time()
        self.status = ThinkingSessionStatus.TIMEOUT
        self.error = f"Session timed out after {self.timeout_seconds} seconds"
    
    def mark_cancelled(self, reason: str = "Cancelled by system") -> None:
        """Mark the session as cancelled"""
        self.completed_at = time.time()
        self.status = ThinkingSessionStatus.CANCELLED
        self.error = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the thinking session to a dictionary"""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "core_id": self.core_id,
            "thinking_prompt": self.thinking_prompt[:200] + "..." if len(self.thinking_prompt) > 200 else self.thinking_prompt,
            "model_used": self.model_used,
            "thinking_mode": self.thinking_mode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.get_duration(),
            "status": self.status.value,
            "session_type": self.session_type.value,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "parent_session_id": self.parent_session_id,
            "child_session_ids": self.child_session_ids,
            "related_agent_ids": self.related_agent_ids,
            "api_call_id": self.api_call_id,
            "retry_count": self.retry_count,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "has_result": self.result is not None,
            "has_error": self.error is not None,
            "is_completed": self.is_completed(),
            "is_active": self.is_active(),
            "is_timed_out": self.is_timed_out(),
            "openrouter_usage": self.openrouter_usage
        }


@dataclass
class SynchronizedThinkingRequest:
    """Request for synchronized thinking session across multiple agents"""
    request_id: str
    thinking_prompt: str
    participating_agent_ids: List[str]
    session_type: ThinkingSessionType = ThinkingSessionType.SYNCHRONIZED
    timeout_seconds: float = 300.0
    priority: int = 5
    
    # Coordination settings
    require_all_agents: bool = True  # Whether all agents must participate
    min_agents_required: int = 1  # Minimum agents needed for session
    max_wait_time: float = 60.0  # Max time to wait for agents to join
    
    # Result aggregation settings
    aggregation_strategy: str = "consensus"  # consensus, majority, weighted, all
    result_format: str = "structured"  # structured, narrative, summary
    
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "thinking_prompt": self.thinking_prompt[:200] + "..." if len(self.thinking_prompt) > 200 else self.thinking_prompt,
            "participating_agent_ids": self.participating_agent_ids,
            "session_type": self.session_type.value,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "require_all_agents": self.require_all_agents,
            "min_agents_required": self.min_agents_required,
            "max_wait_time": self.max_wait_time,
            "aggregation_strategy": self.aggregation_strategy,
            "result_format": self.result_format,
            "created_at": self.created_at
        }


class ThinkingSessionCoordinator:
    """
    Coordinates thinking sessions between multiple agents with synchronization,
    timeout handling, and result aggregation capabilities.
    """
    
    def __init__(self):
        """Initialize the thinking session coordinator"""
        # Session tracking
        self.active_sessions: Dict[str, AgentThinkingSession] = {}
        self.completed_sessions: List[AgentThinkingSession] = []
        self.synchronized_requests: Dict[str, SynchronizedThinkingRequest] = {}
        
        # Thread safety
        self._sessions_lock = threading.RLock()
        self._requests_lock = threading.RLock()
        
        # Timeout monitoring
        self._shutdown = False
        self._timeout_monitor_thread: Optional[threading.Thread] = None
        self._start_timeout_monitoring()
        
        # Session callbacks
        self._session_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("ThinkingSessionCoordinator initialized")
    
    def create_individual_session(
        self,
        agent: Agent,
        thinking_prompt: str,
        model_used: str = "auto",
        thinking_mode: str = "high",
        timeout_seconds: float = 300.0,
        priority: int = 5
    ) -> AgentThinkingSession:
        """
        Create an individual thinking session for a single agent
        
        Args:
            agent: The agent that will perform the thinking
            thinking_prompt: The prompt for the thinking session
            model_used: The model to use for thinking
            thinking_mode: The thinking mode (high, standard, low)
            timeout_seconds: Session timeout in seconds
            priority: Session priority (1-10)
            
        Returns:
            AgentThinkingSession instance
        """
        session_id = str(uuid.uuid4())
        
        session = AgentThinkingSession(
            session_id=session_id,
            agent_id=agent.agent_id,
            core_id=agent.core_id,
            thinking_prompt=thinking_prompt,
            model_used=model_used,
            thinking_mode=thinking_mode,
            started_at=time.time(),
            status=ThinkingSessionStatus.PENDING,
            session_type=ThinkingSessionType.INDIVIDUAL,
            timeout_seconds=timeout_seconds,
            priority=priority
        )
        
        with self._sessions_lock:
            self.active_sessions[session_id] = session
        
        logger.info(f"Created individual thinking session {session_id} for agent {agent.agent_id}")
        return session
    
    def create_synchronized_session(
        self,
        request: SynchronizedThinkingRequest
    ) -> Dict[str, AgentThinkingSession]:
        """
        Create synchronized thinking sessions for multiple agents
        
        Args:
            request: Synchronized thinking request with configuration
            
        Returns:
            Dictionary mapping agent_id to AgentThinkingSession
        """
        parent_session_id = str(uuid.uuid4())
        created_sessions = {}
        
        with self._requests_lock:
            self.synchronized_requests[request.request_id] = request
        
        # Create individual sessions for each participating agent
        for agent_id in request.participating_agent_ids:
            session_id = str(uuid.uuid4())
            
            session = AgentThinkingSession(
                session_id=session_id,
                agent_id=agent_id,
                core_id=-1,  # Will be updated when agent is found
                thinking_prompt=request.thinking_prompt,
                model_used="auto",  # Will be determined by agent's configuration
                thinking_mode="high",
                started_at=time.time(),
                status=ThinkingSessionStatus.PENDING,
                session_type=request.session_type,
                timeout_seconds=request.timeout_seconds,
                priority=request.priority,
                parent_session_id=parent_session_id,
                related_agent_ids=request.participating_agent_ids.copy()
            )
            
            created_sessions[agent_id] = session
        
        # Link sessions together
        session_ids = [s.session_id for s in created_sessions.values()]
        for session in created_sessions.values():
            session.child_session_ids = [sid for sid in session_ids if sid != session.session_id]
        
        # Add to active sessions
        with self._sessions_lock:
            for session in created_sessions.values():
                self.active_sessions[session.session_id] = session
        
        logger.info(f"Created synchronized thinking sessions for request {request.request_id} "
                   f"with {len(created_sessions)} agents")
        
        return created_sessions
    
    def start_session(self, session_id: str) -> bool:
        """
        Start a thinking session
        
        Args:
            session_id: ID of the session to start
            
        Returns:
            True if session was started successfully, False otherwise
        """
        with self._sessions_lock:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found in active sessions")
                return False
            
            session = self.active_sessions[session_id]
            
            if session.status != ThinkingSessionStatus.PENDING:
                logger.warning(f"Session {session_id} is not in pending status: {session.status}")
                return False
            
            session.status = ThinkingSessionStatus.ACTIVE
            session.started_at = time.time()  # Update start time
            
            logger.info(f"Started thinking session {session_id} for agent {session.agent_id}")
            return True
    
    def complete_session(
        self,
        session_id: str,
        result: str,
        tokens_used: int = 0,
        cost_estimate: float = 0.0,
        openrouter_usage: Optional[Dict[str, int]] = None
    ) -> bool:
        """
        Complete a thinking session with results
        
        Args:
            session_id: ID of the session to complete
            result: The thinking result
            tokens_used: Number of tokens used
            cost_estimate: Estimated cost of the session
            openrouter_usage: OpenRouter usage statistics
            
        Returns:
            True if session was completed successfully, False otherwise
        """
        with self._sessions_lock:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found in active sessions")
                return False
            
            session = self.active_sessions[session_id]
            session.mark_completed(result, tokens_used, cost_estimate)
            
            if openrouter_usage:
                session.openrouter_usage.update(openrouter_usage)
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            logger.info(f"Completed thinking session {session_id} for agent {session.agent_id}")
            
            # Trigger callbacks
            self._trigger_session_callbacks(session_id, "completed", session)
            
            # Check if this completes a synchronized session
            if session.parent_session_id:
                self._check_synchronized_session_completion(session.parent_session_id)
            
            return True
    
    def fail_session(self, session_id: str, error: str) -> bool:
        """
        Mark a thinking session as failed
        
        Args:
            session_id: ID of the session to fail
            error: Error message
            
        Returns:
            True if session was marked as failed successfully, False otherwise
        """
        with self._sessions_lock:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found in active sessions")
                return False
            
            session = self.active_sessions[session_id]
            session.mark_failed(error)
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            logger.warning(f"Failed thinking session {session_id} for agent {session.agent_id}: {error}")
            
            # Trigger callbacks
            self._trigger_session_callbacks(session_id, "failed", session)
            
            # Check if this affects a synchronized session
            if session.parent_session_id:
                self._check_synchronized_session_completion(session.parent_session_id)
            
            return True
    
    def cancel_session(self, session_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Cancel a thinking session
        
        Args:
            session_id: ID of the session to cancel
            reason: Reason for cancellation
            
        Returns:
            True if session was cancelled successfully, False otherwise
        """
        with self._sessions_lock:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found in active sessions")
                return False
            
            session = self.active_sessions[session_id]
            session.mark_cancelled(reason)
            
            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
            
            logger.info(f"Cancelled thinking session {session_id} for agent {session.agent_id}: {reason}")
            
            # Trigger callbacks
            self._trigger_session_callbacks(session_id, "cancelled", session)
            
            return True
    
    def get_session(self, session_id: str) -> Optional[AgentThinkingSession]:
        """
        Get a thinking session by ID
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            AgentThinkingSession if found, None otherwise
        """
        with self._sessions_lock:
            # Check active sessions first
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
            
            # Check completed sessions
            for session in self.completed_sessions:
                if session.session_id == session_id:
                    return session
            
            return None
    
    def get_agent_sessions(self, agent_id: str, include_completed: bool = False) -> List[AgentThinkingSession]:
        """
        Get all thinking sessions for a specific agent
        
        Args:
            agent_id: ID of the agent
            include_completed: Whether to include completed sessions
            
        Returns:
            List of AgentThinkingSession instances
        """
        sessions = []
        
        with self._sessions_lock:
            # Get active sessions
            for session in self.active_sessions.values():
                if session.agent_id == agent_id:
                    sessions.append(session)
            
            # Get completed sessions if requested
            if include_completed:
                for session in self.completed_sessions:
                    if session.agent_id == agent_id:
                        sessions.append(session)
        
        return sessions
    
    def get_synchronized_sessions(self, parent_session_id: str) -> List[AgentThinkingSession]:
        """
        Get all sessions that are part of a synchronized thinking session
        
        Args:
            parent_session_id: ID of the parent synchronized session
            
        Returns:
            List of AgentThinkingSession instances
        """
        sessions = []
        
        with self._sessions_lock:
            # Check active sessions
            for session in self.active_sessions.values():
                if session.parent_session_id == parent_session_id:
                    sessions.append(session)
            
            # Check completed sessions
            for session in self.completed_sessions:
                if session.parent_session_id == parent_session_id:
                    sessions.append(session)
        
        return sessions
    
    def aggregate_synchronized_results(
        self,
        parent_session_id: str,
        aggregation_strategy: str = "consensus"
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate results from synchronized thinking sessions
        
        Args:
            parent_session_id: ID of the parent synchronized session
            aggregation_strategy: Strategy for aggregating results
            
        Returns:
            Aggregated results dictionary or None if not ready
        """
        sessions = self.get_synchronized_sessions(parent_session_id)
        
        if not sessions:
            logger.warning(f"No sessions found for parent session {parent_session_id}")
            return None
        
        # Check if all sessions are completed
        completed_sessions = [s for s in sessions if s.is_completed()]
        successful_sessions = [s for s in completed_sessions if s.status == ThinkingSessionStatus.COMPLETED]
        
        if not successful_sessions:
            logger.warning(f"No successful sessions found for parent session {parent_session_id}")
            return None
        
        # Aggregate results based on strategy
        if aggregation_strategy == "consensus":
            return self._aggregate_consensus(successful_sessions)
        elif aggregation_strategy == "majority":
            return self._aggregate_majority(successful_sessions)
        elif aggregation_strategy == "weighted":
            return self._aggregate_weighted(successful_sessions)
        elif aggregation_strategy == "all":
            return self._aggregate_all(successful_sessions)
        else:
            logger.warning(f"Unknown aggregation strategy: {aggregation_strategy}")
            return self._aggregate_all(successful_sessions)
    
    def _aggregate_consensus(self, sessions: List[AgentThinkingSession]) -> Dict[str, Any]:
        """Aggregate results using consensus strategy"""
        return {
            "strategy": "consensus",
            "total_sessions": len(sessions),
            "results": [
                {
                    "agent_id": session.agent_id,
                    "result": session.result,
                    "confidence": 0.8,  # Could be enhanced with actual confidence scoring
                    "tokens_used": session.tokens_used,
                    "duration": session.get_duration()
                }
                for session in sessions
            ],
            "consensus_result": self._find_consensus_result(sessions),
            "aggregated_at": time.time()
        }
    
    def _aggregate_majority(self, sessions: List[AgentThinkingSession]) -> Dict[str, Any]:
        """Aggregate results using majority strategy"""
        return {
            "strategy": "majority",
            "total_sessions": len(sessions),
            "results": [
                {
                    "agent_id": session.agent_id,
                    "result": session.result,
                    "tokens_used": session.tokens_used,
                    "duration": session.get_duration()
                }
                for session in sessions
            ],
            "majority_result": self._find_majority_result(sessions),
            "aggregated_at": time.time()
        }
    
    def _aggregate_weighted(self, sessions: List[AgentThinkingSession]) -> Dict[str, Any]:
        """Aggregate results using weighted strategy based on agent roles"""
        return {
            "strategy": "weighted",
            "total_sessions": len(sessions),
            "results": [
                {
                    "agent_id": session.agent_id,
                    "result": session.result,
                    "weight": self._get_agent_weight(session.agent_id),
                    "tokens_used": session.tokens_used,
                    "duration": session.get_duration()
                }
                for session in sessions
            ],
            "weighted_result": self._find_weighted_result(sessions),
            "aggregated_at": time.time()
        }
    
    def _aggregate_all(self, sessions: List[AgentThinkingSession]) -> Dict[str, Any]:
        """Aggregate all results without filtering"""
        return {
            "strategy": "all",
            "total_sessions": len(sessions),
            "results": [
                {
                    "agent_id": session.agent_id,
                    "result": session.result,
                    "tokens_used": session.tokens_used,
                    "duration": session.get_duration(),
                    "status": session.status.value
                }
                for session in sessions
            ],
            "combined_result": "\n\n".join([
                f"Agent {session.agent_id}: {session.result}"
                for session in sessions if session.result
            ]),
            "aggregated_at": time.time()
        }
    
    def _find_consensus_result(self, sessions: List[AgentThinkingSession]) -> str:
        """Find consensus result from multiple sessions"""
        # Simple implementation - could be enhanced with NLP similarity analysis
        results = [session.result for session in sessions if session.result]
        
        if not results:
            return "No results available for consensus"
        
        if len(results) == 1:
            return results[0]
        
        # For now, return a summary indicating multiple perspectives
        return f"Consensus from {len(results)} agents: Multiple perspectives provided. " \
               f"Primary result: {results[0][:200]}..."
    
    def _find_majority_result(self, sessions: List[AgentThinkingSession]) -> str:
        """Find majority result from multiple sessions"""
        results = [session.result for session in sessions if session.result]
        
        if not results:
            return "No results available for majority"
        
        # Simple majority - could be enhanced with similarity analysis
        return results[0] if results else "No majority result found"
    
    def _find_weighted_result(self, sessions: List[AgentThinkingSession]) -> str:
        """Find weighted result based on agent roles"""
        if not sessions:
            return "No results available for weighting"
        
        # Sort by agent weight (higher weight = more influence)
        weighted_sessions = sorted(
            sessions,
            key=lambda s: self._get_agent_weight(s.agent_id),
            reverse=True
        )
        
        # Return result from highest weighted agent
        return weighted_sessions[0].result if weighted_sessions[0].result else "No weighted result available"
    
    def _get_agent_weight(self, agent_id: str) -> float:
        """Get weight for an agent based on its role"""
        # This would be enhanced to look up actual agent roles
        # For now, return default weight
        return 1.0
    
    def _check_synchronized_session_completion(self, parent_session_id: str) -> None:
        """Check if a synchronized session is complete and trigger aggregation"""
        sessions = self.get_synchronized_sessions(parent_session_id)
        
        if not sessions:
            return
        
        # Check if all sessions are completed
        all_completed = all(session.is_completed() for session in sessions)
        
        if all_completed:
            logger.info(f"All sessions completed for synchronized session {parent_session_id}")
            
            # Find the original request
            with self._requests_lock:
                request = None
                for req in self.synchronized_requests.values():
                    if any(s.parent_session_id == parent_session_id for s in sessions):
                        request = req
                        break
                
                if request:
                    # Aggregate results
                    aggregated = self.aggregate_synchronized_results(
                        parent_session_id,
                        request.aggregation_strategy
                    )
                    
                    if aggregated:
                        logger.info(f"Aggregated results for synchronized session {parent_session_id}")
                        # Trigger callbacks for synchronized completion
                        self._trigger_session_callbacks(parent_session_id, "synchronized_completed", aggregated)
    
    def add_session_callback(
        self,
        session_id: str,
        callback: Callable[[str, str, Any], None],
        event_types: List[str] = None
    ) -> None:
        """
        Add a callback for session events
        
        Args:
            session_id: ID of the session to monitor
            callback: Callback function to call on events
            event_types: List of event types to monitor (default: all)
        """
        if event_types is None:
            event_types = ["completed", "failed", "cancelled", "timeout", "synchronized_completed"]
        
        if session_id not in self._session_callbacks:
            self._session_callbacks[session_id] = []
        
        self._session_callbacks[session_id].append((callback, event_types))
    
    def _trigger_session_callbacks(self, session_id: str, event_type: str, data: Any) -> None:
        """Trigger callbacks for a session event"""
        if session_id in self._session_callbacks:
            for callback, event_types in self._session_callbacks[session_id]:
                if event_type in event_types:
                    try:
                        callback(session_id, event_type, data)
                    except Exception as e:
                        logger.error(f"Error in session callback for {session_id}: {e}")
    
    def _start_timeout_monitoring(self) -> None:
        """Start background timeout monitoring thread"""
        if self._timeout_monitor_thread and self._timeout_monitor_thread.is_alive():
            return
        
        self._timeout_monitor_thread = threading.Thread(
            target=self._timeout_monitor_loop,
            daemon=True,
            name="ThinkingSessionTimeoutMonitor"
        )
        self._timeout_monitor_thread.start()
        logger.info("Started thinking session timeout monitoring")
    
    def _timeout_monitor_loop(self) -> None:
        """Background timeout monitoring loop"""
        while not self._shutdown:
            try:
                self._check_session_timeouts()
                time.sleep(30)  # Check timeouts every 30 seconds
            except Exception as e:
                logger.error(f"Error in timeout monitoring loop: {e}")
                time.sleep(60)  # Longer pause on error
    
    def _check_session_timeouts(self) -> None:
        """Check for timed out sessions and handle them"""
        timed_out_sessions = []
        
        with self._sessions_lock:
            for session_id, session in self.active_sessions.items():
                if session.is_timed_out():
                    timed_out_sessions.append(session_id)
        
        # Handle timed out sessions
        for session_id in timed_out_sessions:
            logger.warning(f"Session {session_id} has timed out")
            
            with self._sessions_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    session.mark_timeout()
                    
                    # Move to completed sessions
                    self.completed_sessions.append(session)
                    del self.active_sessions[session_id]
                    
                    # Trigger callbacks
                    self._trigger_session_callbacks(session_id, "timeout", session)
                    
                    # Check if this affects a synchronized session
                    if session.parent_session_id:
                        self._check_synchronized_session_completion(session.parent_session_id)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        with self._sessions_lock:
            active_count = len(self.active_sessions)
            completed_count = len(self.completed_sessions)
            
            # Count by status
            status_counts = {}
            for session in self.active_sessions.values():
                status = session.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            for session in self.completed_sessions:
                status = session.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            type_counts = {}
            for session in list(self.active_sessions.values()) + self.completed_sessions:
                session_type = session.session_type.value
                type_counts[session_type] = type_counts.get(session_type, 0) + 1
            
            # Calculate average duration for completed sessions
            completed_sessions = [s for s in self.completed_sessions if s.is_completed()]
            avg_duration = (
                sum(s.get_duration() for s in completed_sessions) / len(completed_sessions)
                if completed_sessions else 0.0
            )
            
            # Calculate success rate
            successful_sessions = [
                s for s in completed_sessions 
                if s.status == ThinkingSessionStatus.COMPLETED
            ]
            success_rate = (
                len(successful_sessions) / len(completed_sessions)
                if completed_sessions else 0.0
            )
            
            return {
                "active_sessions": active_count,
                "completed_sessions": completed_count,
                "total_sessions": active_count + completed_count,
                "status_counts": status_counts,
                "type_counts": type_counts,
                "average_duration": avg_duration,
                "success_rate": success_rate,
                "synchronized_requests": len(self.synchronized_requests),
                "total_tokens_used": sum(s.tokens_used for s in completed_sessions),
                "total_cost_estimate": sum(s.cost_estimate for s in completed_sessions)
            }
    
    def cleanup_completed_sessions(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up old completed sessions to prevent memory leaks
        
        Args:
            max_age_seconds: Maximum age of sessions to keep (default: 1 hour)
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        cleaned_count = 0
        
        with self._sessions_lock:
            # Filter out old completed sessions
            old_sessions = [
                session for session in self.completed_sessions
                if session.completed_at and (current_time - session.completed_at) > max_age_seconds
            ]
            
            for session in old_sessions:
                self.completed_sessions.remove(session)
                cleaned_count += 1
                
                # Clean up callbacks for old sessions
                if session.session_id in self._session_callbacks:
                    del self._session_callbacks[session.session_id]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old completed sessions")
        
        return cleaned_count
    
    def shutdown(self) -> None:
        """Shutdown the thinking session coordinator"""
        logger.info("Shutting down thinking session coordinator")
        
        # Stop timeout monitoring
        self._shutdown = True
        if self._timeout_monitor_thread and self._timeout_monitor_thread.is_alive():
            self._timeout_monitor_thread.join(timeout=5)
        
        # Cancel all active sessions
        with self._sessions_lock:
            active_session_ids = list(self.active_sessions.keys())
            for session_id in active_session_ids:
                self.cancel_session(session_id, "System shutdown")
        
        logger.info("Thinking session coordinator shutdown completed")


# Global singleton instance
_thinking_session_coordinator_instance = None
_thinking_session_coordinator_lock = threading.Lock()


def get_thinking_session_coordinator() -> ThinkingSessionCoordinator:
    """
    Get the global thinking session coordinator instance (singleton pattern)
    
    Returns:
        ThinkingSessionCoordinator instance
    """
    global _thinking_session_coordinator_instance
    if _thinking_session_coordinator_instance is None:
        with _thinking_session_coordinator_lock:
            if _thinking_session_coordinator_instance is None:
                _thinking_session_coordinator_instance = ThinkingSessionCoordinator()
    return _thinking_session_coordinator_instance