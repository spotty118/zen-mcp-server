"""
Agent Communication System

This module manages communication between agents (CPU cores), including message routing,
team formation, and collaborative coordination. It extends the existing core context
storage to support agent-based communication patterns.

Key Features:
- Message routing between agents
- Team formation and management
- Agent discovery and coordination
- Communication threading and history
- Load balancing across available agents
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable

from utils.agent_core import Agent, AgentMessage, AgentRole, AgentStatus, AgentThought
from utils.core_context_storage import get_core_context_storage
from utils.agent_api_client import AgentAPIClient

logger = logging.getLogger(__name__)


@dataclass
class AgentTeam:
    """Represents a team of agents working together"""
    team_id: str
    team_name: str
    leader_agent: Optional[str]  # Agent ID of team leader
    members: Set[str] = field(default_factory=set)  # Agent IDs
    purpose: str = ""
    created_at: float = field(default_factory=time.time)
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "leader_agent": self.leader_agent,
            "members": list(self.members),
            "purpose": self.purpose,
            "created_at": self.created_at,
            "active": self.active
        }


@dataclass
class SynchronizedThinkingSession:
    """Represents a synchronized thinking session between multiple agents"""
    session_id: str
    participating_agents: Set[str]  # Agent IDs
    coordinator_agent: Optional[str]  # Agent ID of coordinator
    thinking_topic: str
    thinking_phases: List[str] = field(default_factory=lambda: ["analysis", "synthesis", "consensus"])
    current_phase: str = "analysis"
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # phase -> agent_id -> result
    started_at: float = field(default_factory=time.time)
    current_phase_start: float = field(default_factory=time.time)
    phase_timeout: float = 60.0  # Timeout per phase in seconds
    is_active: bool = True
    final_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participating_agents": list(self.participating_agents),
            "coordinator_agent": self.coordinator_agent,
            "thinking_topic": self.thinking_topic,
            "thinking_phases": self.thinking_phases,
            "current_phase": self.current_phase,
            "phase_results": self.phase_results,
            "started_at": self.started_at,
            "current_phase_start": self.current_phase_start,
            "phase_timeout": self.phase_timeout,
            "is_active": self.is_active,
            "has_final_result": self.final_result is not None
        }


@dataclass
class ConversationThread:
    """Tracks a conversation thread between agents"""
    thread_id: str
    participants: Set[str]  # Agent IDs
    messages: List[AgentMessage] = field(default_factory=list)
    topic: str = ""
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the conversation thread"""
        self.messages.append(message)
        self.last_activity = time.time()
        self.participants.add(message.from_agent)
        if message.to_agent != "ALL":
            self.participants.add(message.to_agent)


class AgentCommunicationSystem:
    """
    Central communication hub for managing agent interactions, teams, and conversations.
    """
    
    def __init__(self, max_agents: int = 16):
        self.max_agents = max_agents
        self.agents: Dict[str, Agent] = {}
        self.teams: Dict[str, AgentTeam] = {}
        self.conversation_threads: Dict[str, ConversationThread] = {}
        self.synchronized_thinking_sessions: Dict[str, SynchronizedThinkingSession] = {}
        self.agent_api_clients: Dict[str, AgentAPIClient] = {}  # Agent ID -> API client
        self.message_queue: List[AgentMessage] = []
        self.core_storage = get_core_context_storage()
        
        # Thread safety
        self._agents_lock = threading.RLock()
        self._teams_lock = threading.RLock()
        self._messages_lock = threading.RLock()
        self._thinking_lock = threading.RLock()
        
        # Message handlers by type
        self.message_handlers: Dict[str, Callable] = {
            "insight": self._handle_insight_message,
            "question": self._handle_question_message,
            "request": self._handle_request_message,
            "response": self._handle_response_message,
            "alert": self._handle_alert_message,
            "thinking_sync": self._handle_thinking_sync_message
        }
        
        # Start background message processing
        self._shutdown = False
        self._message_processor = threading.Thread(target=self._process_messages, daemon=True)
        self._message_processor.start()
        
        # Start synchronized thinking monitor
        self._thinking_monitor = threading.Thread(target=self._monitor_thinking_sessions, daemon=True)
        self._thinking_monitor.start()
        
        logger.info(f"Agent communication system initialized for {max_agents} agents")
    
    def register_agent(self, core_id: int, role: AgentRole, agent_id: Optional[str] = None) -> Agent:
        """Register a new agent for a CPU core"""
        if agent_id is None:
            agent_id = f"agent_{role.value}_{core_id}_{int(time.time())}"
        
        with self._agents_lock:
            agent = Agent(agent_id=agent_id, core_id=core_id, role=role)
            self.agents[agent_id] = agent
            
            # Create API client for this agent
            api_client = AgentAPIClient(agent, max_concurrent_calls=3)
            self.agent_api_clients[agent_id] = api_client
            
            # Initialize persistent memory for the agent
            try:
                persistent_memory = agent.get_persistent_memory()
                if persistent_memory:
                    # Try to recover from previous session if snapshots exist
                    if agent.recover_from_failure():
                        logger.info(f"Agent {agent_id} recovered context from previous session")
                    else:
                        # Create initial context snapshot for new agent
                        snapshot_id = agent.create_context_snapshot()
                        if snapshot_id:
                            logger.debug(f"Created initial context snapshot {snapshot_id} for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize persistent memory for agent {agent_id}: {e}")
            
            # Store agent info in core context storage
            self.core_storage.set_core_context(
                "agent_info", 
                agent.get_status_summary(),
                core_id=core_id,
                share_with_others=True
            )
            
            logger.info(f"Registered agent {agent_id} with role {role.value} on core {core_id} with API client and persistent memory")
            return agent
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        with self._agents_lock:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                
                # Create final context snapshot before shutdown
                try:
                    snapshot_id = agent.create_context_snapshot()
                    if snapshot_id:
                        logger.debug(f"Created final context snapshot {snapshot_id} for agent {agent_id}")
                        
                        # Persist important insights before shutdown
                        agent.persist_insight(
                            content=f"Agent shutdown at {time.time()}, final snapshot created",
                            importance=0.8,
                            tags={"shutdown", "snapshot"}
                        )
                except Exception as e:
                    logger.warning(f"Failed to create final snapshot for agent {agent_id}: {e}")
                
                # Remove from all teams
                for team_id in list(agent.team_memberships):
                    self.remove_agent_from_team(agent_id, team_id)
                
                # Update status and remove
                agent.update_status(AgentStatus.OFFLINE)
                del self.agents[agent_id]
                
                # Clean up API client
                if agent_id in self.agent_api_clients:
                    del self.agent_api_clients[agent_id]
                
                logger.info(f"Unregistered agent {agent_id} with persistent memory cleanup")
                return True
            return False
    
    def send_message(
        self, 
        from_agent: str, 
        to_agent: str, 
        message_type: str, 
        content: str,
        priority: int = 5,
        response_to: Optional[str] = None
    ) -> Optional[str]:
        """Send a message between agents"""
        if from_agent not in self.agents:
            logger.warning(f"Unknown sender agent: {from_agent}")
            return None
        
        # Create and queue message
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            priority=priority,
            response_to=response_to
        )
        
        with self._messages_lock:
            self.message_queue.append(message)
            # Sort by priority (higher priority first)
            self.message_queue.sort(key=lambda m: m.priority, reverse=True)
        
        # Update sender's outgoing messages
        self.agents[from_agent].send_message(to_agent, message_type, content, priority)
        
        logger.debug(f"Queued message from {from_agent} to {to_agent}: {message_type}")
        return message.message_id
    
    def _process_messages(self) -> None:
        """Background thread to process queued messages"""
        while not self._shutdown:
            try:
                messages_to_process = []
                
                # Get messages to process
                with self._messages_lock:
                    if self.message_queue:
                        messages_to_process = self.message_queue[:10]  # Process up to 10 at a time
                        self.message_queue = self.message_queue[10:]
                
                # Process each message
                for message in messages_to_process:
                    self._deliver_message(message)
                    
                time.sleep(0.1)  # Brief pause between processing cycles
                
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                time.sleep(1)  # Longer pause on error
    
    def _deliver_message(self, message: AgentMessage) -> None:
        """Deliver a message to its intended recipient(s)"""
        try:
            if message.to_agent == "ALL":
                # Broadcast message
                with self._agents_lock:
                    for agent_id, agent in self.agents.items():
                        if agent_id != message.from_agent:
                            agent.receive_message(message)
            else:
                # Direct message
                with self._agents_lock:
                    if message.to_agent in self.agents:
                        self.agents[message.to_agent].receive_message(message)
                    else:
                        logger.warning(f"Message recipient not found: {message.to_agent}")
                        return
            
            # Handle message based on type
            handler = self.message_handlers.get(message.message_type, self._handle_default_message)
            handler(message)
            
            # Add to conversation thread
            self._add_to_conversation_thread(message)
            
        except Exception as e:
            logger.error(f"Error delivering message {message.message_id}: {e}")
    
    def _handle_insight_message(self, message: AgentMessage) -> None:
        """Handle insight-type messages"""
        # Insights are automatically shared in core context storage
        if message.from_agent in self.agents:
            agent = self.agents[message.from_agent]
            self.core_storage.set_core_context(
                f"insight_{message.message_id}",
                {
                    "content": message.content,
                    "from_agent": message.from_agent,
                    "timestamp": message.timestamp
                },
                core_id=agent.core_id,
                share_with_others=True
            )
    
    def _handle_question_message(self, message: AgentMessage) -> None:
        """Handle question-type messages"""
        # Questions might trigger responses from relevant agents
        # This is a simplified implementation
        pass
    
    def _handle_request_message(self, message: AgentMessage) -> None:
        """Handle request-type messages"""
        # Requests might trigger task delegation
        pass
    
    def _handle_response_message(self, message: AgentMessage) -> None:
        """Handle response-type messages"""
        # Responses complete conversation threads
        pass
    
    def _handle_alert_message(self, message: AgentMessage) -> None:
        """Handle alert-type messages"""
        # Alerts might trigger immediate attention or team formation
        logger.warning(f"Agent alert from {message.from_agent}: {message.content}")
    
    def _handle_default_message(self, message: AgentMessage) -> None:
        """Handle unknown message types"""
        logger.debug(f"Processing message type {message.message_type} from {message.from_agent}")
    
    def _add_to_conversation_thread(self, message: AgentMessage) -> None:
        """Add message to appropriate conversation thread"""
        # Find existing thread or create new one
        thread_id = None
        
        if message.response_to:
            # Find thread containing the message this responds to
            for tid, thread in self.conversation_threads.items():
                if any(msg.message_id == message.response_to for msg in thread.messages):
                    thread_id = tid
                    break
        
        if not thread_id:
            # Create new thread
            thread_id = str(uuid.uuid4())
            self.conversation_threads[thread_id] = ConversationThread(
                thread_id=thread_id,
                participants={message.from_agent}
            )
        
        self.conversation_threads[thread_id].add_message(message)
    
    def create_team(self, team_name: str, purpose: str = "", leader_agent: Optional[str] = None) -> str:
        """Create a new agent team"""
        team_id = str(uuid.uuid4())
        
        with self._teams_lock:
            team = AgentTeam(
                team_id=team_id,
                team_name=team_name,
                leader_agent=leader_agent,
                purpose=purpose
            )
            self.teams[team_id] = team
            
            logger.info(f"Created team {team_name} with ID {team_id}")
            return team_id
    
    def add_agent_to_team(self, agent_id: str, team_id: str) -> bool:
        """Add an agent to a team"""
        with self._agents_lock, self._teams_lock:
            if agent_id not in self.agents or team_id not in self.teams:
                return False
            
            agent = self.agents[agent_id]
            team = self.teams[team_id]
            
            team.members.add(agent_id)
            agent.join_team(team_id)
            
            logger.debug(f"Added agent {agent_id} to team {team.team_name}")
            return True
    
    def remove_agent_from_team(self, agent_id: str, team_id: str) -> bool:
        """Remove an agent from a team"""
        with self._agents_lock, self._teams_lock:
            if agent_id not in self.agents or team_id not in self.teams:
                return False
            
            agent = self.agents[agent_id]
            team = self.teams[team_id]
            
            team.members.discard(agent_id)
            agent.leave_team(team_id)
            
            # If this was the leader, clear leadership
            if team.leader_agent == agent_id:
                team.leader_agent = None
            
            logger.debug(f"Removed agent {agent_id} from team {team.team_name}")
            return True
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """Get all agents with a specific role"""
        with self._agents_lock:
            return [agent for agent in self.agents.values() if agent.role == role]
    
    def get_available_agents(self) -> List[Agent]:
        """Get all active agents"""
        with self._agents_lock:
            return [
                agent for agent in self.agents.values() 
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.WAITING]
            ]
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent communication system"""
        with self._agents_lock, self._teams_lock:
            agent_stats = {}
            for agent_id, agent in self.agents.items():
                agent_stats[agent_id] = agent.get_status_summary()
            
            team_stats = {}
            for team_id, team in self.teams.items():
                team_stats[team_id] = team.to_dict()
            
            return {
                "total_agents": len(self.agents),
                "total_teams": len(self.teams),
                "total_conversations": len(self.conversation_threads),
                "message_queue_size": len(self.message_queue),
                "agents": agent_stats,
                "teams": team_stats,
                "roles_distribution": self._get_role_distribution()
            }
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of agent roles"""
        distribution = defaultdict(int)
        for agent in self.agents.values():
            distribution[agent.role.value] += 1
        return dict(distribution)
    
    # Synchronized Thinking Methods
    
    def start_synchronized_thinking(
        self,
        participating_agents: List[str],
        thinking_topic: str,
        coordinator_agent: Optional[str] = None,
        phases: Optional[List[str]] = None,
        phase_timeout: float = 60.0
    ) -> str:
        """Start a synchronized thinking session between multiple agents"""
        session_id = str(uuid.uuid4())
        
        if phases is None:
            phases = ["analysis", "synthesis", "consensus"]
        
        if coordinator_agent is None and participating_agents:
            # Choose agent with most leadership-oriented role as coordinator
            leadership_roles = [AgentRole.PLANNING_COORDINATOR, AgentRole.CONSENSUS_FACILITATOR, AgentRole.ARCHITECTURE_REVIEWER]
            for role in leadership_roles:
                for agent_id in participating_agents:
                    if agent_id in self.agents and self.agents[agent_id].role == role:
                        coordinator_agent = agent_id
                        break
                if coordinator_agent:
                    break
            
            # Fallback to first agent if no leader found
            if not coordinator_agent:
                coordinator_agent = participating_agents[0]
        
        with self._thinking_lock:
            session = SynchronizedThinkingSession(
                session_id=session_id,
                participating_agents=set(participating_agents),
                coordinator_agent=coordinator_agent,
                thinking_topic=thinking_topic,
                thinking_phases=phases,
                phase_timeout=phase_timeout
            )
            
            self.synchronized_thinking_sessions[session_id] = session
        
        # Notify all participating agents
        for agent_id in participating_agents:
            if agent_id in self.agents:
                self.send_message(
                    from_agent=coordinator_agent,
                    to_agent=agent_id,
                    message_type="thinking_sync",
                    content=f"Starting synchronized thinking session: {thinking_topic}",
                    priority=8
                )
                
                # Update agent status
                self.agents[agent_id].update_status(AgentStatus.THINKING)
                
                # Add thought about joining session
                self.agents[agent_id].add_thought(
                    thought_type="sync_thinking",
                    content=f"Joining synchronized thinking session about: {thinking_topic}",
                    confidence=0.9
                )
        
        logger.info(f"Started synchronized thinking session {session_id} with {len(participating_agents)} agents")
        return session_id
    
    async def execute_thinking_phase(
        self,
        session_id: str,
        agent_id: str,
        phase_prompt: str
    ) -> Optional[str]:
        """Execute a thinking phase for a specific agent"""
        with self._thinking_lock:
            if session_id not in self.synchronized_thinking_sessions:
                logger.warning(f"Thinking session {session_id} not found")
                return None
            
            session = self.synchronized_thinking_sessions[session_id]
            if agent_id not in session.participating_agents:
                logger.warning(f"Agent {agent_id} not part of thinking session {session_id}")
                return None
        
        # Get agent's API client
        if agent_id not in self.agent_api_clients:
            logger.error(f"No API client found for agent {agent_id}")
            return None
        
        api_client = self.agent_api_clients[agent_id]
        
        try:
            # Make API call through agent's own client
            api_call = await api_client.make_api_call(
                prompt=phase_prompt,
                parameters={"temperature": 0.7, "thinking_mode": "high"}
            )
            
            if api_call.status == "completed" and api_call.response:
                # Store result in session
                with self._thinking_lock:
                    if session.current_phase not in session.phase_results:
                        session.phase_results[session.current_phase] = {}
                    session.phase_results[session.current_phase][agent_id] = {
                        "response": api_call.response,
                        "timestamp": time.time(),
                        "execution_time": api_call.execution_time
                    }
                
                # Add thought about completed phase
                if agent_id in self.agents:
                    self.agents[agent_id].add_thought(
                        thought_type="phase_complete",
                        content=f"Completed {session.current_phase} phase of synchronized thinking",
                        confidence=0.9
                    )
                
                return api_call.response
            else:
                logger.error(f"Agent {agent_id} API call failed in thinking session: {api_call.error}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing thinking phase for agent {agent_id}: {e}")
            return None
    
    def get_agent_api_client(self, agent_id: str) -> Optional[AgentAPIClient]:
        """Get the API client for a specific agent"""
        return self.agent_api_clients.get(agent_id)
    
    def get_thinking_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a synchronized thinking session"""
        with self._thinking_lock:
            if session_id not in self.synchronized_thinking_sessions:
                return None
            
            session = self.synchronized_thinking_sessions[session_id]
            return session.to_dict()
    
    def _handle_thinking_sync_message(self, message: AgentMessage) -> None:
        """Handle synchronized thinking messages"""
        logger.debug(f"Processing thinking sync message from {message.from_agent}")
    
    def _monitor_thinking_sessions(self) -> None:
        """Background thread to monitor thinking session timeouts"""
        while not self._shutdown:
            try:
                current_time = time.time()
                
                with self._thinking_lock:
                    sessions_to_advance = []
                    for session_id, session in self.synchronized_thinking_sessions.items():
                        if session.is_active:
                            phase_elapsed = current_time - session.current_phase_start
                            if phase_elapsed >= session.phase_timeout:
                                sessions_to_advance.append(session_id)
                
                # Advance sessions that have timed out (simplified for now)
                for session_id in sessions_to_advance:
                    logger.warning(f"Thinking session {session_id} phase timed out")
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in thinking session monitor: {e}")
                time.sleep(10)
    
    def shutdown(self) -> None:
        """Graceful shutdown of the communication system"""
        self._shutdown = True
        if self._message_processor.is_alive():
            self._message_processor.join(timeout=2)
        if hasattr(self, '_thinking_monitor') and self._thinking_monitor.is_alive():
            self._thinking_monitor.join(timeout=2)
        
        # Update all agents to offline status
        with self._agents_lock:
            for agent in self.agents.values():
                agent.update_status(AgentStatus.OFFLINE)
        
        # Clean up API clients
        self.agent_api_clients.clear()
        
        # Clean up thinking sessions
        with self._thinking_lock:
            self.synchronized_thinking_sessions.clear()


# Global singleton instance
_communication_system_instance = None
_communication_system_lock = threading.Lock()


def get_agent_communication_system() -> AgentCommunicationSystem:
    """Get the global agent communication system instance (singleton pattern)"""
    global _communication_system_instance
    if _communication_system_instance is None:
        with _communication_system_lock:
            if _communication_system_instance is None:
                _communication_system_instance = AgentCommunicationSystem()
    return _communication_system_instance