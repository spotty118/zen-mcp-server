"""
Agent Core - Individual Agent Abstraction Layer

This module defines the core agent abstraction where each CPU core acts as an autonomous
agent with its own personality, expertise area, thoughts, and communication capabilities.
Agents can discover each other, form teams, and collaborate on complex tasks while
maintaining their individual context and decision-making processes.

Key Features:
- Agent personality and expertise definitions
- Individual agent thoughts and decision logging
- Agent-to-agent communication protocols
- Dynamic team formation and coordination
- Agent status tracking and health monitoring
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Predefined agent roles with specific expertise areas"""
    SECURITY_ANALYST = "security_analyst"
    PERFORMANCE_OPTIMIZER = "performance_optimizer" 
    ARCHITECTURE_REVIEWER = "architecture_reviewer"
    CODE_QUALITY_INSPECTOR = "code_quality_inspector"
    DEBUG_SPECIALIST = "debug_specialist"
    PLANNING_COORDINATOR = "planning_coordinator"
    CONSENSUS_FACILITATOR = "consensus_facilitator"
    GENERALIST = "generalist"


class AgentStatus(Enum):
    """Agent operational status"""
    ACTIVE = "active"
    THINKING = "thinking"
    COMMUNICATING = "communicating"
    WAITING = "waiting"
    OFFLINE = "offline"


@dataclass
class AgentThought:
    """Represents a single thought or decision made by an agent"""
    agent_id: str
    timestamp: float
    thought_type: str  # "analysis", "decision", "question", "insight", "concern"
    content: str
    confidence: float  # 0.0 to 1.0
    related_thoughts: List[str] = field(default_factory=list)  # IDs of related thoughts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "thought_type": self.thought_type,
            "content": self.content,
            "confidence": self.confidence,
            "related_thoughts": self.related_thoughts
        }


@dataclass
class AgentMessage:
    """Message sent between agents"""
    message_id: str
    from_agent: str
    to_agent: str  # Can be "ALL" for broadcast
    message_type: str  # "insight", "question", "request", "response", "alert"
    content: str
    priority: int  # 1-10, 10 being highest
    timestamp: float = field(default_factory=time.time)
    response_to: Optional[str] = None  # ID of message this responds to
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "response_to": self.response_to
        }


@dataclass 
class AgentPersonality:
    """Defines an agent's personality and communication style"""
    communication_style: str  # "direct", "analytical", "collaborative", "questioning"
    expertise_confidence: float  # How confident the agent is in its expertise area
    collaboration_preference: str  # "independent", "cooperative", "leading", "supporting"
    decision_making_style: str  # "fast", "thorough", "consensus-seeking", "evidence-based"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "communication_style": self.communication_style,
            "expertise_confidence": self.expertise_confidence,
            "collaboration_preference": self.collaboration_preference,
            "decision_making_style": self.decision_making_style
        }


class Agent:
    """
    Represents an individual agent (CPU core) with its own personality, expertise,
    thoughts, and communication capabilities.
    """
    
    def __init__(
        self, 
        agent_id: str,
        core_id: int,
        role: AgentRole,
        personality: Optional[AgentPersonality] = None
    ):
        self.agent_id = agent_id
        self.core_id = core_id
        self.role = role
        self.personality = personality or self._create_default_personality()
        self.status = AgentStatus.ACTIVE
        self.thoughts: List[AgentThought] = []
        self.incoming_messages: List[AgentMessage] = []
        self.outgoing_messages: List[AgentMessage] = []
        self.context: Dict[str, Any] = {}
        self.team_memberships: Set[str] = set()
        self.created_at = time.time()
        self.last_activity = time.time()
        self._lock = threading.RLock()
        
        logger.info(f"Agent {self.agent_id} created with role {self.role.value} on core {self.core_id}")
    
    def _create_default_personality(self) -> AgentPersonality:
        """Create default personality based on agent role"""
        personality_map = {
            AgentRole.SECURITY_ANALYST: AgentPersonality(
                communication_style="direct",
                expertise_confidence=0.9,
                collaboration_preference="independent",
                decision_making_style="evidence-based"
            ),
            AgentRole.PERFORMANCE_OPTIMIZER: AgentPersonality(
                communication_style="analytical", 
                expertise_confidence=0.85,
                collaboration_preference="cooperative",
                decision_making_style="thorough"
            ),
            AgentRole.ARCHITECTURE_REVIEWER: AgentPersonality(
                communication_style="collaborative",
                expertise_confidence=0.8,
                collaboration_preference="leading",
                decision_making_style="consensus-seeking"
            ),
            AgentRole.DEBUG_SPECIALIST: AgentPersonality(
                communication_style="questioning",
                expertise_confidence=0.75,
                collaboration_preference="supporting",
                decision_making_style="fast"
            ),
            AgentRole.GENERALIST: AgentPersonality(
                communication_style="collaborative",
                expertise_confidence=0.6,
                collaboration_preference="cooperative",
                decision_making_style="consensus-seeking"
            )
        }
        return personality_map.get(self.role, personality_map[AgentRole.GENERALIST])
    
    def add_thought(
        self, 
        thought_type: str, 
        content: str, 
        confidence: float = 0.7,
        related_thoughts: Optional[List[str]] = None
    ) -> str:
        """Add a new thought and return its ID"""
        with self._lock:
            thought = AgentThought(
                agent_id=self.agent_id,
                timestamp=time.time(),
                thought_type=thought_type,
                content=content,
                confidence=confidence,
                related_thoughts=related_thoughts or []
            )
            
            self.thoughts.append(thought)
            self.last_activity = time.time()
            
            # Keep only last 100 thoughts to manage memory
            if len(self.thoughts) > 100:
                self.thoughts = self.thoughts[-100:]
            
            thought_id = f"{self.agent_id}_{len(self.thoughts)}_{int(thought.timestamp)}"
            logger.debug(f"Agent {self.agent_id} added {thought_type} thought: {content[:50]}...")
            return thought_id
    
    def send_message(self, to_agent: str, message_type: str, content: str, priority: int = 5) -> str:
        """Send a message to another agent or broadcast"""
        with self._lock:
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                from_agent=self.agent_id,
                to_agent=to_agent,
                message_type=message_type,
                content=content,
                priority=priority
            )
            
            self.outgoing_messages.append(message)
            self.last_activity = time.time()
            
            logger.debug(f"Agent {self.agent_id} sent {message_type} to {to_agent}: {content[:50]}...")
            return message.message_id
    
    def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent"""
        with self._lock:
            self.incoming_messages.append(message)
            self.last_activity = time.time()
            
            # Automatically add thought about received message
            self.add_thought(
                thought_type="communication",
                content=f"Received {message.message_type} from {message.from_agent}: {message.content}",
                confidence=0.9
            )
            
            logger.debug(f"Agent {self.agent_id} received {message.message_type} from {message.from_agent}")
    
    def get_recent_thoughts(self, limit: int = 10) -> List[AgentThought]:
        """Get recent thoughts from this agent"""
        with self._lock:
            return self.thoughts[-limit:] if self.thoughts else []
    
    def get_unread_messages(self) -> List[AgentMessage]:
        """Get unread messages for this agent"""
        with self._lock:
            # For now, return all incoming messages. In a full implementation,
            # we'd track read status
            return self.incoming_messages.copy()
    
    def update_status(self, status: AgentStatus) -> None:
        """Update agent status"""
        with self._lock:
            old_status = self.status
            self.status = status
            self.last_activity = time.time()
            
            if old_status != status:
                logger.debug(f"Agent {self.agent_id} status changed from {old_status.value} to {status.value}")
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context information for this agent"""
        with self._lock:
            self.context[key] = value
            self.last_activity = time.time()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information for this agent"""
        with self._lock:
            return self.context.get(key, default)
    
    def join_team(self, team_id: str) -> None:
        """Join a team (collaboration group)"""
        with self._lock:
            self.team_memberships.add(team_id)
            logger.debug(f"Agent {self.agent_id} joined team {team_id}")
    
    def leave_team(self, team_id: str) -> None:
        """Leave a team"""
        with self._lock:
            self.team_memberships.discard(team_id)
            logger.debug(f"Agent {self.agent_id} left team {team_id}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of agent status and activity"""
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "core_id": self.core_id,
                "role": self.role.value,
                "status": self.status.value,
                "thought_count": len(self.thoughts),
                "message_count": len(self.incoming_messages),
                "team_count": len(self.team_memberships),
                "last_activity": self.last_activity,
                "personality": self.personality.to_dict(),
                "context_keys": list(self.context.keys())
            }