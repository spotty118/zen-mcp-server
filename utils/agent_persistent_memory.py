"""
Agent Persistent Memory and Context Management

This module implements persistent memory and context management for per-core agents,
allowing them to persist key insights and decisions across thinking sessions and
recover context when agents restart after failures.

Key Features:
- Persistent memory storage for agent insights and decisions
- Context recovery capabilities for agent restarts
- Integration with existing CoreContextStorage for per-core isolation
- Memory consolidation and cleanup strategies
- Cross-session context continuity
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from utils.agent_core import Agent, AgentRole, AgentThought
from utils.core_context_storage import get_core_context_storage, CoreContextStorage

logger = logging.getLogger(__name__)


@dataclass
class AgentMemoryEntry:
    """Represents a single memory entry for an agent"""
    entry_id: str
    agent_id: str
    core_id: int
    entry_type: str  # "insight", "decision", "pattern", "preference", "context"
    content: str
    importance: float  # 0.0 to 1.0, higher means more important to persist
    confidence: float  # 0.0 to 1.0, confidence in this memory
    created_at: float
    last_accessed: float
    access_count: int = 0
    related_entries: List[str] = field(default_factory=list)  # IDs of related memories
    tags: Set[str] = field(default_factory=set)  # Searchable tags
    session_context: Dict[str, Any] = field(default_factory=dict)  # Context when created
    
    def mark_accessed(self) -> None:
        """Mark this memory as accessed"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "agent_id": self.agent_id,
            "core_id": self.core_id,
            "entry_type": self.entry_type,
            "content": self.content,
            "importance": self.importance,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "related_entries": self.related_entries,
            "tags": list(self.tags),
            "session_context": self.session_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemoryEntry':
        """Create AgentMemoryEntry from dictionary"""
        data = data.copy()
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


@dataclass
class AgentContextSnapshot:
    """Snapshot of agent context at a specific point in time"""
    snapshot_id: str
    agent_id: str
    core_id: int
    timestamp: float
    agent_status: str
    recent_thoughts: List[Dict[str, Any]]  # Serialized AgentThought objects
    context_data: Dict[str, Any]
    active_sessions: List[str]  # IDs of active thinking sessions
    team_memberships: List[str]  # Team IDs
    performance_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContextSnapshot':
        """Create AgentContextSnapshot from dictionary"""
        return cls(**data)


class AgentPersistentMemory:
    """
    Manages persistent memory and context for individual agents
    """
    
    def __init__(self, agent: Agent, storage_dir: Optional[str] = None):
        """
        Initialize persistent memory for an agent
        
        Args:
            agent: The agent this memory belongs to
            storage_dir: Directory to store persistent memory files (optional)
        """
        self.agent = agent
        self.storage_dir = Path(storage_dir or "logs/agent_memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory storage
        self.memory_entries: Dict[str, AgentMemoryEntry] = {}
        self.context_snapshots: Dict[str, AgentContextSnapshot] = {}
        
        # Core context storage integration
        self.core_storage = get_core_context_storage()
        
        # Thread safety
        self._memory_lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        
        # Memory management settings
        self.max_memory_entries = 1000  # Maximum number of memory entries to keep
        self.max_context_snapshots = 50  # Maximum number of context snapshots
        self.cleanup_interval = 3600  # Cleanup every hour
        self.importance_threshold = 0.3  # Minimum importance to persist
        
        # Background cleanup
        self._shutdown = False
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        # Load existing memory from disk
        self._load_persistent_memory()
        
        logger.info(f"Persistent memory initialized for agent {agent.agent_id} "
                   f"with {len(self.memory_entries)} existing entries")
    
    def persist_insight(
        self,
        content: str,
        importance: float = 0.7,
        confidence: float = 0.8,
        tags: Optional[Set[str]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Persist an insight from the agent
        
        Args:
            content: The insight content
            importance: Importance level (0.0-1.0)
            confidence: Confidence level (0.0-1.0)
            tags: Optional tags for categorization
            session_context: Context from the current session
            
        Returns:
            Entry ID of the persisted insight
        """
        return self._create_memory_entry(
            entry_type="insight",
            content=content,
            importance=importance,
            confidence=confidence,
            tags=tags or set(),
            session_context=session_context or {}
        )
    
    def persist_decision(
        self,
        content: str,
        importance: float = 0.8,
        confidence: float = 0.9,
        tags: Optional[Set[str]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Persist a decision made by the agent
        
        Args:
            content: The decision content
            importance: Importance level (0.0-1.0)
            confidence: Confidence level (0.0-1.0)
            tags: Optional tags for categorization
            session_context: Context from the current session
            
        Returns:
            Entry ID of the persisted decision
        """
        return self._create_memory_entry(
            entry_type="decision",
            content=content,
            importance=importance,
            confidence=confidence,
            tags=tags or set(),
            session_context=session_context or {}
        )
    
    def persist_pattern(
        self,
        content: str,
        importance: float = 0.6,
        confidence: float = 0.7,
        tags: Optional[Set[str]] = None,
        session_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Persist a pattern recognized by the agent
        
        Args:
            content: The pattern description
            importance: Importance level (0.0-1.0)
            confidence: Confidence level (0.0-1.0)
            tags: Optional tags for categorization
            session_context: Context from the current session
            
        Returns:
            Entry ID of the persisted pattern
        """
        return self._create_memory_entry(
            entry_type="pattern",
            content=content,
            importance=importance,
            confidence=confidence,
            tags=tags or set(),
            session_context=session_context or {}
        )
    
    def _create_memory_entry(
        self,
        entry_type: str,
        content: str,
        importance: float,
        confidence: float,
        tags: Set[str],
        session_context: Dict[str, Any]
    ) -> str:
        """Create and store a memory entry"""
        if importance < self.importance_threshold:
            logger.debug(f"Skipping memory entry with low importance: {importance}")
            return ""
        
        entry_id = str(uuid.uuid4())
        current_time = time.time()
        
        with self._memory_lock:
            entry = AgentMemoryEntry(
                entry_id=entry_id,
                agent_id=self.agent.agent_id,
                core_id=self.agent.core_id,
                entry_type=entry_type,
                content=content,
                importance=importance,
                confidence=confidence,
                created_at=current_time,
                last_accessed=current_time,
                tags=tags,
                session_context=session_context
            )
            
            self.memory_entries[entry_id] = entry
            
            # Enforce memory limits by removing least important entries if needed
            if len(self.memory_entries) > self.max_memory_entries:
                self._enforce_memory_limits()
            
            # Store in core context storage for sharing with other agents if important enough
            if importance >= 0.8:
                self.core_storage.set_core_context(
                    f"agent_memory_{entry_type}_{entry_id}",
                    entry.to_dict(),
                    core_id=self.agent.core_id,
                    share_with_others=True
                )
            
            logger.debug(f"Created {entry_type} memory entry {entry_id} for agent {self.agent.agent_id}")
            
            # Trigger persistence to disk
            self._persist_memory_to_disk()
            
            return entry_id
    
    def retrieve_memories(
        self,
        entry_type: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        min_importance: float = 0.0,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[AgentMemoryEntry]:
        """
        Retrieve memories based on criteria
        
        Args:
            entry_type: Filter by entry type
            tags: Filter by tags (any match)
            min_importance: Minimum importance threshold
            min_confidence: Minimum confidence threshold
            limit: Maximum number of entries to return
            
        Returns:
            List of matching memory entries, sorted by importance and recency
        """
        with self._memory_lock:
            matching_entries = []
            
            for entry in self.memory_entries.values():
                # Apply filters
                if entry_type and entry.entry_type != entry_type:
                    continue
                if entry.importance < min_importance:
                    continue
                if entry.confidence < min_confidence:
                    continue
                if tags and not tags.intersection(entry.tags):
                    continue
                
                # Mark as accessed
                entry.mark_accessed()
                matching_entries.append(entry)
            
            # Sort by importance (descending) then by recency (descending)
            matching_entries.sort(
                key=lambda e: (e.importance, e.created_at),
                reverse=True
            )
            
            return matching_entries[:limit]
    
    def create_context_snapshot(self) -> str:
        """
        Create a snapshot of the current agent context
        
        Returns:
            Snapshot ID
        """
        snapshot_id = str(uuid.uuid4())
        current_time = time.time()
        
        with self._snapshot_lock:
            # Gather current agent state
            recent_thoughts = [
                thought.to_dict() for thought in self.agent.get_recent_thoughts(20)
            ]
            
            # Get performance metrics from agent status if available
            performance_metrics = {}
            try:
                from utils.per_core_agent_manager import get_per_core_agent_manager
                manager = get_per_core_agent_manager()
                if manager and self.agent.agent_id in manager.agent_statuses:
                    status = manager.agent_statuses[self.agent.agent_id]
                    performance_metrics = {
                        "total_api_calls": status.total_api_calls,
                        "success_rate": status.success_rate,
                        "memory_usage_mb": status.memory_usage_mb,
                        "active_thinking_sessions": status.active_thinking_sessions
                    }
            except Exception as e:
                logger.debug(f"Could not gather performance metrics: {e}")
            
            snapshot = AgentContextSnapshot(
                snapshot_id=snapshot_id,
                agent_id=self.agent.agent_id,
                core_id=self.agent.core_id,
                timestamp=current_time,
                agent_status=self.agent.status.value,
                recent_thoughts=recent_thoughts,
                context_data=self.agent.context.copy(),
                active_sessions=[],  # Would be populated by thinking session manager
                team_memberships=list(self.agent.team_memberships),
                performance_metrics=performance_metrics
            )
            
            self.context_snapshots[snapshot_id] = snapshot
            
            # Store in core context storage
            self.core_storage.set_core_context(
                f"agent_snapshot_{snapshot_id}",
                snapshot.to_dict(),
                core_id=self.agent.core_id,
                share_with_others=False  # Snapshots are agent-specific
            )
            
            logger.debug(f"Created context snapshot {snapshot_id} for agent {self.agent.agent_id}")
            
            # Trigger persistence to disk
            self._persist_snapshots_to_disk()
            
            return snapshot_id
    
    def recover_from_snapshot(self, snapshot_id: str) -> bool:
        """
        Recover agent context from a snapshot
        
        Args:
            snapshot_id: ID of the snapshot to recover from
            
        Returns:
            True if recovery was successful, False otherwise
        """
        with self._snapshot_lock:
            if snapshot_id not in self.context_snapshots:
                logger.warning(f"Snapshot {snapshot_id} not found for agent {self.agent.agent_id}")
                return False
            
            snapshot = self.context_snapshots[snapshot_id]
            
            try:
                # Restore agent context
                self.agent.context.update(snapshot.context_data)
                
                # Restore team memberships
                self.agent.team_memberships = set(snapshot.team_memberships)
                
                # Recreate recent thoughts (as new thoughts to avoid ID conflicts)
                for thought_data in snapshot.recent_thoughts[-10:]:  # Only restore last 10
                    self.agent.add_thought(
                        thought_type=f"recovered_{thought_data['thought_type']}",
                        content=f"[RECOVERED] {thought_data['content']}",
                        confidence=thought_data['confidence'] * 0.8  # Reduce confidence for recovered thoughts
                    )
                
                logger.info(f"Successfully recovered agent {self.agent.agent_id} from snapshot {snapshot_id}")
                
                # Create a memory entry about the recovery
                self.persist_insight(
                    content=f"Agent context recovered from snapshot {snapshot_id} created at {snapshot.timestamp}",
                    importance=0.9,
                    confidence=0.95,
                    tags={"recovery", "context", "snapshot"},
                    session_context={"recovery_snapshot_id": snapshot_id}
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to recover agent {self.agent.agent_id} from snapshot {snapshot_id}: {e}")
                return False
    
    def get_latest_snapshot(self) -> Optional[AgentContextSnapshot]:
        """Get the most recent context snapshot"""
        with self._snapshot_lock:
            if not self.context_snapshots:
                return None
            
            latest_snapshot = max(
                self.context_snapshots.values(),
                key=lambda s: s.timestamp
            )
            return latest_snapshot
    
    def _persist_memory_to_disk(self) -> None:
        """Persist memory entries to disk"""
        try:
            memory_file = self.storage_dir / f"agent_{self.agent.agent_id}_memory.json"
            
            with self._memory_lock:
                memory_data = {
                    "agent_id": self.agent.agent_id,
                    "core_id": self.agent.core_id,
                    "last_updated": time.time(),
                    "entries": [entry.to_dict() for entry in self.memory_entries.values()]
                }
                
                with open(memory_file, 'w') as f:
                    json.dump(memory_data, f, indent=2)
                
                logger.debug(f"Persisted {len(self.memory_entries)} memory entries to {memory_file}")
                
        except Exception as e:
            logger.error(f"Failed to persist memory to disk for agent {self.agent.agent_id}: {e}")
    
    def _persist_snapshots_to_disk(self) -> None:
        """Persist context snapshots to disk"""
        try:
            snapshots_file = self.storage_dir / f"agent_{self.agent.agent_id}_snapshots.json"
            
            with self._snapshot_lock:
                snapshots_data = {
                    "agent_id": self.agent.agent_id,
                    "core_id": self.agent.core_id,
                    "last_updated": time.time(),
                    "snapshots": [snapshot.to_dict() for snapshot in self.context_snapshots.values()]
                }
                
                with open(snapshots_file, 'w') as f:
                    json.dump(snapshots_data, f, indent=2)
                
                logger.debug(f"Persisted {len(self.context_snapshots)} snapshots to {snapshots_file}")
                
        except Exception as e:
            logger.error(f"Failed to persist snapshots to disk for agent {self.agent.agent_id}: {e}")
    
    def _load_persistent_memory(self) -> None:
        """Load persistent memory from disk"""
        try:
            # Load memory entries
            memory_file = self.storage_dir / f"agent_{self.agent.agent_id}_memory.json"
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                with self._memory_lock:
                    for entry_data in memory_data.get("entries", []):
                        entry = AgentMemoryEntry.from_dict(entry_data)
                        self.memory_entries[entry.entry_id] = entry
                
                logger.info(f"Loaded {len(self.memory_entries)} memory entries for agent {self.agent.agent_id}")
            
            # Load context snapshots
            snapshots_file = self.storage_dir / f"agent_{self.agent.agent_id}_snapshots.json"
            if snapshots_file.exists():
                with open(snapshots_file, 'r') as f:
                    snapshots_data = json.load(f)
                
                with self._snapshot_lock:
                    for snapshot_data in snapshots_data.get("snapshots", []):
                        snapshot = AgentContextSnapshot.from_dict(snapshot_data)
                        self.context_snapshots[snapshot.snapshot_id] = snapshot
                
                logger.info(f"Loaded {len(self.context_snapshots)} snapshots for agent {self.agent.agent_id}")
                
        except Exception as e:
            logger.error(f"Failed to load persistent memory for agent {self.agent.agent_id}: {e}")
    
    def _cleanup_worker(self) -> None:
        """Background worker for memory cleanup"""
        while not self._shutdown:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_entries()
            except Exception as e:
                logger.error(f"Error in memory cleanup worker: {e}")
    
    def _enforce_memory_limits(self) -> None:
        """Enforce memory limits by removing least important entries"""
        if len(self.memory_entries) <= self.max_memory_entries:
            return
        
        # Keep the most important and recently accessed entries
        sorted_entries = sorted(
            self.memory_entries.items(),
            key=lambda x: (x[1].importance, x[1].last_accessed),
            reverse=True
        )
        
        entries_to_keep = dict(sorted_entries[:self.max_memory_entries])
        removed_count = len(self.memory_entries) - len(entries_to_keep)
        self.memory_entries = entries_to_keep
        
        logger.debug(f"Enforced memory limits for agent {self.agent.agent_id}, removed {removed_count} entries")
    
    def _cleanup_old_entries(self) -> None:
        """Clean up old and low-importance memory entries"""
        current_time = time.time()
        cleanup_threshold = 7 * 24 * 3600  # 7 days
        
        with self._memory_lock:
            # Remove old, low-importance entries
            entries_to_remove = []
            for entry_id, entry in self.memory_entries.items():
                age = current_time - entry.created_at
                if (age > cleanup_threshold and 
                    entry.importance < 0.5 and 
                    entry.access_count < 3):
                    entries_to_remove.append(entry_id)
            
            for entry_id in entries_to_remove:
                del self.memory_entries[entry_id]
            
            # Enforce memory limits
            self._enforce_memory_limits()
        
        with self._snapshot_lock:
            # Remove old snapshots
            if len(self.context_snapshots) > self.max_context_snapshots:
                sorted_snapshots = sorted(
                    self.context_snapshots.items(),
                    key=lambda x: x[1].timestamp,
                    reverse=True
                )
                
                snapshots_to_keep = dict(sorted_snapshots[:self.max_context_snapshots])
                self.context_snapshots = snapshots_to_keep
        
        if entries_to_remove:
            logger.debug(f"Cleaned up {len(entries_to_remove)} old memory entries for agent {self.agent.agent_id}")
            self._persist_memory_to_disk()
            self._persist_snapshots_to_disk()
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent's persistent memory"""
        with self._memory_lock, self._snapshot_lock:
            entry_types = {}
            total_importance = 0.0
            total_confidence = 0.0
            
            for entry in self.memory_entries.values():
                entry_types[entry.entry_type] = entry_types.get(entry.entry_type, 0) + 1
                total_importance += entry.importance
                total_confidence += entry.confidence
            
            num_entries = len(self.memory_entries)
            
            return {
                "agent_id": self.agent.agent_id,
                "total_memory_entries": num_entries,
                "total_context_snapshots": len(self.context_snapshots),
                "entry_types": entry_types,
                "average_importance": total_importance / num_entries if num_entries > 0 else 0.0,
                "average_confidence": total_confidence / num_entries if num_entries > 0 else 0.0,
                "storage_directory": str(self.storage_dir),
                "cleanup_interval": self.cleanup_interval,
                "importance_threshold": self.importance_threshold
            }
    
    def shutdown(self) -> None:
        """Graceful shutdown of persistent memory"""
        self._shutdown = True
        
        # Final persistence
        self._persist_memory_to_disk()
        self._persist_snapshots_to_disk()
        
        logger.info(f"Persistent memory shutdown completed for agent {self.agent.agent_id}")


# Global registry for agent persistent memory instances
_agent_memory_instances: Dict[str, AgentPersistentMemory] = {}
_agent_memory_lock = threading.Lock()


def get_agent_persistent_memory(agent: Agent) -> AgentPersistentMemory:
    """
    Get or create persistent memory instance for an agent
    
    Args:
        agent: The agent to get persistent memory for
        
    Returns:
        AgentPersistentMemory instance
    """
    global _agent_memory_instances
    
    with _agent_memory_lock:
        if agent.agent_id not in _agent_memory_instances:
            _agent_memory_instances[agent.agent_id] = AgentPersistentMemory(agent)
        
        return _agent_memory_instances[agent.agent_id]


def shutdown_all_agent_memories() -> None:
    """Shutdown all agent persistent memory instances"""
    global _agent_memory_instances
    
    with _agent_memory_lock:
        for memory_instance in _agent_memory_instances.values():
            try:
                memory_instance.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent memory: {e}")
        
        _agent_memory_instances.clear()
    
    logger.info("All agent persistent memory instances shut down")