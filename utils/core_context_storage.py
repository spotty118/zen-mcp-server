"""
Core-Specific Context Storage Backend

This module provides per-CPU-core context isolation with inter-core sharing capabilities.
Each CPU core gets its own context storage space while allowing selective sharing of
relevant information between cores for enhanced parallel processing.

Key Features:
- Per-core context isolation for optimal cache locality
- Inter-core context sharing for collaborative processing  
- Thread-safe operations with minimal lock contention
- Memory-efficient context synchronization
- Graceful degradation on systems without core affinity support
"""

import logging
import os
import threading
import time
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass, field
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class CoreContext:
    """Context storage for a specific CPU core"""
    core_id: int
    context_data: Dict[str, Any] = field(default_factory=dict)
    shared_keys: Set[str] = field(default_factory=set)  # Keys to share with other cores
    last_access: float = field(default_factory=time.time)
    memory_usage: float = 0.0  # Track memory usage in MB
    
    def update_access_time(self):
        """Update last access timestamp"""
        self.last_access = time.time()


@dataclass
class SharedContext:
    """Shared context information across cores"""
    key: str
    value: Any
    source_core: int
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    
    def mark_accessed(self):
        """Mark this shared context as accessed"""
        self.access_count += 1


class CoreContextStorage:
    """Thread-safe storage with per-core context isolation and inter-core sharing"""
    
    def __init__(self, max_cores: int = 16):
        self.max_cores = max_cores
        self._core_contexts: Dict[int, CoreContext] = {}
        self._shared_contexts: Dict[str, SharedContext] = {}
        self._core_lock = threading.RLock()  # Use RLock to allow recursive locking
        self._shared_lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._shutdown = False
        
        # Start background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Core context storage initialized for {max_cores} cores")
    
    def _get_current_core_id(self) -> Optional[int]:
        """Attempt to get current CPU core ID"""
        try:
            # Try to get CPU affinity if available
            if hasattr(os, 'sched_getaffinity'):
                affinity = os.sched_getaffinity(0)
                if len(affinity) == 1:
                    return next(iter(affinity))
        except (AttributeError, OSError):
            pass
        
        # Fallback: use thread ID modulo max_cores for consistent assignment
        thread_id = threading.get_ident()
        return thread_id % self.max_cores
    
    def _ensure_core_context(self, core_id: int) -> CoreContext:
        """Ensure a core context exists for the given core ID"""
        if core_id not in self._core_contexts:
            with self._core_lock:
                if core_id not in self._core_contexts:
                    self._core_contexts[core_id] = CoreContext(core_id=core_id)
                    logger.debug(f"Created context for core {core_id}")
        return self._core_contexts[core_id]
    
    def set_core_context(self, key: str, value: Any, core_id: Optional[int] = None, 
                        share_with_others: bool = False) -> None:
        """Set context value for a specific core"""
        if core_id is None:
            core_id = self._get_current_core_id()
        
        if core_id is None:
            logger.warning("Could not determine core ID, using core 0")
            core_id = 0
        
        with self._core_lock:
            context = self._ensure_core_context(core_id)
            context.context_data[key] = value
            context.update_access_time()
            
            # Add to shared keys if sharing is enabled
            if share_with_others:
                context.shared_keys.add(key)
                self._add_to_shared_context(key, value, core_id)
        
        logger.debug(f"Set context '{key}' for core {core_id} (share: {share_with_others})")
    
    def get_core_context(self, key: str, core_id: Optional[int] = None, 
                        check_shared: bool = True) -> Optional[Any]:
        """Get context value from a specific core, optionally checking shared contexts"""
        if core_id is None:
            core_id = self._get_current_core_id()
        
        if core_id is None:
            core_id = 0
        
        # First check local core context
        with self._core_lock:
            if core_id in self._core_contexts:
                context = self._core_contexts[core_id]
                if key in context.context_data:
                    context.update_access_time()
                    return context.context_data[key]
        
        # If not found locally and shared checking is enabled, check shared contexts
        if check_shared:
            return self._get_from_shared_context(key)
        
        return None
    
    def _add_to_shared_context(self, key: str, value: Any, source_core: int) -> None:
        """Add context to shared storage"""
        with self._shared_lock:
            shared_key = f"core_{source_core}_{key}"
            self._shared_contexts[shared_key] = SharedContext(
                key=key,
                value=value,
                source_core=source_core
            )
            logger.debug(f"Added '{key}' to shared context from core {source_core}")
    
    def _get_from_shared_context(self, key: str) -> Optional[Any]:
        """Get value from shared context, returning most recent if multiple cores have it"""
        with self._shared_lock:
            matching_contexts = [
                ctx for shared_key, ctx in self._shared_contexts.items()
                if ctx.key == key
            ]
            
            if matching_contexts:
                # Return the most recently updated context
                latest_context = max(matching_contexts, key=lambda c: c.timestamp)
                latest_context.mark_accessed()
                logger.debug(f"Retrieved '{key}' from shared context (source: core {latest_context.source_core})")
                return latest_context.value
        
        return None
    
    def share_context_between_cores(self, key: str, source_core: int, target_cores: Set[int]) -> bool:
        """Explicitly share context between specific cores"""
        with self._core_lock:
            if source_core not in self._core_contexts:
                return False
            
            source_context = self._core_contexts[source_core]
            if key not in source_context.context_data:
                return False
            
            value = source_context.context_data[key]
            
            # Share with target cores
            for target_core in target_cores:
                if target_core != source_core:
                    target_context = self._ensure_core_context(target_core)
                    target_context.context_data[f"shared_from_{source_core}_{key}"] = value
                    target_context.update_access_time()
            
            logger.debug(f"Shared '{key}' from core {source_core} to cores {target_cores}")
            return True
    
    def get_core_statistics(self) -> Dict[str, Any]:
        """Get statistics about core context usage"""
        with self._core_lock, self._shared_lock:
            stats = {
                "total_cores": len(self._core_contexts),
                "total_shared_contexts": len(self._shared_contexts),
                "cores": {}
            }
            
            for core_id, context in self._core_contexts.items():
                stats["cores"][core_id] = {
                    "context_count": len(context.context_data),
                    "shared_keys": len(context.shared_keys),
                    "last_access": context.last_access,
                    "memory_usage": context.memory_usage
                }
            
            # Add shared context statistics
            shared_stats = {}
            for shared_key, shared_ctx in self._shared_contexts.items():
                key = shared_ctx.key
                if key not in shared_stats:
                    shared_stats[key] = []
                shared_stats[key].append({
                    "source_core": shared_ctx.source_core,
                    "access_count": shared_ctx.access_count,
                    "timestamp": shared_ctx.timestamp
                })
            
            stats["shared_breakdown"] = shared_stats
            return stats
    
    def clear_core_context(self, core_id: int) -> None:
        """Clear all context for a specific core"""
        with self._core_lock:
            if core_id in self._core_contexts:
                del self._core_contexts[core_id]
                logger.debug(f"Cleared context for core {core_id}")
    
    def _cleanup_worker(self):
        """Background thread for cleaning up expired contexts"""
        while not self._shutdown:
            time.sleep(self._cleanup_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired contexts and shared data"""
        current_time = time.time()
        timeout = 3600  # 1 hour timeout for core contexts
        
        with self._core_lock:
            expired_cores = [
                core_id for core_id, context in self._core_contexts.items()
                if current_time - context.last_access > timeout
            ]
            for core_id in expired_cores:
                del self._core_contexts[core_id]
        
        with self._shared_lock:
            expired_shared = [
                key for key, shared_ctx in self._shared_contexts.items()
                if current_time - shared_ctx.timestamp > timeout
            ]
            for key in expired_shared:
                del self._shared_contexts[key]
        
        if expired_cores or expired_shared:
            logger.debug(f"Cleaned up {len(expired_cores)} core contexts and {len(expired_shared)} shared contexts")
    
    def shutdown(self):
        """Graceful shutdown of background threads"""
        self._shutdown = True
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1)


# Global singleton instance
_core_storage_instance = None
_core_storage_lock = threading.Lock()


def get_core_context_storage() -> CoreContextStorage:
    """Get the global core context storage instance (singleton pattern)"""
    global _core_storage_instance
    if _core_storage_instance is None:
        with _core_storage_lock:
            if _core_storage_instance is None:
                _core_storage_instance = CoreContextStorage()
    return _core_storage_instance