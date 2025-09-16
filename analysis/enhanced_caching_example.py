"""
Enhanced Caching Layer for Zen MCP Server

This module demonstrates how the existing caching architecture can be enhanced
without introducing LangChain complexity. It provides distributed caching
capabilities while maintaining the sophisticated agent-based architecture.

This is a minimal implementation example showing how to extend existing
capabilities rather than replacing them with LangChain.
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    access_count: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def mark_accessed(self) -> None:
        """Mark entry as accessed and update statistics"""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheBackend(ABC):
    """Abstract cache backend interface"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass
    
    @abstractmethod
    async def set(self, entry: CacheEntry) -> bool:
        """Set cache entry"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        pass
    
    @abstractmethod
    async def clear_by_tags(self, tags: List[str]) -> int:
        """Clear entries matching any of the given tags"""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                entry.mark_accessed()
                return entry
            elif entry:
                # Remove expired entry
                del self._cache[key]
            return None
    
    async def set(self, entry: CacheEntry) -> bool:
        async with self._lock:
            # Calculate size if not provided
            if entry.size_bytes == 0:
                entry.size_bytes = len(str(entry.value).encode('utf-8'))
            
            # Evict if necessary
            await self._evict_if_needed(entry.size_bytes)
            
            self._cache[entry.key] = entry
            return True
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._cache.pop(key, None) is not None
    
    async def clear_by_tags(self, tags: List[str]) -> int:
        async with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
            
            return len(to_remove)
    
    async def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries to make room for new entry"""
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        
        while (len(self._cache) >= self.max_size or 
               current_memory + new_entry_size > self.max_memory_bytes):
            if not self._cache:
                break
            
            # Find LRU entry
            lru_key = min(self._cache.keys(), 
                         key=lambda k: self._cache[k].accessed_at)
            removed_entry = self._cache.pop(lru_key)
            current_memory -= removed_entry.size_bytes


class EnhancedCacheLayer:
    """
    Enhanced caching layer that integrates with existing Zen MCP architecture
    
    This demonstrates how to enhance the existing caching without LangChain:
    - Supports the existing agent-based architecture
    - Integrates with conversation memory
    - Provides model response caching
    - Maintains provider-specific caching
    """
    
    def __init__(self, backend: CacheBackend):
        self.backend = backend
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    async def cache_model_response(
        self, 
        provider: str, 
        model: str, 
        prompt_hash: str, 
        response: str,
        ttl: float = 3600,  # 1 hour default
        agent_role: Optional[str] = None
    ) -> None:
        """Cache model response with intelligent tagging"""
        
        cache_key = f"model_response:{provider}:{model}:{prompt_hash}"
        tags = [f"provider:{provider}", f"model:{model}"]
        
        if agent_role:
            tags.append(f"agent:{agent_role}")
        
        entry = CacheEntry(
            key=cache_key,
            value=response,
            ttl=ttl,
            tags=tags
        )
        
        await self.backend.set(entry)
        logger.debug(f"Cached model response for {provider}:{model} with agent role {agent_role}")
    
    async def get_cached_response(
        self, 
        provider: str, 
        model: str, 
        prompt_hash: str
    ) -> Optional[str]:
        """Get cached model response"""
        
        cache_key = f"model_response:{provider}:{model}:{prompt_hash}"
        entry = await self.backend.get(cache_key)
        
        self._stats['total_requests'] += 1
        
        if entry:
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for {provider}:{model}")
            return entry.value
        else:
            self._stats['misses'] += 1
            logger.debug(f"Cache miss for {provider}:{model}")
            return None
    
    async def cache_agent_context(
        self, 
        agent_id: str, 
        context_data: Dict[str, Any],
        ttl: float = 1800  # 30 minutes
    ) -> None:
        """Cache agent-specific context data"""
        
        cache_key = f"agent_context:{agent_id}"
        entry = CacheEntry(
            key=cache_key,
            value=context_data,
            ttl=ttl,
            tags=[f"agent:{agent_id}", "agent_context"]
        )
        
        await self.backend.set(entry)
    
    async def get_agent_context(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached agent context"""
        
        cache_key = f"agent_context:{agent_id}"
        entry = await self.backend.get(cache_key)
        return entry.value if entry else None
    
    async def invalidate_agent_cache(self, agent_id: str) -> int:
        """Invalidate all cache entries for a specific agent"""
        return await self.backend.clear_by_tags([f"agent:{agent_id}"])
    
    async def invalidate_provider_cache(self, provider: str) -> int:
        """Invalidate all cache entries for a specific provider"""
        return await self.backend.clear_by_tags([f"provider:{provider}"])
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        hit_rate = (self._stats['hits'] / max(self._stats['total_requests'], 1)) * 100
        return {
            **self._stats,
            'hit_rate_percent': round(hit_rate, 2)
        }


def create_prompt_hash(prompt: str, system_prompt: str = "", **kwargs) -> str:
    """Create deterministic hash for prompt caching"""
    
    # Include all relevant parameters that affect the response
    hash_content = {
        'prompt': prompt,
        'system_prompt': system_prompt,
        'temperature': kwargs.get('temperature', 0.3),
        'max_tokens': kwargs.get('max_tokens'),
        # Add other relevant parameters
    }
    
    content_str = json.dumps(hash_content, sort_keys=True)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


# Integration example with existing architecture
class CacheAwareModelProvider:
    """
    Example showing how to integrate enhanced caching with existing provider architecture
    without disrupting the sophisticated provider registry and agent systems
    """
    
    def __init__(self, base_provider, cache_layer: EnhancedCacheLayer):
        self.base_provider = base_provider
        self.cache_layer = cache_layer
    
    async def generate_content_cached(
        self, 
        prompt: str, 
        model_name: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        agent_role: Optional[str] = None,
        **kwargs
    ):
        """Generate content with intelligent caching"""
        
        # Create cache key
        prompt_hash = create_prompt_hash(
            prompt, 
            system_prompt or "", 
            temperature=temperature, 
            **kwargs
        )
        
        # Try cache first
        cached_response = await self.cache_layer.get_cached_response(
            self.base_provider.get_provider_type().value,
            model_name,
            prompt_hash
        )
        
        if cached_response:
            logger.info(f"Using cached response for {model_name} with agent {agent_role}")
            return cached_response
        
        # Generate new response
        response = self.base_provider.generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            **kwargs
        )
        
        # Cache the response
        await self.cache_layer.cache_model_response(
            provider=self.base_provider.get_provider_type().value,
            model=model_name,
            prompt_hash=prompt_hash,
            response=response.content,
            agent_role=agent_role
        )
        
        return response


# Usage example showing integration with existing architecture
async def example_usage():
    """Example showing how this enhances existing capabilities"""
    
    # Initialize enhanced caching (could be Redis, SQLite, etc.)
    cache_backend = MemoryCacheBackend(max_size=1000, max_memory_mb=50)
    cache_layer = EnhancedCacheLayer(cache_backend)
    
    # This integrates with existing provider registry without disruption
    from providers.registry import ModelProviderRegistry
    
    # Get existing provider (maintains all current sophistication)
    base_provider = ModelProviderRegistry.get_provider_for_model("gemini-2.5-flash")
    
    if base_provider:
        # Wrap with caching without losing existing features
        cached_provider = CacheAwareModelProvider(base_provider, cache_layer)
        
        # Use with agent context (maintains existing agent architecture)
        response = await cached_provider.generate_content_cached(
            prompt="Analyze this code for security issues",
            model_name="gemini-2.5-flash",
            agent_role="security_analyst",
            temperature=0.3
        )
        
        print(f"Cache stats: {cache_layer.get_cache_stats()}")


if __name__ == "__main__":
    # This demonstrates enhancement without architectural disruption
    asyncio.run(example_usage())