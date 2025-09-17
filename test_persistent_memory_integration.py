#!/usr/bin/env python3
"""
Integration test for agent persistent memory functionality

This test verifies that the persistent memory and context management system
works correctly with the per-core agent coordination system.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

from utils.agent_core import Agent, AgentRole, AgentStatus
from utils.agent_persistent_memory import AgentPersistentMemory, get_agent_persistent_memory
from utils.core_context_storage import get_core_context_storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_persistent_memory():
    """Test basic persistent memory functionality"""
    logger.info("Testing basic persistent memory functionality...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test agent with unique ID
        agent_id = f"test_agent_001_{int(time.time())}"
        agent = Agent(
            agent_id=agent_id,
            core_id=0,
            role=AgentRole.SECURITY_ANALYST
        )
        
        # Initialize persistent memory with custom storage directory
        memory = AgentPersistentMemory(agent, storage_dir=temp_dir)
        
        # Test persisting insights
        insight_id = memory.persist_insight(
            content="Detected potential security vulnerability in authentication module",
            importance=0.9,
            confidence=0.85,
            tags={"security", "authentication", "vulnerability"},
            session_context={"module": "auth", "severity": "high"}
        )
        
        assert insight_id, "Failed to persist insight"
        logger.info(f"Successfully persisted insight: {insight_id}")
        
        # Test persisting decisions
        decision_id = memory.persist_decision(
            content="Decided to implement additional input validation for user registration",
            importance=0.8,
            confidence=0.9,
            tags={"security", "validation", "registration"},
            session_context={"action": "implement", "priority": "high"}
        )
        
        assert decision_id, "Failed to persist decision"
        logger.info(f"Successfully persisted decision: {decision_id}")
        
        # Test retrieving memories
        security_memories = memory.retrieve_memories(
            tags={"security"},
            min_importance=0.7,
            limit=10
        )
        
        assert len(security_memories) == 2, f"Expected 2 security memories, got {len(security_memories)}"
        logger.info(f"Successfully retrieved {len(security_memories)} security memories")
        
        # Test context snapshots
        snapshot_id = memory.create_context_snapshot()
        assert snapshot_id, "Failed to create context snapshot"
        logger.info(f"Successfully created context snapshot: {snapshot_id}")
        
        # Test memory statistics
        stats = memory.get_memory_statistics()
        assert stats["total_memory_entries"] == 2, f"Expected 2 memory entries, got {stats['total_memory_entries']}"
        assert stats["total_context_snapshots"] == 1, f"Expected 1 snapshot, got {stats['total_context_snapshots']}"
        logger.info(f"Memory statistics: {stats}")
        
        # Test persistence to disk
        memory_file = Path(temp_dir) / f"agent_{agent.agent_id}_memory.json"
        snapshots_file = Path(temp_dir) / f"agent_{agent.agent_id}_snapshots.json"
        
        assert memory_file.exists(), "Memory file was not created"
        assert snapshots_file.exists(), "Snapshots file was not created"
        logger.info("Memory and snapshots successfully persisted to disk")
        
        # Test loading from disk (simulate restart)
        memory2 = AgentPersistentMemory(agent, storage_dir=temp_dir)
        
        assert len(memory2.memory_entries) == 2, "Failed to load memory entries from disk"
        assert len(memory2.context_snapshots) == 1, "Failed to load snapshots from disk"
        logger.info("Successfully loaded persistent memory from disk")
        
        # Clean up
        memory.shutdown()
        memory2.shutdown()
        
        logger.info("✅ Basic persistent memory test passed")


def test_agent_integration():
    """Test integration with Agent class"""
    logger.info("Testing agent integration...")
    
    # Create test agent with unique ID
    agent_id = f"test_agent_002_{int(time.time())}"
    agent = Agent(
        agent_id=agent_id,
        core_id=1,
        role=AgentRole.PERFORMANCE_OPTIMIZER
    )
    
    # Test agent persistent memory methods
    insight_id = agent.persist_insight(
        content="Identified performance bottleneck in database query optimization",
        importance=0.8,
        tags={"performance", "database", "optimization"}
    )
    
    assert insight_id, "Failed to persist insight through agent"
    logger.info(f"Successfully persisted insight through agent: {insight_id}")
    
    decision_id = agent.persist_decision(
        content="Decided to implement query caching for frequently accessed data",
        importance=0.9,
        tags={"performance", "caching", "database"}
    )
    
    assert decision_id, "Failed to persist decision through agent"
    logger.info(f"Successfully persisted decision through agent: {decision_id}")
    
    # Test retrieving relevant memories
    relevant_memories = agent.retrieve_relevant_memories(
        tags={"performance"},
        limit=5
    )
    
    assert len(relevant_memories) == 2, f"Expected 2 relevant memories, got {len(relevant_memories)}"
    logger.info(f"Successfully retrieved {len(relevant_memories)} relevant memories through agent")
    
    # Test context snapshot creation
    snapshot_id = agent.create_context_snapshot()
    assert snapshot_id, "Failed to create context snapshot through agent"
    logger.info(f"Successfully created context snapshot through agent: {snapshot_id}")
    
    # Test recovery simulation
    # Add some context to the agent
    agent.set_context("current_task", "database_optimization")
    agent.set_context("optimization_level", "aggressive")
    agent.add_thought("analysis", "Database queries are taking too long", 0.8)
    
    # Create another snapshot with context
    snapshot_id2 = agent.create_context_snapshot()
    assert snapshot_id2, "Failed to create second context snapshot"
    
    # Simulate failure recovery
    recovery_success = agent.recover_from_failure()
    assert recovery_success, "Failed to recover from failure"
    logger.info("Successfully recovered agent from failure")
    
    # Verify context was restored
    assert "current_task" in agent.context, "Context was not restored after recovery"
    assert agent.context["current_task"] == "database_optimization", "Context value was not restored correctly"
    logger.info("Context successfully restored after recovery")
    
    logger.info("✅ Agent integration test passed")


def test_core_context_storage_integration():
    """Test integration with CoreContextStorage"""
    logger.info("Testing core context storage integration...")
    
    # Get core context storage
    core_storage = get_core_context_storage()
    
    # Create test agent with unique ID
    agent_id = f"test_agent_003_{int(time.time())}"
    agent = Agent(
        agent_id=agent_id,
        core_id=2,
        role=AgentRole.ARCHITECTURE_REVIEWER
    )
    
    # Test persisting agent memory to core storage
    memory_data = {
        "agent_id": agent.agent_id,
        "insights": ["Architecture needs refactoring", "Microservices pattern recommended"],
        "decisions": ["Implement event-driven architecture"],
        "timestamp": time.time()
    }
    
    core_storage.persist_agent_memory(agent.agent_id, memory_data, core_id=agent.core_id)
    logger.info("Successfully persisted agent memory to core storage")
    
    # Test retrieving agent memory from core storage
    retrieved_memory = core_storage.retrieve_agent_memory(agent.agent_id, core_id=agent.core_id)
    assert retrieved_memory is not None, "Failed to retrieve agent memory from core storage"
    assert retrieved_memory["agent_id"] == agent.agent_id, "Retrieved memory has incorrect agent ID"
    logger.info("Successfully retrieved agent memory from core storage")
    
    # Test sharing insights across cores
    insight_data = {
        "content": "Identified common architectural pattern across multiple services",
        "importance": 0.9,
        "tags": ["architecture", "pattern", "services"]
    }
    
    core_storage.share_agent_insight(agent.agent_id, insight_data, core_id=agent.core_id)
    logger.info("Successfully shared agent insight across cores")
    
    # Test retrieving shared insights
    shared_insights = core_storage.get_shared_insights(max_age_seconds=3600)
    logger.info(f"Retrieved {len(shared_insights)} shared insights")
    
    # Debug: check what's in shared contexts
    stats = core_storage.get_core_statistics()
    logger.info(f"Core storage stats: {stats}")
    
    assert len(shared_insights) >= 1, f"Failed to retrieve shared insights. Got {len(shared_insights)} insights. Stats: {stats}"
    
    found_insight = False
    for insight in shared_insights:
        if insight["agent_id"] == agent.agent_id:
            found_insight = True
            break
    
    assert found_insight, "Shared insight not found in retrieved insights"
    logger.info(f"Successfully retrieved {len(shared_insights)} shared insights")
    
    logger.info("✅ Core context storage integration test passed")


def test_memory_cleanup():
    """Test memory cleanup functionality"""
    logger.info("Testing memory cleanup functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test agent with unique ID
        agent_id = f"test_agent_004_{int(time.time())}"
        agent = Agent(
            agent_id=agent_id,
            core_id=3,
            role=AgentRole.DEBUG_SPECIALIST
        )
        
        # Initialize persistent memory with aggressive cleanup settings
        memory = AgentPersistentMemory(agent, storage_dir=temp_dir)
        memory.max_memory_entries = 5  # Low limit for testing
        memory.importance_threshold = 0.5  # Moderate threshold
        
        # Create multiple memory entries with varying importance
        entries = []
        for i in range(10):
            importance = 0.3 + (i * 0.1)  # 0.3 to 1.2
            entry_id = memory.persist_insight(
                content=f"Test insight {i}",
                importance=min(importance, 1.0),
                confidence=0.8,
                tags={f"test_{i}"}
            )
            if entry_id:  # Only count entries that were actually persisted
                entries.append(entry_id)
        
        # Check that low-importance entries were filtered out
        assert len(memory.memory_entries) <= memory.max_memory_entries, "Memory cleanup did not respect max entries limit"
        logger.info(f"Memory entries after creation: {len(memory.memory_entries)}")
        
        # Trigger manual cleanup
        memory._cleanup_old_entries()
        
        # Verify cleanup worked
        assert len(memory.memory_entries) <= memory.max_memory_entries, "Manual cleanup failed"
        logger.info(f"Memory entries after cleanup: {len(memory.memory_entries)}")
        
        # Verify that higher importance entries were kept
        remaining_entries = list(memory.memory_entries.values())
        if remaining_entries:
            avg_importance = sum(entry.importance for entry in remaining_entries) / len(remaining_entries)
            assert avg_importance >= 0.7, f"Cleanup did not preserve high-importance entries (avg: {avg_importance})"
            logger.info(f"Average importance of remaining entries: {avg_importance:.2f}")
        
        memory.shutdown()
        
        logger.info("✅ Memory cleanup test passed")


def main():
    """Run all integration tests"""
    logger.info("Starting persistent memory integration tests...")
    
    try:
        test_basic_persistent_memory()
        test_agent_integration()
        test_core_context_storage_integration()
        test_memory_cleanup()
        
        logger.info("🎉 All persistent memory integration tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    
    finally:
        # Clean up any global instances
        try:
            from utils.agent_persistent_memory import shutdown_all_agent_memories
            shutdown_all_agent_memories()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


if __name__ == "__main__":
    main()