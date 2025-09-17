#!/usr/bin/env python3
"""
Demonstration of Agent Persistent Memory and Context Management

This script demonstrates the key features of the persistent memory system
implemented for per-core agent coordination.
"""

import logging
import time
from utils.agent_core import Agent, AgentRole
from utils.agent_persistent_memory import get_agent_persistent_memory
from utils.core_context_storage import get_core_context_storage

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_persistent_memory():
    """Demonstrate the persistent memory system"""
    logger.info("🚀 Starting Persistent Memory Demonstration")
    
    # Create agents with different roles
    security_agent = Agent(
        agent_id="security_agent_demo",
        core_id=0,
        role=AgentRole.SECURITY_ANALYST
    )
    
    performance_agent = Agent(
        agent_id="performance_agent_demo", 
        core_id=1,
        role=AgentRole.PERFORMANCE_OPTIMIZER
    )
    
    logger.info(f"Created agents: {security_agent.agent_id} and {performance_agent.agent_id}")
    
    # Demonstrate insight persistence
    logger.info("\n📝 Demonstrating Insight Persistence")
    
    security_insight_id = security_agent.persist_insight(
        content="Detected potential SQL injection vulnerability in user input validation",
        importance=0.95,
        tags={"security", "sql_injection", "vulnerability", "critical"}
    )
    
    performance_insight_id = performance_agent.persist_insight(
        content="Database connection pooling could improve response times by 40%",
        importance=0.85,
        tags={"performance", "database", "optimization", "connection_pooling"}
    )
    
    logger.info(f"Security insight persisted: {security_insight_id}")
    logger.info(f"Performance insight persisted: {performance_insight_id}")
    
    # Demonstrate decision persistence
    logger.info("\n🎯 Demonstrating Decision Persistence")
    
    security_decision_id = security_agent.persist_decision(
        content="Implement parameterized queries and input sanitization for all user inputs",
        importance=0.9,
        tags={"security", "implementation", "sql_injection", "mitigation"}
    )
    
    performance_decision_id = performance_agent.persist_decision(
        content="Increase database connection pool size from 10 to 25 connections",
        importance=0.8,
        tags={"performance", "database", "configuration", "connection_pool"}
    )
    
    logger.info(f"Security decision persisted: {security_decision_id}")
    logger.info(f"Performance decision persisted: {performance_decision_id}")
    
    # Demonstrate memory retrieval
    logger.info("\n🔍 Demonstrating Memory Retrieval")
    
    security_memories = security_agent.retrieve_relevant_memories(
        tags={"security"},
        limit=10
    )
    
    performance_memories = performance_agent.retrieve_relevant_memories(
        tags={"performance"},
        limit=10
    )
    
    logger.info(f"Security agent retrieved {len(security_memories)} relevant memories")
    for memory in security_memories:
        logger.info(f"  - {memory.entry_type}: {memory.content[:60]}... (importance: {memory.importance})")
    
    logger.info(f"Performance agent retrieved {len(performance_memories)} relevant memories")
    for memory in performance_memories:
        logger.info(f"  - {memory.entry_type}: {memory.content[:60]}... (importance: {memory.importance})")
    
    # Demonstrate context snapshots
    logger.info("\n📸 Demonstrating Context Snapshots")
    
    # Add some context to agents
    security_agent.set_context("current_analysis", "authentication_module")
    security_agent.set_context("threat_level", "high")
    security_agent.add_thought("analysis", "Multiple vulnerabilities found in auth module", 0.9)
    
    performance_agent.set_context("current_optimization", "database_layer")
    performance_agent.set_context("target_improvement", "40%")
    performance_agent.add_thought("analysis", "Connection pooling shows most promise", 0.8)
    
    # Create snapshots
    security_snapshot_id = security_agent.create_context_snapshot()
    performance_snapshot_id = performance_agent.create_context_snapshot()
    
    logger.info(f"Security agent snapshot created: {security_snapshot_id}")
    logger.info(f"Performance agent snapshot created: {performance_snapshot_id}")
    
    # Demonstrate recovery simulation
    logger.info("\n🔄 Demonstrating Failure Recovery")
    
    # Simulate agent failure by clearing context
    original_security_context = security_agent.context.copy()
    security_agent.context.clear()
    security_agent.thoughts.clear()
    
    logger.info("Simulated security agent failure (cleared context and thoughts)")
    logger.info(f"Context before recovery: {security_agent.context}")
    
    # Attempt recovery
    recovery_success = security_agent.recover_from_failure()
    
    if recovery_success:
        logger.info("✅ Security agent successfully recovered from failure!")
        logger.info(f"Context after recovery: {security_agent.context}")
        logger.info(f"Thoughts after recovery: {len(security_agent.thoughts)} thoughts restored")
    else:
        logger.error("❌ Security agent recovery failed")
    
    # Demonstrate core context storage integration
    logger.info("\n🌐 Demonstrating Core Context Storage Integration")
    
    core_storage = get_core_context_storage()
    
    # Share insights across cores
    core_storage.share_agent_insight(
        agent_id=security_agent.agent_id,
        insight_data={
            "content": "Critical security pattern identified across multiple modules",
            "importance": 0.95,
            "affected_modules": ["auth", "user_management", "api_gateway"]
        },
        core_id=security_agent.core_id
    )
    
    core_storage.share_agent_insight(
        agent_id=performance_agent.agent_id,
        insight_data={
            "content": "Performance optimization pattern applicable system-wide",
            "importance": 0.85,
            "potential_improvement": "35-50%",
            "applicable_areas": ["database", "caching", "api_responses"]
        },
        core_id=performance_agent.core_id
    )
    
    # Retrieve shared insights
    shared_insights = core_storage.get_shared_insights(max_age_seconds=3600)
    
    logger.info(f"Retrieved {len(shared_insights)} shared insights from core storage:")
    for insight in shared_insights:
        logger.info(f"  - From {insight['agent_id']}: {insight['insight']['content'][:60]}...")
    
    # Demonstrate memory statistics
    logger.info("\n📊 Demonstrating Memory Statistics")
    
    security_memory = get_agent_persistent_memory(security_agent)
    performance_memory = get_agent_persistent_memory(performance_agent)
    
    security_stats = security_memory.get_memory_statistics()
    performance_stats = performance_memory.get_memory_statistics()
    
    logger.info(f"Security agent memory statistics:")
    logger.info(f"  - Total entries: {security_stats['total_memory_entries']}")
    logger.info(f"  - Total snapshots: {security_stats['total_context_snapshots']}")
    logger.info(f"  - Entry types: {security_stats['entry_types']}")
    logger.info(f"  - Average importance: {security_stats['average_importance']:.2f}")
    
    logger.info(f"Performance agent memory statistics:")
    logger.info(f"  - Total entries: {performance_stats['total_memory_entries']}")
    logger.info(f"  - Total snapshots: {performance_stats['total_context_snapshots']}")
    logger.info(f"  - Entry types: {performance_stats['entry_types']}")
    logger.info(f"  - Average importance: {performance_stats['average_importance']:.2f}")
    
    # Demonstrate cross-agent memory sharing
    logger.info("\n🤝 Demonstrating Cross-Agent Memory Sharing")
    
    # Security agent shares a pattern that might be relevant to performance
    security_agent.persist_insight(
        content="Authentication bottleneck identified - multiple redundant validation calls",
        importance=0.8,
        tags={"security", "performance", "authentication", "bottleneck"}
    )
    
    # Performance agent looks for insights related to performance
    cross_relevant_memories = performance_agent.retrieve_relevant_memories(
        tags={"performance"},
        limit=10
    )
    
    logger.info(f"Performance agent found {len(cross_relevant_memories)} performance-related memories")
    for memory in cross_relevant_memories:
        if "authentication" in memory.content.lower():
            logger.info(f"  - Cross-domain insight: {memory.content}")
    
    # Clean up
    logger.info("\n🧹 Cleaning Up")
    security_memory.shutdown()
    performance_memory.shutdown()
    
    logger.info("🎉 Persistent Memory Demonstration Complete!")


if __name__ == "__main__":
    demonstrate_persistent_memory()