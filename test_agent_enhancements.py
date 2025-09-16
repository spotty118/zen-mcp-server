#!/usr/bin/env python3
"""
Test script for new agent enhancements

This script tests the new agent API calling capabilities, 
synchronized thinking, and automatic agent selection features.
"""

import asyncio
import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.agent_core import AgentRole
from utils.agent_communication import get_agent_communication_system
from utils.automatic_agent_selector import get_automatic_agent_selector, TaskCharacteristics, TaskType, TaskComplexity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_agent_enhancements():
    """Test the new agent enhancement features"""
    
    logger.info("Testing Agent Enhancements")
    logger.info("=" * 50)
    
    # 1. Test Agent Communication System
    logger.info("1. Testing Agent Communication System...")
    comm_system = get_agent_communication_system()
    
    # Register some test agents
    agent1 = comm_system.register_agent(0, AgentRole.SECURITY_ANALYST, "test_security_agent")
    agent2 = comm_system.register_agent(1, AgentRole.PERFORMANCE_OPTIMIZER, "test_performance_agent") 
    agent3 = comm_system.register_agent(2, AgentRole.ARCHITECTURE_REVIEWER, "test_architecture_agent")
    
    logger.info(f"Registered 3 test agents: {[agent1.agent_id, agent2.agent_id, agent3.agent_id]}")
    
    # 2. Test Automatic Agent Selection
    logger.info("\n2. Testing Automatic Agent Selection...")
    agent_selector = get_automatic_agent_selector(comm_system)
    
    # Test system info
    system_info = agent_selector.get_system_info()
    logger.info(f"System capabilities: {system_info}")
    
    # Test task analysis
    test_prompt = "Please review this code for security vulnerabilities and performance issues"
    task_chars = agent_selector.analyze_task_from_prompt(test_prompt, ["file1.py", "file2.py"])
    logger.info(f"Analyzed task: {task_chars.task_type.value}, complexity: {task_chars.complexity.value}")
    
    # Test agent selection
    selected_agents, coordinator = agent_selector.select_agents_for_task(task_chars)
    logger.info(f"Selected agents: {selected_agents}")
    logger.info(f"Coordinator: {coordinator}")
    
    # 3. Test Agent API Clients
    logger.info("\n3. Testing Agent API Clients...")
    
    # Get API client for an agent
    api_client = comm_system.get_agent_api_client(agent1.agent_id)
    if api_client:
        logger.info(f"API client created for {agent1.agent_id}")
        logger.info(f"Preferred providers: {[p.value for p in api_client.preferred_providers]}")
        
        # Test call statistics (should be empty initially)
        stats = api_client.get_call_statistics()
        logger.info(f"Initial API call stats: {stats}")
    else:
        logger.warning(f"No API client found for {agent1.agent_id}")
    
    # 4. Test Synchronized Thinking (setup only, no actual API calls)
    logger.info("\n4. Testing Synchronized Thinking Setup...")
    
    participant_agents = [agent1.agent_id, agent2.agent_id, agent3.agent_id]
    session_id = comm_system.start_synchronized_thinking(
        participating_agents=participant_agents,
        thinking_topic="Test security and performance analysis",
        coordinator_agent=agent3.agent_id  # Architecture reviewer as coordinator
    )
    
    logger.info(f"Started synchronized thinking session: {session_id}")
    
    # Check session status
    session_status = comm_system.get_thinking_session_status(session_id)
    if session_status:
        logger.info(f"Session status: {session_status}")
    
    # 5. Test Agent Communication
    logger.info("\n5. Testing Agent Communication...")
    
    # Send some test messages
    message_id = comm_system.send_message(
        from_agent=agent1.agent_id,
        to_agent=agent2.agent_id,
        message_type="insight",
        content="Found potential security issue in authentication module",
        priority=7
    )
    logger.info(f"Sent message from {agent1.agent_id} to {agent2.agent_id}: {message_id}")
    
    # Check agent thoughts
    agent1_thoughts = agent1.get_recent_thoughts(limit=5)
    logger.info(f"Agent {agent1.agent_id} recent thoughts: {len(agent1_thoughts)}")
    for thought in agent1_thoughts:
        logger.info(f"  - {thought.thought_type}: {thought.content[:50]}...")
    
    # 6. Test Statistics
    logger.info("\n6. Testing System Statistics...")
    
    stats = comm_system.get_agent_statistics()
    logger.info(f"Total agents: {stats['total_agents']}")
    logger.info(f"Role distribution: {stats['roles_distribution']}")
    logger.info(f"Total conversations: {stats['total_conversations']}")
    
    logger.info("\n✅ All tests completed successfully!")
    
    # Cleanup
    logger.info("\nCleaning up...")
    comm_system.shutdown()
    logger.info("Cleanup complete")


def test_shortened_command():
    """Test that the shortened zen command works"""
    logger.info("Testing shortened zen command...")
    
    # The zen command should be executable and show help
    import subprocess
    try:
        result = subprocess.run(["/home/runner/work/zen-mcp-server/zen-mcp-server/zen", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("✅ Zen command works correctly")
            logger.info(f"Output: {result.stdout.strip()}")
        else:
            logger.error(f"❌ Zen command failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Error testing zen command: {e}")


if __name__ == "__main__":
    print("🤖 Testing Zen MCP Server Agent Enhancements")
    print("=" * 50)
    
    # Test shortened command first
    test_shortened_command()
    
    print("\n" + "=" * 50)
    
    # Test async functionality
    try:
        asyncio.run(test_agent_enhancements())
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()