#!/usr/bin/env python3
"""
Test Agent Communication System

Simple test script to validate that the agent communication system is working
correctly with basic agent creation, communication, and team formation.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.agent_core import Agent, AgentRole, AgentStatus, AgentPersonality
from utils.agent_communication import get_agent_communication_system


async def test_agent_communication():
    """Test basic agent communication functionality"""
    print("🤖 Testing Nexus Agent Communication System")
    print("=" * 50)
    
    # Get the communication system
    comm_system = get_agent_communication_system()
    
    # Create some test agents
    print("\n1. Creating test agents...")
    
    # Register agents with different roles
    security_agent = comm_system.register_agent(core_id=0, role=AgentRole.SECURITY_ANALYST)
    performance_agent = comm_system.register_agent(core_id=1, role=AgentRole.PERFORMANCE_OPTIMIZER)
    architecture_agent = comm_system.register_agent(core_id=2, role=AgentRole.ARCHITECTURE_REVIEWER)
    
    print(f"✓ Created security agent: {security_agent.agent_id}")
    print(f"✓ Created performance agent: {performance_agent.agent_id}")
    print(f"✓ Created architecture agent: {architecture_agent.agent_id}")
    
    # Create a team
    print("\n2. Creating agent team...")
    team_id = comm_system.create_team(
        team_name="Code Analysis Team",
        purpose="Collaborative code analysis and review"
    )
    
    # Add agents to team
    comm_system.add_agent_to_team(security_agent.agent_id, team_id)
    comm_system.add_agent_to_team(performance_agent.agent_id, team_id)
    comm_system.add_agent_to_team(architecture_agent.agent_id, team_id)
    
    print(f"✓ Created team: {team_id}")
    print(f"✓ Added 3 agents to team")
    
    # Agents add some thoughts
    print("\n3. Agents thinking...")
    
    security_agent.add_thought(
        thought_type="analysis",
        content="Analyzing code for potential security vulnerabilities",
        confidence=0.8
    )
    
    performance_agent.add_thought(
        thought_type="analysis", 
        content="Evaluating performance characteristics and bottlenecks",
        confidence=0.85
    )
    
    architecture_agent.add_thought(
        thought_type="insight",
        content="System architecture shows good separation of concerns",
        confidence=0.9
    )
    
    print(f"✓ Security agent added thought")
    print(f"✓ Performance agent added thought")
    print(f"✓ Architecture agent added insight")
    
    # Test inter-agent communication
    print("\n4. Testing agent communication...")
    
    # Security agent sends an alert
    message_id = comm_system.send_message(
        from_agent=security_agent.agent_id,
        to_agent="ALL",
        message_type="alert",
        content="Found potential SQL injection vulnerability in login function",
        priority=9
    )
    
    print(f"✓ Security agent sent alert (message ID: {message_id})")
    
    # Performance agent responds
    response_id = comm_system.send_message(
        from_agent=performance_agent.agent_id,
        to_agent=security_agent.agent_id,
        message_type="response",
        content="The login function is also a performance bottleneck - fixing security issue may improve performance",
        priority=7,
        response_to=message_id
    )
    
    print(f"✓ Performance agent responded (message ID: {response_id})")
    
    # Architecture agent shares insight
    insight_id = comm_system.send_message(
        from_agent=architecture_agent.agent_id,
        to_agent="ALL",
        message_type="insight",
        content="Consider implementing a proper authentication service to address both security and performance concerns",
        priority=8
    )
    
    print(f"✓ Architecture agent shared insight (message ID: {insight_id})")
    
    # Wait a moment for message processing
    await asyncio.sleep(0.5)
    
    # Check agent status
    print("\n5. Agent status summary...")
    
    stats = comm_system.get_agent_statistics()
    
    print(f"✓ Total agents: {stats['total_agents']}")
    print(f"✓ Total teams: {stats['total_teams']}")
    print(f"✓ Total conversations: {stats['total_conversations']}")
    print(f"✓ Message queue size: {stats['message_queue_size']}")
    
    # Show agent roles distribution
    print(f"✓ Agent roles: {stats['roles_distribution']}")
    
    # Show each agent's status
    for agent_id, agent_info in stats['agents'].items():
        role = agent_info['role']
        thought_count = agent_info['thought_count']
        message_count = agent_info['message_count']
        print(f"  - {role}: {thought_count} thoughts, {message_count} messages")
    
    print("\n6. Testing agent message retrieval...")
    
    # Check if agents received messages
    perf_messages = performance_agent.get_unread_messages()
    arch_messages = architecture_agent.get_unread_messages()
    
    print(f"✓ Performance agent has {len(perf_messages)} messages")
    print(f"✓ Architecture agent has {len(arch_messages)} messages")
    
    # Show message details
    for msg in perf_messages:
        print(f"  - From {msg.from_agent}: {msg.content[:50]}...")
    
    print("\n7. Testing agent thoughts...")
    
    # Show recent thoughts from each agent
    for agent in [security_agent, performance_agent, architecture_agent]:
        thoughts = agent.get_recent_thoughts(limit=3)
        print(f"✓ {agent.role.value} has {len(thoughts)} thoughts:")
        for thought in thoughts:
            print(f"  - {thought.thought_type}: {thought.content[:50]}...")
    
    print("\n✅ Agent communication test completed successfully!")
    print("=" * 50)
    
    # Cleanup
    comm_system.shutdown()
    
    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_agent_communication())
        if result:
            print("\n🎉 All tests passed! Agent communication system is working.")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)