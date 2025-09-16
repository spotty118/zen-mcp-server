#!/usr/bin/env python3
"""
Zen MCP Server Agent Enhancements Demo

This script demonstrates the new agent enhancement features:
1. Shortened zen command for server startup
2. Agents making their own API calls
3. Synchronized thinking between agents
4. Automatic agent selection based on task and core count

Usage: python agent_enhancements_demo.py
"""

import asyncio
import os
import sys
import logging
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.agent_core import AgentRole
from utils.agent_communication import get_agent_communication_system
from utils.automatic_agent_selector import (
    get_automatic_agent_selector, 
    TaskCharacteristics, 
    TaskType, 
    TaskComplexity
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_shortened_command():
    """Demonstrate the shortened zen command"""
    print("\n🚀 DEMO 1: Shortened Server Command")
    print("=" * 50)
    
    zen_path = project_root / "zen"
    if zen_path.exists():
        print("✅ Zen command exists at:", zen_path)
        
        # Test the zen command
        try:
            result = subprocess.run([str(zen_path), "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Zen command works! Version: {result.stdout.strip()}")
                print(f"📝 Usage: ./zen [OPTIONS] instead of ./run-server.sh [OPTIONS]")
            else:
                print(f"❌ Zen command failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error testing zen command: {e}")
    else:
        print("❌ Zen command not found")


async def demo_agent_api_calls():
    """Demonstrate agents making their own API calls"""
    print("\n🤖 DEMO 2: Agent Direct API Calls")
    print("=" * 50)
    
    # Get the agent communication system
    comm_system = get_agent_communication_system()
    
    # Register some agents with different roles
    security_agent = comm_system.register_agent(0, AgentRole.SECURITY_ANALYST, "demo_security")
    performance_agent = comm_system.register_agent(1, AgentRole.PERFORMANCE_OPTIMIZER, "demo_performance")
    
    print(f"✅ Registered agents: {security_agent.agent_id}, {performance_agent.agent_id}")
    
    # Get their API clients
    security_client = comm_system.get_agent_api_client(security_agent.agent_id)
    performance_client = comm_system.get_agent_api_client(performance_agent.agent_id)
    
    if security_client and performance_client:
        print(f"✅ Security agent preferred providers: {[p.value for p in security_client.preferred_providers]}")
        print(f"✅ Performance agent preferred providers: {[p.value for p in performance_client.preferred_providers]}")
        
        # Show initial statistics
        sec_stats = security_client.get_call_statistics()
        perf_stats = performance_client.get_call_statistics()
        
        print(f"📊 Security agent API stats: {sec_stats['total_calls']} calls, {sec_stats['success_rate']:.1%} success rate")
        print(f"📊 Performance agent API stats: {perf_stats['total_calls']} calls, {perf_stats['success_rate']:.1%} success rate")
        
        print("💡 Note: Agents can now make independent API calls to their preferred providers!")
        
    else:
        print("❌ Could not get API clients for agents")
    
    # Cleanup
    comm_system.shutdown()


async def demo_automatic_agent_selection():
    """Demonstrate automatic agent selection based on task characteristics"""
    print("\n🎯 DEMO 3: Automatic Agent Selection")
    print("=" * 50)
    
    # Get communication system and agent selector
    comm_system = get_agent_communication_system()
    agent_selector = get_automatic_agent_selector(comm_system)
    
    # Show system capabilities
    system_info = agent_selector.get_system_info()
    print(f"🖥️  System: {system_info['total_cores']} cores, {system_info['memory_gb']:.1f}GB RAM")
    print(f"⚡ Performance tier: {'High' if system_info['is_high_performance'] else 'Standard'}")
    print(f"🤖 Recommended max agents: {system_info['recommended_max_agents']}")
    
    # Test different task types
    test_tasks = [
        ("Review this authentication code for security vulnerabilities", ["auth.py"], TaskType.SECURITY_ANALYSIS),
        ("Analyze the performance bottlenecks in this web application", ["app.py", "db.py", "cache.py"], TaskType.PERFORMANCE_OPTIMIZATION),
        ("Design a microservices architecture for this monolithic application", ["monolith.py"], TaskType.ARCHITECTURE_DESIGN),
        ("Debug this intermittent crash in the payment processing system", ["payment.py"], TaskType.DEBUGGING),
        ("Perform comprehensive code review of this entire project", ["src/", "tests/", "docs/"], TaskType.CODE_REVIEW),
    ]
    
    for prompt, files, expected_task_type in test_tasks:
        print(f"\n📋 Task: {prompt[:50]}...")
        print(f"📁 Files: {len(files)} file(s)")
        
        # Analyze task characteristics
        task_chars = agent_selector.analyze_task_from_prompt(prompt, files)
        print(f"🔍 Detected: {task_chars.task_type.value} ({task_chars.complexity.value} complexity)")
        
        # Select agents
        selected_agents, coordinator = agent_selector.select_agents_for_task(task_chars)
        print(f"🤖 Selected {len(selected_agents)} agents")
        print(f"👨‍💼 Coordinator: {coordinator or 'None'}")
        
        # Verify the task type detection
        status = "✅" if task_chars.task_type == expected_task_type else "⚠️ "
        print(f"{status} Expected {expected_task_type.value}, got {task_chars.task_type.value}")
    
    # Cleanup
    comm_system.shutdown()


async def demo_synchronized_thinking():
    """Demonstrate synchronized thinking between agents"""
    print("\n🧠 DEMO 4: Synchronized Thinking")
    print("=" * 50)
    
    # Get communication system
    comm_system = get_agent_communication_system()
    
    # Register multiple agents for collaboration
    agents = []
    roles = [AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.ARCHITECTURE_REVIEWER]
    
    for i, role in enumerate(roles):
        agent = comm_system.register_agent(i, role, f"sync_demo_{role.value}")
        agents.append(agent)
    
    print(f"✅ Registered {len(agents)} agents for synchronized thinking:")
    for agent in agents:
        print(f"   🤖 {agent.agent_id} ({agent.role.value})")
    
    # Start a synchronized thinking session
    participant_ids = [agent.agent_id for agent in agents]
    session_id = comm_system.start_synchronized_thinking(
        participating_agents=participant_ids,
        thinking_topic="Analyze security and performance of authentication system",
        phases=["analysis", "synthesis", "consensus"],
        phase_timeout=30.0  # Short timeout for demo
    )
    
    print(f"🚀 Started synchronized thinking session: {session_id}")
    
    # Check session status
    session_status = comm_system.get_thinking_session_status(session_id)
    if session_status:
        print(f"📊 Session status: {session_status['current_phase']} phase")
        print(f"👥 Participants: {len(session_status['participating_agents'])} agents")
        print(f"👨‍💼 Coordinator: {session_status['coordinator_agent']}")
        print(f"⏰ Phase timeout: {session_status['phase_timeout']} seconds")
    
    # Show agent thoughts
    print(f"\n💭 Agent thoughts:")
    for agent in agents:
        thoughts = agent.get_recent_thoughts(limit=2)
        for thought in thoughts:
            print(f"   {agent.role.value}: {thought.content}")
    
    print("💡 Note: In real usage, agents would execute API calls and share insights across phases!")
    
    # Cleanup
    comm_system.shutdown()


def demo_integration_example():
    """Show how all features work together"""
    print("\n🌟 DEMO 5: Complete Integration Example")
    print("=" * 50)
    
    print("🎯 Complete Workflow:")
    print("1. Start server with shortened command: ./zen")
    print("2. User submits complex task (e.g., 'review security of payment system')")
    print("3. System analyzes task → Security Analysis, High Complexity")
    print("4. Auto-selects optimal agents: Security Analyst + Architecture Reviewer")
    print("5. Agents make direct API calls to their preferred providers")
    print("6. Synchronized thinking: Analysis → Synthesis → Consensus phases")
    print("7. Agents communicate insights and coordinate responses")
    print("8. Final unified response delivered to user")
    
    print("\n🔄 Agent Autonomy Features:")
    print("✅ Each agent has independent API calling capabilities")
    print("✅ Role-based provider preferences (Security → OpenAI, Performance → Google)")
    print("✅ Per-agent rate limiting and error handling")
    print("✅ Agent-to-agent communication and insight sharing")
    print("✅ Automatic team formation and coordination")
    print("✅ CPU core optimization based on system capabilities")
    
    print("\n⚡ Performance Benefits:")
    print("• Parallel API calls reduce overall response time")
    print("• Specialized agents provide domain-specific expertise")
    print("• Automatic scaling based on system resources")
    print("• Intelligent load balancing across available cores")


async def main():
    """Run all demonstrations"""
    print("🤖 Zen MCP Server - Agent Enhancements Demo")
    print("=" * 60)
    print("This demo showcases the new agent enhancement features:")
    print("• Shortened server startup command")
    print("• Agents making direct API calls")
    print("• Synchronized thinking coordination") 
    print("• Automatic agent selection")
    print("• CPU core count optimization")
    
    try:
        # Run all demos
        demo_shortened_command()
        await demo_agent_api_calls()
        await demo_automatic_agent_selection()
        await demo_synchronized_thinking()
        demo_integration_example()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 To use these features:")
        print("   ./zen --help                    # Start server with shortened command")
        print("   python test_agent_enhancements.py  # Run comprehensive tests")
        
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())