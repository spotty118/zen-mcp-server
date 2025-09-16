"""Tests for the agent communication system."""

import asyncio
import os
import sys
from typing import Set

# Ensure the project root is available on the Python path when executed directly
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import agent_communication as agent_comm_module
from utils.agent_core import AgentRole


async def _run_agent_communication_flow() -> None:
    """Verify agents can collaborate, form teams, and exchange messages."""
    # Start from a clean singleton instance for deterministic assertions
    if agent_comm_module._communication_system_instance is not None:
        agent_comm_module._communication_system_instance.shutdown()
        agent_comm_module._communication_system_instance = None

    comm_system = agent_comm_module.get_agent_communication_system()

    try:
        security_agent = comm_system.register_agent(core_id=0, role=AgentRole.SECURITY_ANALYST)
        performance_agent = comm_system.register_agent(core_id=1, role=AgentRole.PERFORMANCE_OPTIMIZER)
        architecture_agent = comm_system.register_agent(core_id=2, role=AgentRole.ARCHITECTURE_REVIEWER)

        agent_ids: Set[str] = {
            security_agent.agent_id,
            performance_agent.agent_id,
            architecture_agent.agent_id,
        }

        team_id = comm_system.create_team(
            team_name="Code Analysis Team",
            purpose="Collaborative code analysis and review",
        )

        assert comm_system.add_agent_to_team(security_agent.agent_id, team_id)
        assert comm_system.add_agent_to_team(performance_agent.agent_id, team_id)
        assert comm_system.add_agent_to_team(architecture_agent.agent_id, team_id)

        assert security_agent.add_thought(
            thought_type="analysis",
            content="Analyzing code for potential security vulnerabilities",
            confidence=0.8,
        )
        assert performance_agent.add_thought(
            thought_type="analysis",
            content="Evaluating performance characteristics and bottlenecks",
            confidence=0.85,
        )
        assert architecture_agent.add_thought(
            thought_type="insight",
            content="System architecture shows good separation of concerns",
            confidence=0.9,
        )

        alert_id = comm_system.send_message(
            from_agent=security_agent.agent_id,
            to_agent="ALL",
            message_type="alert",
            content="Found potential SQL injection vulnerability in login function",
            priority=9,
        )
        assert alert_id is not None

        response_id = comm_system.send_message(
            from_agent=performance_agent.agent_id,
            to_agent=security_agent.agent_id,
            message_type="response",
            content=(
                "The login function is also a performance bottleneck - fixing security issue may "
                "improve performance"
            ),
            priority=7,
            response_to=alert_id,
        )
        assert response_id is not None

        insight_id = comm_system.send_message(
            from_agent=architecture_agent.agent_id,
            to_agent="ALL",
            message_type="insight",
            content=(
                "Consider implementing a proper authentication service to address both security and "
                "performance concerns"
            ),
            priority=8,
        )
        assert insight_id is not None

        # Allow background threads to deliver queued messages
        await asyncio.sleep(0.5)

        stats = comm_system.get_agent_statistics()

        assert stats["total_agents"] == 3
        assert stats["total_teams"] == 1
        assert stats["total_conversations"] == 2
        assert stats["message_queue_size"] == 0

        assert stats["roles_distribution"] == {
            AgentRole.SECURITY_ANALYST.value: 1,
            AgentRole.PERFORMANCE_OPTIMIZER.value: 1,
            AgentRole.ARCHITECTURE_REVIEWER.value: 1,
        }

        assert team_id in stats["teams"]
        team_info = stats["teams"][team_id]
        assert set(team_info["members"]) == agent_ids
        assert team_info["team_name"] == "Code Analysis Team"
        assert team_info["purpose"] == "Collaborative code analysis and review"

        security_messages = security_agent.get_unread_messages()
        performance_messages = performance_agent.get_unread_messages()
        architecture_messages = architecture_agent.get_unread_messages()

        assert {msg.message_type for msg in security_messages} == {"response", "insight"}
        assert any(msg.response_to == alert_id for msg in security_messages if msg.message_type == "response")

        assert {msg.message_type for msg in performance_messages} == {"alert", "insight"}
        assert any("SQL injection" in msg.content for msg in performance_messages)

        assert {msg.message_type for msg in architecture_messages} == {"alert"}
        assert any(msg.from_agent == security_agent.agent_id for msg in architecture_messages)

        security_thoughts = [thought.content for thought in security_agent.get_recent_thoughts()]
        performance_thoughts = [thought.content for thought in performance_agent.get_recent_thoughts()]
        architecture_thoughts = [thought.content for thought in architecture_agent.get_recent_thoughts()]

        assert any("Analyzing code for potential security vulnerabilities" in content for content in security_thoughts)
        assert any(
            "Evaluating performance characteristics and bottlenecks" in content
            for content in performance_thoughts
        )
        assert any(
            "System architecture shows good separation of concerns" in content
            for content in architecture_thoughts
        )

    finally:
        comm_system.shutdown()
        agent_comm_module._communication_system_instance = None


def test_agent_communication_basic_flow() -> None:
    """Wrapper so pytest can execute the async test logic."""
    asyncio.run(_run_agent_communication_flow())
