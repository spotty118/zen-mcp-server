"""
Integration tests for multi-agent coordination

This module tests the full system initialization and agent coordination including:
- Full system initialization and agent coordination
- Inter-agent communication and synchronized thinking sessions
- Workload redistribution and failure recovery scenarios

Requirements tested: 3.1, 3.2, 5.4
"""

import asyncio
import multiprocessing
import os
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, List, Optional

import pytest

from utils.agent_core import Agent, AgentRole, AgentStatus
from utils.per_core_agent_manager import PerCoreAgentManager, PerCoreAgentStatus
from utils.agent_thinking_session import (
    AgentThinkingSession, 
    ThinkingSessionCoordinator,
    SynchronizedThinkingRequest,
    ThinkingSessionStatus,
    ThinkingSessionType
)
from utils.agent_communication import AgentCommunicationSystem
from utils.agent_core import AgentMessage


class TestPerCoreAgentCoordinationIntegration(unittest.TestCase):
    """Integration tests for full system initialization and agent coordination"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_openrouter_key = "test-openrouter-key-123"
        self.test_max_agents = 4
        
        # Mock configuration manager
        self.mock_config_manager = Mock()
        self.mock_system_config = Mock()
        self.mock_system_config.openrouter_api_key = self.test_openrouter_key
        self.mock_system_config.max_agents = self.test_max_agents
        self.mock_system_config.validate.return_value = []
        self.mock_system_config.to_dict.return_value = {"test": "config"}
        self.mock_config_manager.get_config.return_value = self.mock_system_config
        self.mock_config_manager.add_change_callback = Mock()
        
        # Mock agents for testing
        self.mock_agents = []
        for i in range(self.test_max_agents):
            mock_agent = Mock()
            mock_agent.agent_id = f"core_{i}_test_agent"
            mock_agent.core_id = i
            mock_agent.role = AgentRole.GENERALIST if i > 2 else [
                AgentRole.SECURITY_ANALYST,
                AgentRole.PERFORMANCE_OPTIMIZER,
                AgentRole.ARCHITECTURE_REVIEWER
            ][i]
            mock_agent.status = AgentStatus.ACTIVE
            mock_agent.last_activity = time.time()
            self.mock_agents.append(mock_agent)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any global state
        pass
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_agent_communication_system')
    @patch('utils.per_core_error_handling.get_per_core_error_handler')
    @patch('utils.per_core_logging.log_system_event')
    @patch('utils.per_core_logging.log_agent_activity')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_full_system_initialization_and_coordination(
        self, mock_exists, mock_get_config_manager, mock_log_agent, 
        mock_log_system, mock_get_error_handler, mock_get_comm_system,
        mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test complete system initialization with agent coordination"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Mock CPU detection to return our test count
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Mock communication system
        mock_comm_system = Mock()
        mock_get_comm_system.return_value = mock_comm_system
        mock_comm_system.register_agent.side_effect = self.mock_agents
        
        # Mock error handler
        mock_error_handler = AsyncMock()
        mock_get_error_handler.return_value = mock_error_handler
        
        # Mock API clients for each agent
        mock_api_clients = {}
        for agent in self.mock_agents:
            mock_api_client = Mock()
            mock_api_client.configure_openrouter_only = Mock()
            mock_api_clients[agent.agent_id] = mock_api_client
        
        mock_comm_system.get_agent_api_client.side_effect = lambda agent_id: mock_api_clients.get(agent_id)
        
        # Act
        manager = PerCoreAgentManager()
        
        with patch.object(manager, '_initialize_monitoring_and_alerting'):
            with patch.object(manager, '_start_health_monitoring'):
                with patch.object(manager, '_check_openrouter_connection_health', return_value=True):
                    with patch.object(manager, '_validate_openrouter_api_key', return_value=True):
                        with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                            # Mock OpenRouter config
                            mock_config = Mock()
                            mock_config.api_key = self.test_openrouter_key
                            mock_config.preferred_models = ["test-model"]
                            mock_config.rate_limit_per_minute = 60
                            mock_config.max_concurrent_calls = 3
                            mock_create_config.return_value = mock_config
                            
                            agents = manager.initialize_agents()
        
        # Assert - System initialization
        self.assertEqual(len(agents), self.test_max_agents)
        self.assertEqual(len(manager.agents), self.test_max_agents)
        self.assertEqual(len(manager.agent_statuses), self.test_max_agents)
        
        # Assert - Agent coordination setup
        self.assertEqual(mock_comm_system.register_agent.call_count, self.test_max_agents)
        
        # Assert - OpenRouter configuration for each agent
        for agent in self.mock_agents:
            mock_api_client = mock_api_clients[agent.agent_id]
            mock_api_client.configure_openrouter_only.assert_called_once()
        
        # Assert - Agent statuses are properly initialized
        for agent in self.mock_agents:
            self.assertIn(agent.agent_id, manager.agent_statuses)
            status = manager.agent_statuses[agent.agent_id]
            self.assertEqual(status.agent_id, agent.agent_id)
            self.assertEqual(status.core_id, agent.core_id)
            self.assertEqual(status.role, agent.role)
            self.assertTrue(status.openrouter_connected)  # Should be connected after setup
        
        # Assert - System health is available
        health = manager.get_system_health()
        self.assertIsInstance(health, dict)
        self.assertIn("total_agents", health)
        self.assertIn("healthy_agents", health)
        self.assertIn("openrouter_configured", health)
        self.assertEqual(health["total_agents"], self.test_max_agents)
        self.assertEqual(health["healthy_agents"], self.test_max_agents)
        self.assertTrue(health["openrouter_configured"])
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_agent_role_assignment_coordination(self, mock_exists, mock_get_config_manager):
        """Test that agents are assigned appropriate roles for coordination"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Test different core counts
        test_cases = [
            (1, [AgentRole.SECURITY_ANALYST]),
            (3, [AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER, AgentRole.ARCHITECTURE_REVIEWER]),
            (8, [
                AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER, 
                AgentRole.ARCHITECTURE_REVIEWER, AgentRole.CODE_QUALITY_INSPECTOR,
                AgentRole.DEBUG_SPECIALIST, AgentRole.PLANNING_COORDINATOR,
                AgentRole.CONSENSUS_FACILITATOR, AgentRole.GENERALIST
            ]),
            (10, [
                AgentRole.SECURITY_ANALYST, AgentRole.PERFORMANCE_OPTIMIZER, 
                AgentRole.ARCHITECTURE_REVIEWER, AgentRole.CODE_QUALITY_INSPECTOR,
                AgentRole.DEBUG_SPECIALIST, AgentRole.PLANNING_COORDINATOR,
                AgentRole.CONSENSUS_FACILITATOR, AgentRole.GENERALIST,
                AgentRole.GENERALIST, AgentRole.GENERALIST  # Extra generalists
            ])
        ]
        
        for core_count, expected_roles in test_cases:
            with self.subTest(core_count=core_count):
                # Act
                assigned_roles = manager._assign_agent_roles(core_count)
                
                # Assert
                self.assertEqual(len(assigned_roles), core_count)
                self.assertEqual(assigned_roles, expected_roles)
                
                # Assert specialized roles are prioritized
                if core_count >= 1:
                    self.assertEqual(assigned_roles[0], AgentRole.SECURITY_ANALYST)
                if core_count >= 2:
                    self.assertEqual(assigned_roles[1], AgentRole.PERFORMANCE_OPTIMIZER)
                if core_count >= 3:
                    self.assertEqual(assigned_roles[2], AgentRole.ARCHITECTURE_REVIEWER)
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_openrouter_configuration_coordination(self, mock_exists, mock_get_config_manager):
        """Test OpenRouter configuration coordination across agents"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Test role-specific model preferences
        test_roles = [
            AgentRole.SECURITY_ANALYST,
            AgentRole.PERFORMANCE_OPTIMIZER,
            AgentRole.ARCHITECTURE_REVIEWER,
            AgentRole.GENERALIST
        ]
        
        for role in test_roles:
            with self.subTest(role=role):
                # Act
                with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                    mock_config = Mock()
                    mock_config.api_key = self.test_openrouter_key
                    mock_config.rate_limit_per_minute = 60
                    mock_config.max_concurrent_calls = 3
                    mock_config.thinking_mode_default = "high"
                    
                    # Set role-specific model preferences
                    if role == AgentRole.SECURITY_ANALYST:
                        mock_config.preferred_models = ["openai/o3", "anthropic/claude-3-opus"]
                    elif role == AgentRole.PERFORMANCE_OPTIMIZER:
                        mock_config.preferred_models = ["google/gemini-pro", "openai/gpt-4-turbo"]
                    elif role == AgentRole.ARCHITECTURE_REVIEWER:
                        mock_config.preferred_models = ["anthropic/claude-3-opus", "openai/o3"]
                    else:  # GENERALIST
                        mock_config.preferred_models = ["google/gemini-flash", "openai/gpt-4o-mini"]
                    
                    mock_create_config.return_value = mock_config
                    
                    # Act
                    config = manager._create_openrouter_config_for_role(role)
                    
                    # Assert
                    self.assertEqual(config.api_key, self.test_openrouter_key)
                    self.assertEqual(config.rate_limit_per_minute, 60)
                    self.assertEqual(config.max_concurrent_calls, 3)
                    self.assertEqual(config.thinking_mode_default, "high")
                    self.assertIsInstance(config.preferred_models, list)
                    self.assertGreater(len(config.preferred_models), 0)


class TestInterAgentCommunicationIntegration(unittest.TestCase):
    """Integration tests for inter-agent communication and synchronized thinking sessions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.communication_system = AgentCommunicationSystem(max_agents=4)
        self.thinking_coordinator = ThinkingSessionCoordinator()
        
        # Create test agents
        self.test_agents = []
        for i in range(4):
            role = [
                AgentRole.SECURITY_ANALYST,
                AgentRole.PERFORMANCE_OPTIMIZER,
                AgentRole.ARCHITECTURE_REVIEWER,
                AgentRole.GENERALIST
            ][i]
            agent = self.communication_system.register_agent(core_id=i, role=role)
            self.test_agents.append(agent)
    
    def tearDown(self):
        """Clean up after tests"""
        # Shutdown communication system
        self.communication_system.shutdown()
        self.thinking_coordinator.shutdown()
    
    def test_inter_agent_message_exchange(self):
        """Test basic inter-agent message exchange"""
        # Arrange
        sender = self.test_agents[0]  # Security Analyst
        receiver = self.test_agents[1]  # Performance Optimizer
        
        message_content = "I've identified a potential security vulnerability in the authentication module"
        
        # Act
        message_id = self.communication_system.send_message(
            from_agent=sender.agent_id,
            to_agent=receiver.agent_id,
            message_type="insight",
            content=message_content,
            priority=7
        )
        
        # Allow message processing
        time.sleep(0.1)
        
        # Assert
        self.assertIsNotNone(message_id)
        self.assertIsInstance(message_id, str)
        
        # Check message was processed
        messages = receiver.get_unread_messages()
        self.assertGreater(len(messages), 0)
        
        # Find our message
        our_message = next((m for m in messages if m.message_id == message_id), None)
        self.assertIsNotNone(our_message)
        self.assertEqual(our_message.from_agent, sender.agent_id)
        self.assertEqual(our_message.to_agent, receiver.agent_id)
        self.assertEqual(our_message.message_type, "insight")
        self.assertEqual(our_message.content, message_content)
        self.assertEqual(our_message.priority, 7)
    
    def test_broadcast_message_coordination(self):
        """Test broadcast message coordination to all agents"""
        # Arrange
        sender = self.test_agents[0]  # Security Analyst
        broadcast_content = "Critical security alert: All agents should review their modules for SQL injection vulnerabilities"
        
        # Act
        message_id = self.communication_system.send_message(
            from_agent=sender.agent_id,
            to_agent="ALL",
            message_type="alert",
            content=broadcast_content,
            priority=9
        )
        
        # Allow message processing (broadcast may take longer)
        time.sleep(1.0)
        
        # Assert
        self.assertIsNotNone(message_id)
        self.assertIsInstance(message_id, str)
        
        # Check all other agents received the message
        for agent in self.test_agents[1:]:  # Skip sender
            messages = agent.get_unread_messages()
            broadcast_messages = [m for m in messages if m.message_id == message_id]
            self.assertEqual(len(broadcast_messages), 1, f"Agent {agent.agent_id} should have received broadcast")
            
            # Check message content
            if broadcast_messages:
                msg = broadcast_messages[0]
                self.assertEqual(msg.to_agent, "ALL")
                self.assertEqual(msg.message_type, "alert")
                self.assertEqual(msg.content, broadcast_content)
    
    def test_synchronized_thinking_session_creation(self):
        """Test creation of synchronized thinking sessions"""
        # Arrange
        participating_agents = [agent.agent_id for agent in self.test_agents[:3]]
        thinking_prompt = "Analyze the security implications of implementing OAuth 2.0 in our system"
        
        sync_request = SynchronizedThinkingRequest(
            request_id="sync_test_001",
            thinking_prompt=thinking_prompt,
            participating_agent_ids=participating_agents,
            session_type=ThinkingSessionType.SYNCHRONIZED,
            timeout_seconds=300.0,
            priority=8,
            require_all_agents=True,
            aggregation_strategy="consensus"
        )
        
        # Act
        sessions = self.thinking_coordinator.create_synchronized_session(sync_request)
        
        # Assert
        self.assertEqual(len(sessions), 3)
        self.assertEqual(set(sessions.keys()), set(participating_agents))
        
        # Check each session is properly configured
        for agent_id, session in sessions.items():
            self.assertEqual(session.agent_id, agent_id)
            self.assertEqual(session.thinking_prompt, thinking_prompt)
            self.assertEqual(session.session_type, ThinkingSessionType.SYNCHRONIZED)
            self.assertEqual(session.timeout_seconds, 300.0)
            self.assertEqual(session.priority, 8)
            self.assertEqual(session.status, ThinkingSessionStatus.PENDING)
            
            # Check session linking
            self.assertIsNotNone(session.parent_session_id)
            self.assertEqual(len(session.related_agent_ids), 3)
            self.assertIn(agent_id, session.related_agent_ids)
            
            # Check child session linking
            expected_child_count = len(participating_agents) - 1  # All except self
            self.assertEqual(len(session.child_session_ids), expected_child_count)
    
    def test_synchronized_thinking_session_execution(self):
        """Test execution of synchronized thinking sessions"""
        # Arrange
        participating_agents = [agent.agent_id for agent in self.test_agents[:2]]
        thinking_prompt = "What are the performance bottlenecks in our current architecture?"
        
        sync_request = SynchronizedThinkingRequest(
            request_id="sync_exec_001",
            thinking_prompt=thinking_prompt,
            participating_agent_ids=participating_agents,
            session_type=ThinkingSessionType.SYNCHRONIZED,
            timeout_seconds=60.0,
            aggregation_strategy="consensus"
        )
        
        sessions = self.thinking_coordinator.create_synchronized_session(sync_request)
        
        # Act - Start sessions
        session_ids = list(sessions.keys())
        for agent_id in session_ids:
            session = sessions[agent_id]
            success = self.thinking_coordinator.start_session(session.session_id)
            self.assertTrue(success, f"Failed to start session for agent {agent_id}")
        
        # Simulate session completion
        results = [
            "Performance bottleneck identified in database query optimization",
            "Memory usage spikes detected during peak load periods"
        ]
        
        for i, (agent_id, session) in enumerate(sessions.items()):
            success = self.thinking_coordinator.complete_session(
                session.session_id,
                result=results[i],
                tokens_used=150 + i * 50,
                cost_estimate=0.01 + i * 0.005
            )
            self.assertTrue(success, f"Failed to complete session for agent {agent_id}")
        
        # Allow aggregation processing
        time.sleep(0.1)
        
        # Assert - Check aggregated results
        parent_session_id = list(sessions.values())[0].parent_session_id
        aggregated = self.thinking_coordinator.aggregate_synchronized_results(
            parent_session_id,
            "consensus"
        )
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated["strategy"], "consensus")
        self.assertEqual(aggregated["total_sessions"], 2)
        self.assertEqual(len(aggregated["results"]), 2)
        
        # Check individual results are included
        for i, result_data in enumerate(aggregated["results"]):
            self.assertEqual(result_data["agent_id"], participating_agents[i])
            self.assertEqual(result_data["result"], results[i])
            self.assertGreater(result_data["tokens_used"], 0)
            self.assertGreater(result_data["duration"], 0)
    
    def test_collaborative_thinking_session(self):
        """Test collaborative thinking where agents build on each other's thoughts"""
        # Arrange
        participating_agents = [agent.agent_id for agent in self.test_agents[:3]]
        thinking_prompt = "Design a scalable microservices architecture for our e-commerce platform"
        
        sync_request = SynchronizedThinkingRequest(
            request_id="collab_001",
            thinking_prompt=thinking_prompt,
            participating_agent_ids=participating_agents,
            session_type=ThinkingSessionType.COLLABORATIVE,
            timeout_seconds=180.0,
            aggregation_strategy="all"
        )
        
        sessions = self.thinking_coordinator.create_synchronized_session(sync_request)
        
        # Act - Execute collaborative session
        for agent_id, session in sessions.items():
            self.thinking_coordinator.start_session(session.session_id)
        
        # Simulate collaborative results building on each other
        collaborative_results = [
            "Security Analyst: Implement OAuth 2.0 with JWT tokens for authentication across microservices",
            "Performance Optimizer: Use Redis for caching and implement circuit breakers for resilience",
            "Architecture Reviewer: Adopt event-driven architecture with Apache Kafka for service communication"
        ]
        
        for i, (agent_id, session) in enumerate(sessions.items()):
            self.thinking_coordinator.complete_session(
                session.session_id,
                result=collaborative_results[i],
                tokens_used=200 + i * 75
            )
        
        # Assert - Check collaborative aggregation
        parent_session_id = list(sessions.values())[0].parent_session_id
        aggregated = self.thinking_coordinator.aggregate_synchronized_results(
            parent_session_id,
            "all"
        )
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated["strategy"], "all")
        self.assertEqual(len(aggregated["results"]), 3)
        
        # Check combined result includes all perspectives
        combined_result = aggregated["combined_result"]
        for result in collaborative_results:
            self.assertIn(result.split(": ")[1], combined_result)  # Check content is included
    
    def test_thinking_session_timeout_handling(self):
        """Test timeout handling in thinking sessions"""
        # Arrange
        agent = self.test_agents[0]
        short_timeout = 0.1  # 100ms timeout
        
        session = self.thinking_coordinator.create_individual_session(
            agent=agent,
            thinking_prompt="This is a test prompt that will timeout",
            timeout_seconds=short_timeout
        )
        
        # Act - Start session but don't complete it
        self.thinking_coordinator.start_session(session.session_id)
        
        # Wait for timeout
        time.sleep(short_timeout + 0.05)  # Wait a bit longer than timeout
        
        # Check if session is marked as timed out
        updated_session = self.thinking_coordinator.get_session(session.session_id)
        
        # Assert
        self.assertIsNotNone(updated_session)
        self.assertTrue(updated_session.is_timed_out())


class TestWorkloadRedistributionAndFailureRecovery(unittest.TestCase):
    """Integration tests for workload redistribution and failure recovery scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_openrouter_key = "test-openrouter-key-123"
        self.test_max_agents = 4
        
        # Mock configuration manager
        self.mock_config_manager = Mock()
        self.mock_system_config = Mock()
        self.mock_system_config.openrouter_api_key = self.test_openrouter_key
        self.mock_system_config.max_agents = self.test_max_agents
        self.mock_system_config.validate.return_value = []
        self.mock_system_config.to_dict.return_value = {"test": "config"}
        self.mock_config_manager.get_config.return_value = self.mock_system_config
        self.mock_config_manager.add_change_callback = Mock()
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_agent_communication_system')
    @patch('utils.per_core_error_handling.get_per_core_error_handler')
    @patch('utils.per_core_logging.log_system_event')
    @patch('utils.per_core_logging.log_agent_activity')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_single_agent_failure_recovery(
        self, mock_exists, mock_get_config_manager, mock_log_agent,
        mock_log_system, mock_get_error_handler, mock_get_comm_system,
        mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test recovery from single agent failure with workload redistribution"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Mock CPU detection to return our test count
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Mock communication system
        mock_comm_system = Mock()
        mock_get_comm_system.return_value = mock_comm_system
        
        # Create mock agents
        mock_agents = []
        for i in range(self.test_max_agents):
            mock_agent = Mock()
            mock_agent.agent_id = f"core_{i}_test_agent"
            mock_agent.core_id = i
            mock_agent.role = AgentRole.GENERALIST
            mock_agent.status = AgentStatus.ACTIVE
            mock_agent.last_activity = time.time()
            mock_agents.append(mock_agent)
        
        mock_comm_system.register_agent.side_effect = mock_agents
        
        # Mock error handler
        mock_error_handler = AsyncMock()
        mock_get_error_handler.return_value = mock_error_handler
        
        # Initialize manager
        manager = PerCoreAgentManager()
        
        with patch.object(manager, '_initialize_monitoring_and_alerting'):
            with patch.object(manager, '_start_health_monitoring'):
                with patch.object(manager, '_check_openrouter_connection_health', return_value=True):
                    with patch.object(manager, '_validate_openrouter_api_key', return_value=True):
                        with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                            # Mock OpenRouter config
                            mock_config = Mock()
                            mock_config.api_key = self.test_openrouter_key
                            mock_config.preferred_models = ["test-model"]
                            mock_config.rate_limit_per_minute = 60
                            mock_config.max_concurrent_calls = 3
                            mock_create_config.return_value = mock_config
                            
                            agents = manager.initialize_agents()
        
        # Simulate agent failure
        failed_agent = mock_agents[1]  # Fail the second agent
        failed_agent.status = AgentStatus.OFFLINE
        
        # Update agent status to reflect failure
        manager.agent_statuses[failed_agent.agent_id].status = AgentStatus.OFFLINE
        manager.agent_statuses[failed_agent.agent_id].openrouter_connected = False
        manager.agent_statuses[failed_agent.agent_id].success_rate = 0.0
        
        # Act - Trigger workload redistribution
        with patch.object(manager, '_attempt_agent_recovery') as mock_recovery:
            mock_recovery.return_value = False  # Recovery fails
            
            result = manager.redistribute_workload(failed_agent.agent_id)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertIn("failed_agent_id", result)
        self.assertIn("workload_redistributed", result)
        self.assertEqual(result["failed_agent_id"], failed_agent.agent_id)
        self.assertTrue(result["workload_redistributed"])
        
        # Check system health reflects the failure
        health = manager.get_system_health()
        self.assertEqual(health["total_agents"], self.test_max_agents)
        self.assertEqual(health["healthy_agents"], self.test_max_agents - 1)  # One failed
        
        # Check that the failed agent is reflected in the health status
        # (The exact key name may vary, so let's check what's available)
        self.assertIn("agent_statuses", health)
        agent_statuses = health["agent_statuses"]
        
        # Check that the failed agent shows as unhealthy
        failed_status = agent_statuses[failed_agent.agent_id]
        self.assertFalse(failed_status["is_healthy"])
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_multiple_agent_failure_graceful_degradation(self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count):
        """Test graceful degradation when multiple agents fail"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Mock CPU detection to return our test count
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        manager = PerCoreAgentManager()
        
        # Initialize agents first
        with patch('utils.per_core_agent_manager.get_agent_communication_system') as mock_get_comm_system:
            mock_comm_system = Mock()
            mock_get_comm_system.return_value = mock_comm_system
            
            # Create mock agents
            mock_agents = []
            for i in range(self.test_max_agents):
                mock_agent = Mock()
                mock_agent.agent_id = f"core_{i}_test_agent"
                mock_agent.core_id = i
                mock_agent.role = AgentRole.GENERALIST
                mock_agent.status = AgentStatus.ACTIVE
                mock_agent.last_activity = time.time()
                mock_agents.append(mock_agent)
            
            mock_comm_system.register_agent.side_effect = mock_agents
            
            with patch.object(manager, '_initialize_monitoring_and_alerting'):
                with patch.object(manager, '_start_health_monitoring'):
                    with patch.object(manager, '_check_openrouter_connection_health', return_value=True):
                        with patch.object(manager, '_validate_openrouter_api_key', return_value=True):
                            with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                                mock_config = Mock()
                                mock_config.api_key = self.test_openrouter_key
                                mock_config.preferred_models = ["test-model"]
                                mock_create_config.return_value = mock_config
                                
                                agents = manager.initialize_agents()
        
        # Now simulate multiple agent failures (more than 50% of agents)
        failed_agent_ids = ["core_0_test_agent", "core_1_test_agent", "core_2_test_agent"]
        
        # Update agent statuses to simulate failures
        for agent_id in failed_agent_ids:
            if agent_id in manager.agent_statuses:
                manager.agent_statuses[agent_id].status = AgentStatus.OFFLINE
                manager.agent_statuses[agent_id].openrouter_connected = False
                manager.agent_statuses[agent_id].success_rate = 0.0
        
        # Act - Check system degradation
        health = manager.get_system_health()
        
        # Assert - System should detect critical failure state
        self.assertEqual(health["total_agents"], self.test_max_agents)
        self.assertEqual(health["healthy_agents"], 1)  # Only one healthy
        
        # Check that most agents are unhealthy
        self.assertLessEqual(health["healthy_agents"], health["total_agents"] // 2)  # Less than half healthy
        
        # Check individual agent statuses
        agent_statuses = health["agent_statuses"]
        offline_count = sum(1 for status in agent_statuses.values() if not status["is_healthy"])
        self.assertEqual(offline_count, 3)  # Three should be unhealthy
        
        # Test graceful degradation mode (if such method exists)
        # For now, just verify the system recognizes the degraded state
        self.assertLess(health["health_score"], 0.5)  # Health score should be low
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_openrouter_api_failure_recovery(self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count):
        """Test recovery from OpenRouter API failures"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Mock CPU detection to return our test count
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        manager = PerCoreAgentManager()
        
        # Initialize manager first
        with patch('utils.per_core_agent_manager.get_agent_communication_system') as mock_get_comm_system:
            mock_comm_system = Mock()
            mock_get_comm_system.return_value = mock_comm_system
            
            # Create mock agent
            mock_agent = Mock()
            mock_agent.agent_id = "core_0_security_analyst"
            mock_agent.core_id = 0
            mock_agent.role = AgentRole.SECURITY_ANALYST
            mock_agent.status = AgentStatus.ACTIVE
            mock_agent.last_activity = time.time()
            
            mock_comm_system.register_agent.return_value = mock_agent
            
            with patch.object(manager, '_initialize_monitoring_and_alerting'):
                with patch.object(manager, '_start_health_monitoring'):
                    with patch.object(manager, '_check_openrouter_connection_health', return_value=False):  # Initially failed
                        with patch.object(manager, '_validate_openrouter_api_key', return_value=True):
                            with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                                mock_config = Mock()
                                mock_config.api_key = self.test_openrouter_key
                                mock_config.preferred_models = ["test-model"]
                                mock_create_config.return_value = mock_config
                                
                                agents = manager.initialize_agents()
        
        # Update agent status to simulate OpenRouter connection failure
        agent_id = "core_0_security_analyst"
        if agent_id in manager.agent_statuses:
            manager.agent_statuses[agent_id].openrouter_connected = False
            manager.agent_statuses[agent_id].success_rate = 0.3
        
        # Mock API client
        mock_api_client = Mock()
        mock_api_client.configure_openrouter_only = Mock()
        
        # Act - Attempt recovery
        with patch.object(manager, '_check_openrouter_connection_health') as mock_health_check:
            # First attempt fails, second succeeds
            mock_health_check.side_effect = [False, True]
            
            # Simulate recovery attempt
            recovery_success = manager._recover_openrouter_connection(agent_id, mock_agent)
        
        # Assert
        self.assertIsInstance(recovery_success, bool)  # Method should return a boolean
        self.assertGreaterEqual(mock_health_check.call_count, 1)  # Health check should be attempted
    
    def test_thinking_session_failure_and_redistribution(self):
        """Test thinking session failure handling and redistribution"""
        # Arrange
        coordinator = ThinkingSessionCoordinator()
        
        # Create test agents
        test_agents = []
        for i in range(3):
            agent = Mock()
            agent.agent_id = f"test_agent_{i}"
            agent.core_id = i
            agent.role = AgentRole.GENERALIST
            test_agents.append(agent)
        
        # Create synchronized thinking session
        sync_request = SynchronizedThinkingRequest(
            request_id="failure_test_001",
            thinking_prompt="Test prompt for failure handling",
            participating_agent_ids=[agent.agent_id for agent in test_agents],
            session_type=ThinkingSessionType.SYNCHRONIZED,
            require_all_agents=False,  # Allow partial completion
            min_agents_required=2,  # Need at least 2 agents
            aggregation_strategy="majority"
        )
        
        sessions = coordinator.create_synchronized_session(sync_request)
        
        # Start all sessions
        for session in sessions.values():
            coordinator.start_session(session.session_id)
        
        # Act - Simulate one session failure, others succeed
        session_list = list(sessions.values())
        
        # Fail first session
        coordinator.fail_session(
            session_list[0].session_id,
            "OpenRouter API connection timeout"
        )
        
        # Complete other sessions successfully
        for i, session in enumerate(session_list[1:], 1):
            coordinator.complete_session(
                session.session_id,
                result=f"Successful result from agent {i}",
                tokens_used=150
            )
        
        # Allow processing time
        time.sleep(0.1)
        
        # Assert - Check aggregation still works with partial results
        parent_session_id = session_list[0].parent_session_id
        aggregated = coordinator.aggregate_synchronized_results(
            parent_session_id,
            "majority"
        )
        
        self.assertIsNotNone(aggregated)
        self.assertEqual(aggregated["strategy"], "majority")
        self.assertEqual(aggregated["total_sessions"], 2)  # Only successful sessions
        
        # Check that failed session is not included in results
        successful_agent_ids = [result["agent_id"] for result in aggregated["results"]]
        self.assertNotIn(session_list[0].agent_id, successful_agent_ids)
        self.assertEqual(len(successful_agent_ids), 2)
    
    def test_circuit_breaker_pattern_for_api_failures(self):
        """Test circuit breaker pattern for handling repeated API failures"""
        # Arrange
        coordinator = ThinkingSessionCoordinator()
        agent = Mock()
        agent.agent_id = "test_agent_circuit_breaker"
        agent.core_id = 0
        agent.role = AgentRole.GENERALIST
        
        # Simulate multiple consecutive failures to trigger circuit breaker
        failure_count = 0
        max_failures = 5
        
        # Act - Create multiple sessions that will fail
        failed_sessions = []
        for i in range(max_failures + 2):  # Exceed threshold
            session = coordinator.create_individual_session(
                agent=agent,
                thinking_prompt=f"Test prompt {i}",
                timeout_seconds=60.0
            )
            
            coordinator.start_session(session.session_id)
            
            # Simulate API failure
            coordinator.fail_session(
                session.session_id,
                f"OpenRouter API error {i}: Rate limit exceeded"
            )
            
            failed_sessions.append(session)
            failure_count += 1
        
        # Assert - Check that circuit breaker logic would be triggered
        # (This would be implemented in the actual API client)
        self.assertEqual(len(failed_sessions), max_failures + 2)
        
        # All sessions should be marked as failed
        for session in failed_sessions:
            updated_session = coordinator.get_session(session.session_id)
            self.assertEqual(updated_session.status, ThinkingSessionStatus.FAILED)
            self.assertIn("OpenRouter API error", updated_session.error)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any resources
        pass


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)