"""
Unit tests for PerCoreAgentManager core agent management functionality

This module tests the core functionality of the PerCoreAgentManager including:
- Agent creation and lifecycle management
- OpenRouter configuration and API client initialization
- Agent role assignment and CPU core detection
- Health monitoring and status tracking
- Error handling and recovery

Requirements tested: 1.1, 1.2, 2.1
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


class TestPerCoreAgentManager(unittest.TestCase):
    """Test PerCoreAgentManager agent creation and lifecycle management"""
    
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
        
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any global state
        pass
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_init_with_config_manager(self, mock_exists, mock_get_config_manager):
        """Test PerCoreAgentManager initialization with config manager"""
        # Arrange
        mock_exists.return_value = False  # No recovery state
        mock_get_config_manager.return_value = self.mock_config_manager
        
        # Act
        manager = PerCoreAgentManager(config_manager=self.mock_config_manager)
        
        # Assert
        self.assertEqual(manager.config_manager, self.mock_config_manager)
        self.assertEqual(manager.openrouter_api_key, self.test_openrouter_key)
        self.assertEqual(manager.max_agents, self.test_max_agents)
        self.assertIsInstance(manager.agents, dict)
        self.assertIsInstance(manager.agent_statuses, dict)
        self.assertFalse(manager._shutdown)
        self.mock_config_manager.add_change_callback.assert_called_once()
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_init_with_legacy_parameters(self, mock_exists, mock_get_config_manager):
        """Test PerCoreAgentManager initialization with legacy parameters"""
        # Arrange
        mock_exists.return_value = False  # No recovery state
        mock_get_config_manager.return_value = self.mock_config_manager
        legacy_key = "legacy-key-456"
        legacy_max = 2
        
        # Act
        manager = PerCoreAgentManager(
            openrouter_api_key=legacy_key,
            max_agents=legacy_max
        )
        
        # Assert
        # Should use config manager but legacy values should override
        self.assertEqual(manager.openrouter_api_key, legacy_key)  # Legacy value should override
        self.assertEqual(manager.max_agents, legacy_max)  # Legacy value should override
        # System config should be updated with legacy values
        self.assertEqual(manager.system_config.openrouter_api_key, legacy_key)
        self.assertEqual(manager.system_config.max_agents, legacy_max)
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    def test_detect_cpu_cores(self, mock_os_cpu_count, mock_mp_cpu_count):
        """Test CPU core detection using multiple methods"""
        # Test case 1: Both methods return valid values
        mock_os_cpu_count.return_value = 8
        mock_mp_cpu_count.return_value = 8
        
        with patch('utils.per_core_agent_config.get_per_core_agent_config_manager') as mock_get_config:
            mock_get_config.return_value = self.mock_config_manager
            with patch('utils.per_core_agent_manager.os.path.exists', return_value=False):
                manager = PerCoreAgentManager()
                self.assertEqual(manager.detected_cores, 8)
        
        # Test case 2: Different values, should use minimum
        mock_os_cpu_count.return_value = 12
        mock_mp_cpu_count.return_value = 8
        
        with patch('utils.per_core_agent_config.get_per_core_agent_config_manager') as mock_get_config:
            mock_get_config.return_value = self.mock_config_manager
            with patch('utils.per_core_agent_manager.os.path.exists', return_value=False):
                manager = PerCoreAgentManager()
                self.assertEqual(manager.detected_cores, 8)
        
        # Test case 3: One method returns None
        mock_os_cpu_count.return_value = None
        mock_mp_cpu_count.return_value = 6
        
        with patch('utils.per_core_agent_config.get_per_core_agent_config_manager') as mock_get_config:
            mock_get_config.return_value = self.mock_config_manager
            with patch('utils.per_core_agent_manager.os.path.exists', return_value=False):
                manager = PerCoreAgentManager()
                self.assertEqual(manager.detected_cores, 4)  # Fallback to 4
        
        # Test case 4: Exception handling
        mock_os_cpu_count.side_effect = Exception("CPU detection error")
        mock_mp_cpu_count.side_effect = Exception("MP detection error")
        
        with patch('utils.per_core_agent_config.get_per_core_agent_config_manager') as mock_get_config:
            mock_get_config.return_value = self.mock_config_manager
            with patch('utils.per_core_agent_manager.os.path.exists', return_value=False):
                manager = PerCoreAgentManager()
                self.assertEqual(manager.detected_cores, 4)  # Default fallback
    
    def test_assign_agent_roles(self):
        """Test agent role assignment strategy based on available cores"""
        with patch('utils.per_core_agent_config.get_per_core_agent_config_manager') as mock_get_config:
            mock_get_config.return_value = self.mock_config_manager
            with patch('utils.per_core_agent_manager.os.path.exists', return_value=False):
                manager = PerCoreAgentManager()
        
        # Test case 1: Fewer agents than specialized roles
        roles = manager._assign_agent_roles(4)
        expected_roles = [
            AgentRole.SECURITY_ANALYST,
            AgentRole.PERFORMANCE_OPTIMIZER,
            AgentRole.ARCHITECTURE_REVIEWER,
            AgentRole.CODE_QUALITY_INSPECTOR
        ]
        self.assertEqual(roles, expected_roles)
        
        # Test case 2: More agents than specialized roles
        roles = manager._assign_agent_roles(10)
        self.assertEqual(len(roles), 10)
        # First 8 should be specialized roles
        self.assertEqual(roles[0], AgentRole.SECURITY_ANALYST)
        self.assertEqual(roles[7], AgentRole.GENERALIST)
        # Remaining should be generalists
        self.assertEqual(roles[8], AgentRole.GENERALIST)
        self.assertEqual(roles[9], AgentRole.GENERALIST)
        
        # Test case 3: Single agent
        roles = manager._assign_agent_roles(1)
        self.assertEqual(roles, [AgentRole.SECURITY_ANALYST])
        
        # Test case 4: Zero agents (edge case)
        roles = manager._assign_agent_roles(0)
        self.assertEqual(roles, [])


class TestPerCoreAgentManagerOpenRouter(unittest.TestCase):
    """Test OpenRouter configuration and API client initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_openrouter_key = "test-openrouter-key-123"
        self.test_max_agents = 2
        
        # Mock configuration manager
        self.mock_config_manager = Mock()
        self.mock_system_config = Mock()
        self.mock_system_config.openrouter_api_key = self.test_openrouter_key
        self.mock_system_config.max_agents = self.test_max_agents
        self.mock_system_config.validate.return_value = []
        self.mock_system_config.to_dict.return_value = {"test": "config"}
        self.mock_config_manager.get_config.return_value = self.mock_system_config
        self.mock_config_manager.add_change_callback = Mock()
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_validate_openrouter_api_key(self, mock_exists, mock_get_config_manager):
        """Test OpenRouter API key validation"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Test valid key format
        valid_keys = [
            "sk-or-v1-1234567890abcdef",
            "test-key-123",
            "openrouter-api-key-456"
        ]
        
        for key in valid_keys:
            with patch.object(manager, '_validate_openrouter_api_key') as mock_validate:
                mock_validate.return_value = True
                result = manager._validate_openrouter_api_key(key)
                self.assertTrue(result)
        
        # Test invalid keys
        invalid_keys = [
            "",
            None,
            "short",
            "your_api_key_here",
            "placeholder"
        ]
        
        for key in invalid_keys:
            with patch.object(manager, '_validate_openrouter_api_key') as mock_validate:
                mock_validate.return_value = False
                result = manager._validate_openrouter_api_key(key)
                self.assertFalse(result)
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_create_openrouter_config_for_role(self, mock_exists, mock_get_config_manager):
        """Test OpenRouter configuration creation for different agent roles"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Test different roles get appropriate model preferences
        test_cases = [
            (AgentRole.SECURITY_ANALYST, ["openai/o3", "anthropic/claude-3-opus"]),
            (AgentRole.PERFORMANCE_OPTIMIZER, ["google/gemini-pro", "openai/gpt-4-turbo"]),
            (AgentRole.ARCHITECTURE_REVIEWER, ["anthropic/claude-3-opus", "openai/o3"]),
            (AgentRole.GENERALIST, ["google/gemini-flash", "openai/gpt-4o-mini"])
        ]
        
        for role, expected_models in test_cases:
            with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                mock_config = Mock()
                mock_config.preferred_models = expected_models
                mock_config.rate_limit_per_minute = 60
                mock_config.max_concurrent_calls = 3
                mock_create_config.return_value = mock_config
                
                config = manager._create_openrouter_config_for_role(role)
                self.assertEqual(config.preferred_models, expected_models)
                self.assertEqual(config.rate_limit_per_minute, 60)
                self.assertEqual(config.max_concurrent_calls, 3)
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_configure_agent_openrouter_success(self, mock_exists, mock_get_config_manager):
        """Test successful OpenRouter configuration for an agent"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.role = AgentRole.SECURITY_ANALYST
        
        # Mock communication system and API client
        mock_comm_system = Mock()
        mock_api_client = Mock()
        mock_comm_system.get_agent_api_client.return_value = mock_api_client
        manager.communication_system = mock_comm_system
        
        # Initialize agent status
        manager._initialize_agent_status(mock_agent)
        
        # Mock validation and health check
        with patch.object(manager, '_validate_openrouter_api_key', return_value=True):
            with patch.object(manager, '_create_openrouter_config_for_role') as mock_create_config:
                with patch.object(manager, '_check_openrouter_connection_health', return_value=True):
                    mock_config = Mock()
                    mock_create_config.return_value = mock_config
                    
                    # Act
                    manager._configure_agent_openrouter(mock_agent)
        
        # Assert
        mock_api_client.configure_openrouter_only.assert_called_once_with(
            self.test_openrouter_key, mock_config
        )
        
        # Check status was updated
        status = manager.agent_statuses[mock_agent.agent_id]
        self.assertTrue(status.openrouter_connected)
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_configure_agent_openrouter_invalid_key(self, mock_exists, mock_get_config_manager):
        """Test OpenRouter configuration with invalid API key"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.role = AgentRole.SECURITY_ANALYST
        
        # Mock communication system
        mock_comm_system = Mock()
        mock_api_client = Mock()
        mock_comm_system.get_agent_api_client.return_value = mock_api_client
        manager.communication_system = mock_comm_system
        
        # Initialize agent status
        manager._initialize_agent_status(mock_agent)
        
        # Mock invalid key validation
        with patch.object(manager, '_validate_openrouter_api_key', return_value=False):
            # Act
            manager._configure_agent_openrouter(mock_agent)
        
        # Assert
        mock_api_client.configure_openrouter_only.assert_not_called()
        
        # Check status remains disconnected
        status = manager.agent_statuses[mock_agent.agent_id]
        self.assertFalse(status.openrouter_connected)
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_check_openrouter_connection_health(self, mock_exists, mock_get_config_manager):
        """Test OpenRouter connection health check"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        mock_api_client = Mock()
        
        # Test successful health check
        with patch.object(manager, '_check_openrouter_connection_health') as mock_health_check:
            mock_health_check.return_value = True
            result = manager._check_openrouter_connection_health("test_agent", mock_api_client)
            self.assertTrue(result)
        
        # Test failed health check
        with patch.object(manager, '_check_openrouter_connection_health') as mock_health_check:
            mock_health_check.return_value = False
            result = manager._check_openrouter_connection_health("test_agent", mock_api_client)
            self.assertFalse(result)


class TestPerCoreAgentManagerLifecycle(unittest.TestCase):
    """Test agent lifecycle management including initialization and cleanup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_openrouter_key = "test-openrouter-key-123"
        self.test_max_agents = 2
        
        # Mock configuration manager
        self.mock_config_manager = Mock()
        self.mock_system_config = Mock()
        self.mock_system_config.openrouter_api_key = self.test_openrouter_key
        self.mock_system_config.max_agents = self.test_max_agents
        self.mock_system_config.validate.return_value = []
        self.mock_system_config.to_dict.return_value = {"test": "config"}
        self.mock_config_manager.get_config.return_value = self.mock_system_config
        self.mock_config_manager.add_change_callback = Mock()
    
    @patch('utils.per_core_agent_manager.get_agent_communication_system')
    @patch('utils.per_core_error_handling.get_per_core_error_handler')
    @patch('utils.per_core_logging.log_system_event')
    @patch('utils.per_core_logging.log_agent_activity')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_initialize_agents_success(self, mock_exists, mock_get_config_manager,
                                     mock_log_agent, mock_log_system, 
                                     mock_get_error_handler, mock_get_comm_system):
        """Test successful agent initialization"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Mock communication system
        mock_comm_system = Mock()
        mock_get_comm_system.return_value = mock_comm_system
        
        # Mock agents
        mock_agents = []
        for i in range(manager.effective_max_agents):
            mock_agent = Mock()
            mock_agent.agent_id = f"core_{i}_security_analyst"
            mock_agent.core_id = i
            mock_agent.role = AgentRole.SECURITY_ANALYST
            mock_agent.status = AgentStatus.ACTIVE
            mock_agent.last_activity = time.time()
            mock_agents.append(mock_agent)
        
        mock_comm_system.register_agent.side_effect = mock_agents
        
        # Mock error handler
        mock_error_handler = AsyncMock()
        mock_get_error_handler.return_value = mock_error_handler
        
        # Mock monitoring initialization
        with patch.object(manager, '_initialize_monitoring_and_alerting'):
            with patch.object(manager, '_start_health_monitoring'):
                with patch.object(manager, '_configure_agent_openrouter'):
                    # Act
                    result = manager.initialize_agents()
        
        # Assert
        self.assertEqual(len(result), manager.effective_max_agents)
        self.assertEqual(len(manager.agents), manager.effective_max_agents)
        self.assertEqual(len(manager.agent_statuses), manager.effective_max_agents)
        
        # Verify communication system calls
        self.assertEqual(mock_comm_system.register_agent.call_count, manager.effective_max_agents)
        
        # Verify logging calls
        mock_log_system.assert_called()
        mock_log_agent.assert_called()
    
    @patch('utils.per_core_agent_manager.get_agent_communication_system')
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_initialize_agents_already_initialized(self, mock_exists, mock_get_config_manager, mock_get_comm_system):
        """Test that initialize_agents skips if agents already exist"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        # Pre-populate agents
        mock_agent = Mock()
        manager.agents[0] = mock_agent
        
        # Act
        result = manager.initialize_agents()
        
        # Assert
        self.assertEqual(result, [mock_agent])
        mock_get_comm_system.assert_not_called()
    
    @patch('utils.per_core_agent_config.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    def test_initialize_agent_status(self, mock_exists, mock_get_config_manager):
        """Test agent status initialization"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        manager = PerCoreAgentManager()
        
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.core_id = 0
        mock_agent.role = AgentRole.SECURITY_ANALYST
        mock_agent.status = AgentStatus.ACTIVE
        mock_agent.last_activity = time.time()
        
        # Act
        manager._initialize_agent_status(mock_agent)
        
        # Assert
        self.assertIn(mock_agent.agent_id, manager.agent_statuses)
        status = manager.agent_statuses[mock_agent.agent_id]
        self.assertIsInstance(status, PerCoreAgentStatus)
        self.assertEqual(status.agent_id, mock_agent.agent_id)
        self.assertEqual(status.core_id, mock_agent.core_id)
        self.assertEqual(status.role, mock_agent.role)
        self.assertEqual(status.status, mock_agent.status)
        self.assertFalse(status.openrouter_connected)  # Initially false
        self.assertEqual(status.active_thinking_sessions, 0)
        self.assertEqual(status.total_api_calls, 0)
        self.assertEqual(status.success_rate, 1.0)  # Start optimistic


class TestPerCoreAgentStatus(unittest.TestCase):
    """Test PerCoreAgentStatus data model and health checks"""
    
    def test_agent_status_creation(self):
        """Test PerCoreAgentStatus creation and basic properties"""
        # Arrange
        current_time = time.time()
        status = PerCoreAgentStatus(
            agent_id="test_agent_123",
            core_id=0,
            role=AgentRole.SECURITY_ANALYST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=2,
            total_api_calls=50,
            success_rate=0.85,
            last_activity=current_time,
            memory_usage_mb=128.5
        )
        
        # Assert
        self.assertEqual(status.agent_id, "test_agent_123")
        self.assertEqual(status.core_id, 0)
        self.assertEqual(status.role, AgentRole.SECURITY_ANALYST)
        self.assertEqual(status.status, AgentStatus.ACTIVE)
        self.assertTrue(status.openrouter_connected)
        self.assertEqual(status.active_thinking_sessions, 2)
        self.assertEqual(status.total_api_calls, 50)
        self.assertEqual(status.success_rate, 0.85)
        self.assertEqual(status.last_activity, current_time)
        self.assertEqual(status.memory_usage_mb, 128.5)
    
    def test_is_healthy(self):
        """Test agent health determination"""
        current_time = time.time()
        
        # Test healthy agent
        healthy_status = PerCoreAgentStatus(
            agent_id="healthy_agent",
            core_id=0,
            role=AgentRole.SECURITY_ANALYST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=1,
            total_api_calls=100,
            success_rate=0.8,
            last_activity=current_time - 60,  # 1 minute ago
            memory_usage_mb=100.0
        )
        self.assertTrue(healthy_status.is_healthy())
        
        # Test unhealthy agent - low success rate
        unhealthy_status = PerCoreAgentStatus(
            agent_id="unhealthy_agent",
            core_id=1,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.3,  # Below 50% threshold
            last_activity=current_time - 60,
            memory_usage_mb=100.0
        )
        self.assertFalse(unhealthy_status.is_healthy())
        
        # Test unhealthy agent - not connected
        disconnected_status = PerCoreAgentStatus(
            agent_id="disconnected_agent",
            core_id=2,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=False,  # Not connected
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.9,
            last_activity=current_time - 60,
            memory_usage_mb=100.0
        )
        self.assertFalse(disconnected_status.is_healthy())
        
        # Test unhealthy agent - inactive too long
        inactive_status = PerCoreAgentStatus(
            agent_id="inactive_agent",
            core_id=3,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.9,
            last_activity=current_time - 400,  # Over 5 minutes ago
            memory_usage_mb=100.0
        )
        self.assertFalse(inactive_status.is_healthy())
    
    def test_needs_attention(self):
        """Test agent attention requirement determination"""
        current_time = time.time()
        
        # Test agent that doesn't need attention
        good_status = PerCoreAgentStatus(
            agent_id="good_agent",
            core_id=0,
            role=AgentRole.SECURITY_ANALYST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=1,
            total_api_calls=100,
            success_rate=0.8,
            last_activity=current_time - 60,
            memory_usage_mb=100.0
        )
        self.assertFalse(good_status.needs_attention())
        
        # Test agent that needs attention - not connected
        needs_attention_status = PerCoreAgentStatus(
            agent_id="attention_agent",
            core_id=1,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=False,  # Needs attention
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.8,
            last_activity=current_time - 60,
            memory_usage_mb=100.0
        )
        self.assertTrue(needs_attention_status.needs_attention())
        
        # Test agent that needs attention - very low success rate
        low_success_status = PerCoreAgentStatus(
            agent_id="low_success_agent",
            core_id=2,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.2,  # Below 30% threshold
            last_activity=current_time - 60,
            memory_usage_mb=100.0
        )
        self.assertTrue(low_success_status.needs_attention())
        
        # Test agent that needs attention - inactive too long
        very_inactive_status = PerCoreAgentStatus(
            agent_id="very_inactive_agent",
            core_id=3,
            role=AgentRole.GENERALIST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=0,
            total_api_calls=100,
            success_rate=0.9,
            last_activity=current_time - 700,  # Over 10 minutes ago
            memory_usage_mb=100.0
        )
        self.assertTrue(very_inactive_status.needs_attention())
    
    def test_to_dict(self):
        """Test PerCoreAgentStatus serialization to dictionary"""
        current_time = time.time()
        status = PerCoreAgentStatus(
            agent_id="test_agent_123",
            core_id=0,
            role=AgentRole.SECURITY_ANALYST,
            status=AgentStatus.ACTIVE,
            openrouter_connected=True,
            active_thinking_sessions=2,
            total_api_calls=50,
            success_rate=0.85,
            last_activity=current_time,
            memory_usage_mb=128.5
        )
        
        result = status.to_dict()
        
        expected_keys = {
            "agent_id", "core_id", "role", "status", "openrouter_connected",
            "active_thinking_sessions", "total_api_calls", "success_rate",
            "last_activity", "memory_usage_mb", "is_healthy", "needs_attention"
        }
        
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["agent_id"], "test_agent_123")
        self.assertEqual(result["core_id"], 0)
        self.assertEqual(result["role"], "security_analyst")
        self.assertEqual(result["status"], "active")
        self.assertTrue(result["openrouter_connected"])
        self.assertEqual(result["active_thinking_sessions"], 2)
        self.assertEqual(result["total_api_calls"], 50)
        self.assertEqual(result["success_rate"], 0.85)
        self.assertEqual(result["last_activity"], current_time)
        self.assertEqual(result["memory_usage_mb"], 128.5)
        self.assertIsInstance(result["is_healthy"], bool)
        self.assertIsInstance(result["needs_attention"], bool)


if __name__ == '__main__':
    unittest.main()