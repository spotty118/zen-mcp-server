"""
Unit tests for AgentAPIClient OpenRouter-only enhancements

Tests the OpenRouter-only functionality added to AgentAPIClient as part of
the per-core agent coordination feature.
"""

import pytest
import time
from unittest.mock import Mock, patch

from utils.agent_api_client import AgentAPIClient, OpenRouterConfig
from utils.agent_core import Agent, AgentRole
from providers.base import ProviderType


class TestAgentAPIClientOpenRouter:
    """Test OpenRouter-only enhancements to AgentAPIClient"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.agent = Agent(agent_id='test_agent', core_id=0, role=AgentRole.SECURITY_ANALYST)
        self.client = AgentAPIClient(self.agent)
    
    def test_configure_openrouter_only_default_config(self):
        """Test configuring OpenRouter-only with default configuration"""
        # Configure with default settings
        self.client.configure_openrouter_only('test_api_key')
        
        # Verify configuration
        assert self.client.openrouter_only is True
        assert self.client.openrouter_config is not None
        assert self.client.openrouter_config.api_key == 'test_api_key'
        assert self.client.preferred_providers == [ProviderType.OPENROUTER]
        
        # Verify role-specific model preferences for security analyst
        expected_models = ["openai/o3", "anthropic/claude-3-opus", "google/gemini-2.5-pro"]
        assert self.client.openrouter_config.preferred_models == expected_models
        
        # Verify default settings
        assert self.client.openrouter_config.rate_limit_per_minute == 60
        assert self.client.openrouter_config.max_concurrent_calls == 3
        assert self.client.openrouter_config.thinking_mode_default == "high"
        assert self.client.openrouter_config.temperature_range == (0.3, 0.8)
    
    def test_configure_openrouter_only_custom_config(self):
        """Test configuring OpenRouter-only with custom configuration"""
        custom_config = OpenRouterConfig(
            api_key='custom_key',
            preferred_models=['custom/model-1', 'custom/model-2'],
            rate_limit_per_minute=30,
            max_concurrent_calls=2,
            thinking_mode_default='standard',
            temperature_range=(0.1, 0.9),
            circuit_breaker_threshold=3
        )
        
        self.client.configure_openrouter_only('custom_key', custom_config)
        
        # Verify custom configuration
        assert self.client.openrouter_config.preferred_models == ['custom/model-1', 'custom/model-2']
        assert self.client.openrouter_config.rate_limit_per_minute == 30
        assert self.client.openrouter_config.max_concurrent_calls == 2
        assert self.client.openrouter_config.thinking_mode_default == 'standard'
        assert self.client.openrouter_config.temperature_range == (0.1, 0.9)
        assert self.client.openrouter_config.circuit_breaker_threshold == 3
    
    def test_role_specific_model_preferences(self):
        """Test that different agent roles get appropriate model preferences"""
        role_expectations = {
            AgentRole.SECURITY_ANALYST: ["openai/o3", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
            AgentRole.PERFORMANCE_OPTIMIZER: ["google/gemini-2.5-pro", "openai/o3", "anthropic/claude-3-sonnet"],
            AgentRole.ARCHITECTURE_REVIEWER: ["openai/o3-pro", "anthropic/claude-3-opus", "openai/o3"],
            AgentRole.GENERALIST: ["google/gemini-2.5-flash", "openai/o3-mini", "anthropic/claude-3-haiku"]
        }
        
        for role, expected_models in role_expectations.items():
            agent = Agent(agent_id=f'test_{role.value}', core_id=0, role=role)
            client = AgentAPIClient(agent)
            client.configure_openrouter_only('test_key')
            
            assert client.openrouter_config.preferred_models == expected_models
    
    def test_get_openrouter_usage_stats(self):
        """Test OpenRouter usage statistics functionality"""
        self.client.configure_openrouter_only('test_key')
        
        stats = self.client.get_openrouter_usage_stats()
        
        # Verify stats structure
        expected_keys = [
            'agent_id', 'openrouter_only', 'total_openrouter_calls', 'successful_openrouter_calls',
            'openrouter_success_rate', 'avg_openrouter_execution_time', 'total_thinking_sessions',
            'successful_thinking_sessions', 'thinking_success_rate', 'avg_thinking_execution_time',
            'openrouter_configured', 'circuit_breaker_open', 'failure_count', 'rate_limit_calls_last_minute'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        # Verify initial values
        assert stats['agent_id'] == 'test_agent'
        assert stats['openrouter_only'] is True
        assert stats['total_openrouter_calls'] == 0
        assert stats['openrouter_configured'] is True
        assert stats['circuit_breaker_open'] is False
        assert stats['failure_count'] == 0
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality for OpenRouter failures"""
        config = OpenRouterConfig(
            api_key='test_key',
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=1.0
        )
        self.client.configure_openrouter_only('test_key', config)
        
        # Verify initial state
        assert not self.client._is_openrouter_circuit_open()
        assert self.client.openrouter_failure_count == 0
        
        # Simulate failures
        self.client._handle_openrouter_failure(Exception("Test failure 1"))
        assert self.client.openrouter_failure_count == 1
        assert not self.client._is_openrouter_circuit_open()
        
        self.client._handle_openrouter_failure(Exception("Test failure 2"))
        assert self.client.openrouter_failure_count == 2
        assert self.client._is_openrouter_circuit_open()
        
        # Test success resets failure count
        self.client._handle_openrouter_success()
        assert self.client.openrouter_failure_count == 0
    
    def test_rate_limiting_tracking(self):
        """Test rate limiting call tracking"""
        self.client.configure_openrouter_only('test_key')
        
        # Verify initial state
        assert len(self.client.openrouter_call_times) == 0
        
        # Simulate some call times
        current_time = time.time()
        self.client.openrouter_call_times = [
            current_time - 30,  # 30 seconds ago
            current_time - 10,  # 10 seconds ago
            current_time - 5    # 5 seconds ago
        ]
        
        # Test usage stats reflect rate limiting
        stats = self.client.get_openrouter_usage_stats()
        assert stats['rate_limit_calls_last_minute'] == 3
    
    def test_provider_selection_openrouter_only(self):
        """Test that OpenRouter-only mode forces correct provider selection"""
        # Test normal mode (should have multiple providers)
        initial_providers = self.client.preferred_providers.copy()
        assert len(initial_providers) > 1
        assert ProviderType.OPENROUTER in initial_providers
        
        # Configure for OpenRouter-only
        self.client.configure_openrouter_only('test_key')
        
        # Test OpenRouter-only mode
        assert self.client.preferred_providers == [ProviderType.OPENROUTER]
    
    @pytest.mark.asyncio
    async def test_make_thinking_session_call_parameters(self):
        """Test thinking session call parameter optimization"""
        self.client.configure_openrouter_only('test_key')
        
        # Mock the make_api_call method to capture parameters
        with patch.object(self.client, 'make_api_call') as mock_make_call:
            mock_make_call.return_value = Mock()
            
            # Test thinking session call
            await self.client.make_thinking_session_call("Test prompt", "high")
            
            # Verify the call was made with correct parameters
            mock_make_call.assert_called_once()
            call_args = mock_make_call.call_args
            
            assert call_args[1]['is_thinking_session'] is True
            assert call_args[1]['provider_type'] == ProviderType.OPENROUTER
            assert 'thinking_mode' in call_args[1]['parameters']
            assert 'temperature' in call_args[1]['parameters']
            assert 'max_tokens' in call_args[1]['parameters']
    
    def test_openrouter_config_to_dict(self):
        """Test OpenRouterConfig serialization"""
        config = OpenRouterConfig(
            api_key='test_key',
            preferred_models=['model1', 'model2'],
            rate_limit_per_minute=30
        )
        
        config_dict = config.to_dict()
        
        # Verify serialization (should not include api_key for security)
        assert 'preferred_models' in config_dict
        assert 'rate_limit_per_minute' in config_dict
        assert 'api_key' not in config_dict  # Should not be serialized
        assert config_dict['preferred_models'] == ['model1', 'model2']
        assert config_dict['rate_limit_per_minute'] == 30