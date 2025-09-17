"""
Per-Core Agent Configuration Validator

This module provides comprehensive validation utilities for per-core agent
configurations, including OpenRouter API key validation, model availability
checks, and system resource validation.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.agent_core import AgentRole

logger = logging.getLogger(__name__)


class PerCoreConfigValidator:
    """Validator for per-core agent configurations"""
    
    # Valid OpenRouter API key pattern
    OPENROUTER_KEY_PATTERN = re.compile(r'^sk-or-[a-zA-Z0-9_-]{20,}$')
    
    # Known placeholder values that should be rejected
    PLACEHOLDER_VALUES = {
        'your_api_key_here', 'your_openrouter_key', 'sk-or-placeholder',
        'test_key', 'dummy_key', 'placeholder', 'example_key', 'sample_key',
        'sk-or-test', 'sk-or-demo', 'sk-or-example'
    }
    
    # Valid thinking modes
    VALID_THINKING_MODES = {'low', 'standard', 'high', 'maximum'}
    
    # Valid fallback modes
    VALID_FALLBACK_MODES = {'single_agent', 'error', 'disabled'}
    
    # Reasonable limits for configuration values
    MAX_AGENTS_LIMIT = 64
    MAX_RATE_LIMIT = 1000
    MAX_CONCURRENT_CALLS = 20
    MIN_HEALTH_CHECK_INTERVAL = 10
    MAX_HEALTH_CHECK_INTERVAL = 3600
    MIN_AGENT_TIMEOUT = 30
    MAX_AGENT_TIMEOUT = 1800
    
    def __init__(self):
        """Initialize the validator"""
        self._model_registry_cache: Optional[Dict[str, List[str]]] = None
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
    
    def validate_openrouter_api_key(self, api_key: str) -> Tuple[bool, List[str]]:
        """
        Validate OpenRouter API key format and content
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not api_key:
            errors.append("OpenRouter API key is required")
            return False, errors
        
        if not isinstance(api_key, str):
            errors.append("OpenRouter API key must be a string")
            return False, errors
        
        # Check for placeholder values
        if api_key.lower() in {p.lower() for p in self.PLACEHOLDER_VALUES}:
            errors.append("OpenRouter API key appears to be a placeholder value")
        
        # Check basic format
        if not self.OPENROUTER_KEY_PATTERN.match(api_key):
            errors.append("OpenRouter API key format is invalid (should start with 'sk-or-' followed by 20+ characters)")
        
        # Check minimum length
        if len(api_key) < 25:  # sk-or- (5) + minimum 20 characters
            errors.append("OpenRouter API key is too short")
        
        # Check for suspicious patterns
        if api_key.count('-') > 10:  # Too many dashes might indicate a malformed key
            errors.append("OpenRouter API key contains too many dashes")
        
        # Check for whitespace
        if api_key != api_key.strip():
            errors.append("OpenRouter API key contains leading or trailing whitespace")
        
        return len(errors) == 0, errors
    
    def validate_preferred_models(self, models: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate preferred models list
        
        Args:
            models: List of model names to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not models:
            errors.append("At least one preferred model must be specified")
            return False, errors
        
        if not isinstance(models, list):
            errors.append("Preferred models must be a list")
            return False, errors
        
        # Check each model
        for i, model in enumerate(models):
            if not isinstance(model, str):
                errors.append(f"Model at index {i} must be a string")
                continue
            
            if not model.strip():
                errors.append(f"Model at index {i} is empty")
                continue
            
            # Check for reasonable model name format
            if len(model) > 100:
                errors.append(f"Model name '{model}' is too long")
            
            # Check for suspicious characters
            if any(char in model for char in ['<', '>', '"', "'"]):
                errors.append(f"Model name '{model}' contains suspicious characters")
        
        # Check for duplicates
        if len(models) != len(set(models)):
            errors.append("Preferred models list contains duplicates")
        
        # Validate against available models if possible
        try:
            available_models = self._get_available_openrouter_models()
            if available_models:
                unavailable_models = [m for m in models if m not in available_models]
                if unavailable_models:
                    logger.warning(f"Some preferred models may not be available: {unavailable_models}")
                    # Don't add to errors as models might be available but not in our registry
        except Exception as e:
            logger.debug(f"Could not validate model availability: {e}")
        
        return len(errors) == 0, errors
    
    def validate_rate_limiting(self, rate_limit: int, concurrent_calls: int) -> Tuple[bool, List[str]]:
        """
        Validate rate limiting configuration
        
        Args:
            rate_limit: Rate limit per minute
            concurrent_calls: Maximum concurrent calls
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate rate limit
        if not isinstance(rate_limit, int):
            errors.append("Rate limit per minute must be an integer")
        elif rate_limit <= 0:
            errors.append("Rate limit per minute must be positive")
        elif rate_limit > self.MAX_RATE_LIMIT:
            errors.append(f"Rate limit per minute is too high (maximum: {self.MAX_RATE_LIMIT})")
        
        # Validate concurrent calls
        if not isinstance(concurrent_calls, int):
            errors.append("Max concurrent calls must be an integer")
        elif concurrent_calls <= 0:
            errors.append("Max concurrent calls must be positive")
        elif concurrent_calls > self.MAX_CONCURRENT_CALLS:
            errors.append(f"Max concurrent calls is too high (maximum: {self.MAX_CONCURRENT_CALLS})")
        
        # Check relationship between rate limit and concurrent calls
        if isinstance(rate_limit, int) and isinstance(concurrent_calls, int):
            if concurrent_calls > rate_limit:
                errors.append("Max concurrent calls should not exceed rate limit per minute")
        
        return len(errors) == 0, errors
    
    def validate_temperature_range(self, temp_range: Tuple[float, float]) -> Tuple[bool, List[str]]:
        """
        Validate temperature range configuration
        
        Args:
            temp_range: Tuple of (min_temp, max_temp)
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(temp_range, (tuple, list)):
            errors.append("Temperature range must be a tuple or list")
            return False, errors
        
        if len(temp_range) != 2:
            errors.append("Temperature range must contain exactly two values")
            return False, errors
        
        min_temp, max_temp = temp_range
        
        # Validate individual values
        if not isinstance(min_temp, (int, float)):
            errors.append("Minimum temperature must be a number")
        elif min_temp < 0.0:
            errors.append("Minimum temperature cannot be negative")
        elif min_temp > 2.0:
            errors.append("Minimum temperature cannot exceed 2.0")
        
        if not isinstance(max_temp, (int, float)):
            errors.append("Maximum temperature must be a number")
        elif max_temp < 0.0:
            errors.append("Maximum temperature cannot be negative")
        elif max_temp > 2.0:
            errors.append("Maximum temperature cannot exceed 2.0")
        
        # Validate relationship
        if isinstance(min_temp, (int, float)) and isinstance(max_temp, (int, float)):
            if min_temp >= max_temp:
                errors.append("Minimum temperature must be less than maximum temperature")
            
            if max_temp - min_temp > 1.5:
                errors.append("Temperature range is too wide (maximum range: 1.5)")
        
        return len(errors) == 0, errors
    
    def validate_thinking_mode(self, thinking_mode: str) -> Tuple[bool, List[str]]:
        """
        Validate thinking mode setting
        
        Args:
            thinking_mode: The thinking mode to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(thinking_mode, str):
            errors.append("Thinking mode must be a string")
        elif thinking_mode not in self.VALID_THINKING_MODES:
            errors.append(f"Thinking mode must be one of: {', '.join(sorted(self.VALID_THINKING_MODES))}")
        
        return len(errors) == 0, errors
    
    def validate_system_settings(self, max_agents: Optional[int], health_check_interval: int,
                                agent_timeout: int, fallback_mode: str) -> Tuple[bool, List[str]]:
        """
        Validate system-level settings
        
        Args:
            max_agents: Maximum number of agents (None for auto-detect)
            health_check_interval: Health check interval in seconds
            agent_timeout: Agent timeout in seconds
            fallback_mode: Fallback mode setting
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate max agents
        if max_agents is not None:
            if not isinstance(max_agents, int):
                errors.append("Max agents must be an integer or None")
            elif max_agents <= 0:
                errors.append("Max agents must be positive")
            elif max_agents > self.MAX_AGENTS_LIMIT:
                errors.append(f"Max agents is too high (maximum: {self.MAX_AGENTS_LIMIT})")
        
        # Validate health check interval
        if not isinstance(health_check_interval, int):
            errors.append("Health check interval must be an integer")
        elif health_check_interval < self.MIN_HEALTH_CHECK_INTERVAL:
            errors.append(f"Health check interval is too low (minimum: {self.MIN_HEALTH_CHECK_INTERVAL}s)")
        elif health_check_interval > self.MAX_HEALTH_CHECK_INTERVAL:
            errors.append(f"Health check interval is too high (maximum: {self.MAX_HEALTH_CHECK_INTERVAL}s)")
        
        # Validate agent timeout
        if not isinstance(agent_timeout, int):
            errors.append("Agent timeout must be an integer")
        elif agent_timeout < self.MIN_AGENT_TIMEOUT:
            errors.append(f"Agent timeout is too low (minimum: {self.MIN_AGENT_TIMEOUT}s)")
        elif agent_timeout > self.MAX_AGENT_TIMEOUT:
            errors.append(f"Agent timeout is too high (maximum: {self.MAX_AGENT_TIMEOUT}s)")
        
        # Validate fallback mode
        if not isinstance(fallback_mode, str):
            errors.append("Fallback mode must be a string")
        elif fallback_mode not in self.VALID_FALLBACK_MODES:
            errors.append(f"Fallback mode must be one of: {', '.join(sorted(self.VALID_FALLBACK_MODES))}")
        
        # Check relationships
        if (isinstance(health_check_interval, int) and isinstance(agent_timeout, int) and
            health_check_interval >= agent_timeout):
            errors.append("Health check interval should be less than agent timeout")
        
        return len(errors) == 0, errors
    
    def validate_role_configuration(self, role_name: str, role_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration for a specific agent role
        
        Args:
            role_name: Name of the agent role
            role_config: Configuration dictionary for the role
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate role name
        try:
            AgentRole(role_name)
        except ValueError:
            errors.append(f"Invalid agent role: {role_name}")
            return False, errors
        
        # Validate required fields
        required_fields = ['api_key', 'preferred_models']
        for field in required_fields:
            if field not in role_config:
                errors.append(f"Role {role_name}: Missing required field '{field}'")
        
        # Validate API key
        if 'api_key' in role_config:
            is_valid, api_key_errors = self.validate_openrouter_api_key(role_config['api_key'])
            for error in api_key_errors:
                errors.append(f"Role {role_name}: {error}")
        
        # Validate preferred models
        if 'preferred_models' in role_config:
            is_valid, model_errors = self.validate_preferred_models(role_config['preferred_models'])
            for error in model_errors:
                errors.append(f"Role {role_name}: {error}")
        
        # Validate rate limiting
        rate_limit = role_config.get('rate_limit_per_minute', 60)
        concurrent_calls = role_config.get('max_concurrent_calls', 3)
        is_valid, rate_errors = self.validate_rate_limiting(rate_limit, concurrent_calls)
        for error in rate_errors:
            errors.append(f"Role {role_name}: {error}")
        
        # Validate temperature range
        temp_range = role_config.get('temperature_range', (0.3, 0.8))
        is_valid, temp_errors = self.validate_temperature_range(temp_range)
        for error in temp_errors:
            errors.append(f"Role {role_name}: {error}")
        
        # Validate thinking mode
        thinking_mode = role_config.get('thinking_mode_default', 'standard')
        is_valid, thinking_errors = self.validate_thinking_mode(thinking_mode)
        for error in thinking_errors:
            errors.append(f"Role {role_name}: {error}")
        
        return len(errors) == 0, errors
    
    def validate_environment_variables(self) -> Tuple[bool, List[str]]:
        """
        Validate environment variables related to per-core agent configuration
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check for OpenRouter API key
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            is_valid, key_errors = self.validate_openrouter_api_key(openrouter_key)
            if not is_valid:
                for error in key_errors:
                    warnings.append(f"OPENROUTER_API_KEY: {error}")
        else:
            warnings.append("OPENROUTER_API_KEY not set - per-core agents will use fallback providers")
        
        # Check numeric environment variables
        numeric_env_vars = {
            'PER_CORE_MAX_AGENTS': (1, self.MAX_AGENTS_LIMIT),
            'PER_CORE_HEALTH_CHECK_INTERVAL': (self.MIN_HEALTH_CHECK_INTERVAL, self.MAX_HEALTH_CHECK_INTERVAL),
            'PER_CORE_AGENT_TIMEOUT': (self.MIN_AGENT_TIMEOUT, self.MAX_AGENT_TIMEOUT)
        }
        
        for env_var, (min_val, max_val) in numeric_env_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    num_value = int(value)
                    if num_value < min_val or num_value > max_val:
                        warnings.append(f"{env_var}: Value {num_value} is outside recommended range [{min_val}, {max_val}]")
                except ValueError:
                    warnings.append(f"{env_var}: Invalid integer value '{value}'")
        
        # Check boolean environment variables
        boolean_env_vars = ['ENABLE_PER_CORE_AGENTS', 'PER_CORE_OPENROUTER_REQUIRED']
        for env_var in boolean_env_vars:
            value = os.getenv(env_var)
            if value and value.lower() not in ('true', 'false', '1', '0', 'yes', 'no', 'on', 'off'):
                warnings.append(f"{env_var}: Unclear boolean value '{value}' (use 'true'/'false')")
        
        # Check fallback mode
        fallback_mode = os.getenv('PER_CORE_FALLBACK_MODE')
        if fallback_mode and fallback_mode.lower() not in self.VALID_FALLBACK_MODES:
            warnings.append(f"PER_CORE_FALLBACK_MODE: Invalid value '{fallback_mode}' (use one of: {', '.join(self.VALID_FALLBACK_MODES)})")
        
        return len(warnings) == 0, warnings
    
    def _get_available_openrouter_models(self) -> Optional[List[str]]:
        """
        Get list of available OpenRouter models from the provider registry
        
        Returns:
            List of available model names or None if unavailable
        """
        import time
        
        # Check cache
        current_time = time.time()
        if (self._model_registry_cache and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._model_registry_cache.get('openrouter', [])
        
        try:
            from providers.registry import ModelProviderRegistry
            from providers.base import ProviderType
            
            registry = ModelProviderRegistry()
            openrouter_models = registry.get_available_model_names(ProviderType.OPENROUTER)
            
            # Update cache
            self._model_registry_cache = {'openrouter': openrouter_models}
            self._cache_timestamp = current_time
            
            return openrouter_models
            
        except Exception as e:
            logger.debug(f"Could not get available OpenRouter models: {e}")
            return None


# Global validator instance
_validator: Optional[PerCoreConfigValidator] = None


def get_per_core_config_validator() -> PerCoreConfigValidator:
    """Get the global per-core configuration validator"""
    global _validator
    
    if _validator is None:
        _validator = PerCoreConfigValidator()
    
    return _validator