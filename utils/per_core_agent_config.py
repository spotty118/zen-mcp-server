"""
Per-Core Agent Configuration Management

This module provides secure configuration management for the per-core agent system,
including OpenRouter API key handling, configuration validation, and hot-reload
capabilities for runtime adjustments.

Key Features:
- Secure OpenRouter API key management with environment variable support
- Configuration validation for agent settings and OpenRouter parameters
- Configuration hot-reload capabilities for runtime adjustments
- Role-specific configuration templates and validation
- Configuration persistence and backup
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from utils.agent_core import AgentRole

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterAgentConfig:
    """OpenRouter configuration for a specific agent"""
    api_key: str
    preferred_models: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    max_concurrent_calls: int = 3
    thinking_mode_default: str = "high"
    temperature_range: Tuple[float, float] = (0.3, 0.8)
    fallback_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate the OpenRouter configuration and return any errors"""
        errors = []
        
        # Validate API key
        if not self.api_key or not isinstance(self.api_key, str):
            errors.append("OpenRouter API key is required and must be a string")
        elif len(self.api_key) < 20:
            errors.append("OpenRouter API key appears too short")
        elif self.api_key.lower() in ['your_api_key_here', 'placeholder', 'test_key']:
            errors.append("OpenRouter API key appears to be a placeholder")
        
        # Validate preferred models
        if not self.preferred_models:
            errors.append("At least one preferred model must be specified")
        elif not isinstance(self.preferred_models, list):
            errors.append("Preferred models must be a list")
        
        # Validate rate limiting
        if not isinstance(self.rate_limit_per_minute, int) or self.rate_limit_per_minute <= 0:
            errors.append("Rate limit per minute must be a positive integer")
        elif self.rate_limit_per_minute > 1000:
            errors.append("Rate limit per minute seems unreasonably high (>1000)")
        
        # Validate concurrent calls
        if not isinstance(self.max_concurrent_calls, int) or self.max_concurrent_calls <= 0:
            errors.append("Max concurrent calls must be a positive integer")
        elif self.max_concurrent_calls > 20:
            errors.append("Max concurrent calls seems unreasonably high (>20)")
        
        # Validate thinking mode
        valid_thinking_modes = ["low", "standard", "high", "maximum"]
        if self.thinking_mode_default not in valid_thinking_modes:
            errors.append(f"Thinking mode must be one of: {valid_thinking_modes}")
        
        # Validate temperature range
        if not isinstance(self.temperature_range, (tuple, list)) or len(self.temperature_range) != 2:
            errors.append("Temperature range must be a tuple/list of two values")
        else:
            min_temp, max_temp = self.temperature_range
            if not isinstance(min_temp, (int, float)) or not isinstance(max_temp, (int, float)):
                errors.append("Temperature range values must be numbers")
            elif min_temp < 0.0 or max_temp > 2.0:
                errors.append("Temperature range must be between 0.0 and 2.0")
            elif min_temp >= max_temp:
                errors.append("Temperature range minimum must be less than maximum")
        
        # Validate circuit breaker settings
        if not isinstance(self.circuit_breaker_threshold, int) or self.circuit_breaker_threshold <= 0:
            errors.append("Circuit breaker threshold must be a positive integer")
        
        if not isinstance(self.circuit_breaker_timeout, (int, float)) or self.circuit_breaker_timeout <= 0:
            errors.append("Circuit breaker timeout must be a positive number")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenRouterAgentConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class PerCoreAgentSystemConfig:
    """Complete configuration for the per-core agent system"""
    enabled: bool = True
    max_agents: Optional[int] = None
    openrouter_required: bool = False
    fallback_mode: str = "single_agent"
    health_check_interval: int = 60
    agent_timeout: int = 300
    emergency_mode_threshold: int = 2  # Number of failed agents before emergency mode
    
    # OpenRouter configuration
    openrouter_api_key: Optional[str] = None
    openrouter_referer: str = "https://github.com/BeehiveInnovations/zen-mcp-server"
    openrouter_title: str = "Zen MCP Server"
    
    # Role-specific configurations
    role_configs: Dict[str, OpenRouterAgentConfig] = field(default_factory=dict)
    
    # Multi-agent tool settings
    multi_agent_tools: Set[str] = field(default_factory=lambda: {
        "consensus", "thinkdeep", "parallelthink", "codereview", "secaudit", "analyze"
    })
    
    def validate(self) -> List[str]:
        """Validate the complete system configuration"""
        errors = []
        
        # Validate basic settings
        if not isinstance(self.enabled, bool):
            errors.append("Enabled flag must be a boolean")
        
        if self.max_agents is not None:
            if not isinstance(self.max_agents, int) or self.max_agents <= 0:
                errors.append("Max agents must be a positive integer or None")
            elif self.max_agents > 64:
                errors.append("Max agents seems unreasonably high (>64)")
        
        # Validate fallback mode
        valid_fallback_modes = ["single_agent", "error", "disabled"]
        if self.fallback_mode not in valid_fallback_modes:
            errors.append(f"Fallback mode must be one of: {valid_fallback_modes}")
        
        # Validate intervals and timeouts
        if not isinstance(self.health_check_interval, int) or self.health_check_interval <= 0:
            errors.append("Health check interval must be a positive integer")
        
        if not isinstance(self.agent_timeout, int) or self.agent_timeout <= 0:
            errors.append("Agent timeout must be a positive integer")
        
        if not isinstance(self.emergency_mode_threshold, int) or self.emergency_mode_threshold <= 0:
            errors.append("Emergency mode threshold must be a positive integer")
        
        # Validate OpenRouter settings if required
        if self.openrouter_required and not self.openrouter_api_key:
            errors.append("OpenRouter API key is required when openrouter_required is True")
        
        # Validate role configurations
        for role_name, role_config in self.role_configs.items():
            try:
                # Validate role name
                AgentRole(role_name)
            except ValueError:
                errors.append(f"Invalid agent role: {role_name}")
                continue
            
            # Validate role configuration
            role_errors = role_config.validate()
            for error in role_errors:
                errors.append(f"Role {role_name}: {error}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert set to list for JSON serialization
        data['multi_agent_tools'] = list(self.multi_agent_tools)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerCoreAgentSystemConfig':
        """Create from dictionary"""
        # Convert list back to set
        if 'multi_agent_tools' in data and isinstance(data['multi_agent_tools'], list):
            data['multi_agent_tools'] = set(data['multi_agent_tools'])
        
        # Convert role configs
        if 'role_configs' in data:
            role_configs = {}
            for role_name, role_data in data['role_configs'].items():
                role_configs[role_name] = OpenRouterAgentConfig.from_dict(role_data)
            data['role_configs'] = role_configs
        
        return cls(**data)


class PerCoreAgentConfigManager:
    """
    Configuration manager for the per-core agent system with hot-reload capabilities
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or os.path.join(
            os.path.expanduser("~"), ".zen_mcp", "per_core_agent_config.json"
        )
        self.config_dir = os.path.dirname(self.config_file)
        
        # Configuration state
        self._config: Optional[PerCoreAgentSystemConfig] = None
        self._config_lock = threading.RLock()
        self._last_modified = 0.0
        
        # Hot-reload settings
        self._hot_reload_enabled = True
        self._hot_reload_interval = 5.0  # Check every 5 seconds
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        # Change callbacks
        self._change_callbacks: List[callable] = []
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        logger.info(f"PerCoreAgentConfigManager initialized with config file: {self.config_file}")
    
    def load_config(self, create_default: bool = True) -> PerCoreAgentSystemConfig:
        """
        Load configuration from file and environment variables
        
        Args:
            create_default: Whether to create default config if file doesn't exist
            
        Returns:
            Loaded configuration
        """
        with self._config_lock:
            try:
                # Try to load from file first
                if os.path.exists(self.config_file):
                    with open(self.config_file, 'r') as f:
                        file_data = json.load(f)
                    
                    config = PerCoreAgentSystemConfig.from_dict(file_data)
                    self._last_modified = os.path.getmtime(self.config_file)
                    logger.info(f"Loaded per-core agent configuration from {self.config_file}")
                    
                elif create_default:
                    # Create default configuration
                    config = self._create_default_config()
                    logger.info("Created default per-core agent configuration")
                    
                else:
                    raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
                
                # Override with environment variables
                config = self._apply_environment_overrides(config)
                
                # Validate configuration
                errors = config.validate()
                if errors:
                    error_msg = f"Configuration validation failed: {'; '.join(errors)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                self._config = config
                return config
                
            except Exception as e:
                logger.error(f"Failed to load per-core agent configuration: {e}")
                if self._config is None:
                    # Return minimal safe configuration
                    self._config = PerCoreAgentSystemConfig(enabled=False)
                return self._config
    
    def save_config(self, config: Optional[PerCoreAgentSystemConfig] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config: Configuration to save (uses current config if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        with self._config_lock:
            try:
                config_to_save = config or self._config
                if not config_to_save:
                    logger.error("No configuration to save")
                    return False
                
                # Validate before saving
                errors = config_to_save.validate()
                if errors:
                    logger.error(f"Cannot save invalid configuration: {'; '.join(errors)}")
                    return False
                
                # Create backup of existing config
                if os.path.exists(self.config_file):
                    backup_file = f"{self.config_file}.backup.{int(time.time())}"
                    os.rename(self.config_file, backup_file)
                    logger.debug(f"Created backup: {backup_file}")
                
                # Save new configuration
                with open(self.config_file, 'w') as f:
                    json.dump(config_to_save.to_dict(), f, indent=2)
                
                self._config = config_to_save
                self._last_modified = os.path.getmtime(self.config_file)
                
                logger.info(f"Saved per-core agent configuration to {self.config_file}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save per-core agent configuration: {e}")
                return False
    
    def get_config(self) -> PerCoreAgentSystemConfig:
        """Get current configuration, loading if necessary"""
        with self._config_lock:
            if self._config is None:
                return self.load_config()
            return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self._config_lock:
            try:
                current_config = self.get_config()
                config_dict = current_config.to_dict()
                
                # Apply updates
                self._deep_update(config_dict, updates)
                
                # Create new configuration
                new_config = PerCoreAgentSystemConfig.from_dict(config_dict)
                
                # Validate new configuration
                errors = new_config.validate()
                if errors:
                    logger.error(f"Configuration update validation failed: {'; '.join(errors)}")
                    return False
                
                # Save and apply new configuration
                if self.save_config(new_config):
                    self._notify_config_changed(current_config, new_config)
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                return False
    
    def start_hot_reload(self) -> None:
        """Start hot-reload monitoring thread"""
        if self._hot_reload_thread and self._hot_reload_thread.is_alive():
            logger.warning("Hot-reload thread already running")
            return
        
        self._shutdown = False
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_worker,
            name="PerCoreAgentConfigHotReload",
            daemon=True
        )
        self._hot_reload_thread.start()
        logger.info("Started per-core agent configuration hot-reload monitoring")
    
    def stop_hot_reload(self) -> None:
        """Stop hot-reload monitoring thread"""
        self._shutdown = True
        if self._hot_reload_thread and self._hot_reload_thread.is_alive():
            self._hot_reload_thread.join(timeout=10)
        logger.info("Stopped per-core agent configuration hot-reload monitoring")
    
    def add_change_callback(self, callback: callable) -> None:
        """Add callback to be notified of configuration changes"""
        self._change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: callable) -> None:
        """Remove configuration change callback"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
    
    def get_openrouter_config_for_role(self, role: AgentRole) -> Optional[OpenRouterAgentConfig]:
        """
        Get OpenRouter configuration for a specific agent role
        
        Args:
            role: The agent role
            
        Returns:
            OpenRouter configuration or None if not configured
        """
        config = self.get_config()
        
        # Check for role-specific configuration
        role_config = config.role_configs.get(role.value)
        if role_config:
            return role_config
        
        # Check for default configuration
        if config.openrouter_api_key:
            return self._create_default_openrouter_config_for_role(role, config.openrouter_api_key)
        
        return None
    
    def validate_openrouter_api_key(self, api_key: str) -> Tuple[bool, List[str]]:
        """
        Validate OpenRouter API key
        
        Args:
            api_key: The API key to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not api_key or not isinstance(api_key, str):
            errors.append("API key is required and must be a string")
            return False, errors
        
        # Basic format validation
        if not api_key.startswith('sk-or-'):
            errors.append("OpenRouter API key should start with 'sk-or-'")
        
        # Length validation
        if len(api_key) < 20:
            errors.append("API key appears too short")
        
        # Placeholder detection
        placeholder_values = [
            'your_api_key_here', 'your_openrouter_key', 'sk-or-placeholder',
            'test_key', 'dummy_key', 'placeholder'
        ]
        
        if api_key.lower() in [p.lower() for p in placeholder_values]:
            errors.append("API key appears to be a placeholder")
        
        return len(errors) == 0, errors
    
    def _create_default_config(self) -> PerCoreAgentSystemConfig:
        """Create default configuration"""
        # Get OpenRouter API key from environment
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        config = PerCoreAgentSystemConfig(
            enabled=os.getenv('ENABLE_PER_CORE_AGENTS', 'true').lower() in ('true', '1', 'yes', 'on'),
            max_agents=self._get_env_int('PER_CORE_MAX_AGENTS'),
            openrouter_required=os.getenv('PER_CORE_OPENROUTER_REQUIRED', 'false').lower() in ('true', '1', 'yes', 'on'),
            fallback_mode=os.getenv('PER_CORE_FALLBACK_MODE', 'single_agent').lower(),
            health_check_interval=self._get_env_int('PER_CORE_HEALTH_CHECK_INTERVAL') or 60,
            agent_timeout=self._get_env_int('PER_CORE_AGENT_TIMEOUT') or 300,
            openrouter_api_key=openrouter_api_key,
            openrouter_referer=os.getenv('OPENROUTER_REFERER', 'https://github.com/BeehiveInnovations/zen-mcp-server'),
            openrouter_title=os.getenv('OPENROUTER_TITLE', 'Zen MCP Server')
        )
        
        # Create default role configurations if OpenRouter key is available
        if openrouter_api_key:
            config.role_configs = self._create_default_role_configs(openrouter_api_key)
        
        return config
    
    def _create_default_role_configs(self, api_key: str) -> Dict[str, OpenRouterAgentConfig]:
        """Create default role-specific configurations"""
        role_configs = {}
        
        # Role-specific model preferences and settings
        role_settings = {
            AgentRole.SECURITY_ANALYST: {
                "models": ["openai/o3", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
                "rate_limit": 45, "concurrent": 2, "thinking_mode": "high", "temp_range": (0.2, 0.6)
            },
            AgentRole.PERFORMANCE_OPTIMIZER: {
                "models": ["google/gemini-2.5-pro", "openai/o3", "anthropic/claude-3-sonnet"],
                "rate_limit": 60, "concurrent": 3, "thinking_mode": "high", "temp_range": (0.3, 0.7)
            },
            AgentRole.ARCHITECTURE_REVIEWER: {
                "models": ["openai/o3-pro", "anthropic/claude-3-opus", "openai/o3"],
                "rate_limit": 40, "concurrent": 2, "thinking_mode": "high", "temp_range": (0.4, 0.8)
            },
            AgentRole.CODE_QUALITY_INSPECTOR: {
                "models": ["openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet"],
                "rate_limit": 60, "concurrent": 3, "thinking_mode": "standard", "temp_range": (0.2, 0.5)
            },
            AgentRole.DEBUG_SPECIALIST: {
                "models": ["openai/o3", "anthropic/claude-3-sonnet", "google/gemini-2.5-pro"],
                "rate_limit": 70, "concurrent": 4, "thinking_mode": "standard", "temp_range": (0.3, 0.6)
            },
            AgentRole.PLANNING_COORDINATOR: {
                "models": ["openai/o3-pro", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
                "rate_limit": 50, "concurrent": 3, "thinking_mode": "high", "temp_range": (0.5, 0.8)
            },
            AgentRole.CONSENSUS_FACILITATOR: {
                "models": ["anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro"],
                "rate_limit": 55, "concurrent": 3, "thinking_mode": "high", "temp_range": (0.4, 0.7)
            },
            AgentRole.GENERALIST: {
                "models": ["google/gemini-2.5-flash", "openai/o3-mini", "anthropic/claude-3-haiku"],
                "rate_limit": 80, "concurrent": 4, "thinking_mode": "standard", "temp_range": (0.4, 0.7)
            }
        }
        
        for role, settings in role_settings.items():
            role_configs[role.value] = OpenRouterAgentConfig(
                api_key=api_key,
                preferred_models=settings["models"],
                rate_limit_per_minute=settings["rate_limit"],
                max_concurrent_calls=settings["concurrent"],
                thinking_mode_default=settings["thinking_mode"],
                temperature_range=settings["temp_range"]
            )
        
        return role_configs
    
    def _create_default_openrouter_config_for_role(self, role: AgentRole, api_key: str) -> OpenRouterAgentConfig:
        """Create default OpenRouter configuration for a role"""
        default_configs = self._create_default_role_configs(api_key)
        return default_configs.get(role.value, OpenRouterAgentConfig(
            api_key=api_key,
            preferred_models=["google/gemini-2.5-flash", "openai/o3-mini"],
            rate_limit_per_minute=60,
            max_concurrent_calls=3,
            thinking_mode_default="standard",
            temperature_range=(0.4, 0.7)
        ))
    
    def _apply_environment_overrides(self, config: PerCoreAgentSystemConfig) -> PerCoreAgentSystemConfig:
        """Apply environment variable overrides to configuration"""
        # Override basic settings
        if 'ENABLE_PER_CORE_AGENTS' in os.environ:
            config.enabled = os.getenv('ENABLE_PER_CORE_AGENTS', 'true').lower() in ('true', '1', 'yes', 'on')
        
        if 'PER_CORE_MAX_AGENTS' in os.environ:
            max_agents_value = self._get_env_int('PER_CORE_MAX_AGENTS')
            if max_agents_value is not None:
                config.max_agents = max_agents_value
        
        if 'PER_CORE_OPENROUTER_REQUIRED' in os.environ:
            config.openrouter_required = os.getenv('PER_CORE_OPENROUTER_REQUIRED', 'false').lower() in ('true', '1', 'yes', 'on')
        
        if 'PER_CORE_FALLBACK_MODE' in os.environ:
            config.fallback_mode = os.getenv('PER_CORE_FALLBACK_MODE', 'single_agent').lower()
        
        if 'PER_CORE_HEALTH_CHECK_INTERVAL' in os.environ:
            health_check_value = self._get_env_int('PER_CORE_HEALTH_CHECK_INTERVAL', 60)
            if health_check_value is not None:
                config.health_check_interval = health_check_value
        
        if 'PER_CORE_AGENT_TIMEOUT' in os.environ:
            agent_timeout_value = self._get_env_int('PER_CORE_AGENT_TIMEOUT', 300)
            if agent_timeout_value is not None:
                config.agent_timeout = agent_timeout_value
        
        # Override OpenRouter settings
        if 'OPENROUTER_API_KEY' in os.environ:
            api_key = os.getenv('OPENROUTER_API_KEY')
            if api_key:
                config.openrouter_api_key = api_key
                # Also update role configs if they exist
                if config.role_configs:
                    for role_config in config.role_configs.values():
                        role_config.api_key = api_key
                elif api_key:
                    # Create default role configs with the new API key
                    config.role_configs = self._create_default_role_configs(api_key)
        
        if 'OPENROUTER_REFERER' in os.environ:
            referer = os.getenv('OPENROUTER_REFERER')
            if referer:
                config.openrouter_referer = referer
        
        if 'OPENROUTER_TITLE' in os.environ:
            title = os.getenv('OPENROUTER_TITLE')
            if title:
                config.openrouter_title = title
        
        return config
    
    def _get_env_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get integer value from environment variable"""
        value = os.getenv(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for {key}: {value}")
            return default
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update dictionary with another dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _hot_reload_worker(self) -> None:
        """Worker thread for hot-reload monitoring"""
        while not self._shutdown:
            try:
                if os.path.exists(self.config_file):
                    current_modified = os.path.getmtime(self.config_file)
                    
                    if current_modified > self._last_modified:
                        logger.info("Configuration file changed, reloading...")
                        
                        old_config = self._config
                        new_config = self.load_config(create_default=False)
                        
                        if new_config != old_config:
                            self._notify_config_changed(old_config, new_config)
                
                time.sleep(self._hot_reload_interval)
                
            except Exception as e:
                logger.error(f"Error in hot-reload worker: {e}")
                time.sleep(self._hot_reload_interval)
    
    def _notify_config_changed(self, old_config: Optional[PerCoreAgentSystemConfig], 
                              new_config: PerCoreAgentSystemConfig) -> None:
        """Notify callbacks of configuration changes"""
        for callback in self._change_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")


# Global configuration manager instance
_config_manager: Optional[PerCoreAgentConfigManager] = None
_config_manager_lock = threading.Lock()


def get_per_core_agent_config_manager() -> PerCoreAgentConfigManager:
    """Get the global per-core agent configuration manager"""
    global _config_manager
    
    with _config_manager_lock:
        if _config_manager is None:
            _config_manager = PerCoreAgentConfigManager()
            _config_manager.start_hot_reload()
        
        return _config_manager


def shutdown_config_manager() -> None:
    """Shutdown the global configuration manager"""
    global _config_manager
    
    with _config_manager_lock:
        if _config_manager:
            _config_manager.stop_hot_reload()
            _config_manager = None