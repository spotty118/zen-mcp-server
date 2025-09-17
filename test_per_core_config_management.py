#!/usr/bin/env python3
"""
Test script for per-core agent configuration management

This script tests the configuration management system including:
- Configuration creation and validation
- OpenRouter API key handling
- Hot-reload capabilities
- Configuration persistence
"""

import json
import os
import tempfile
import time
from pathlib import Path

from utils.per_core_agent_config import (
    PerCoreAgentConfigManager,
    PerCoreAgentSystemConfig,
    OpenRouterAgentConfig
)
from utils.per_core_config_validator import get_per_core_config_validator
from utils.agent_core import AgentRole


def test_basic_configuration():
    """Test basic configuration creation and validation"""
    print("🧪 Testing basic configuration creation and validation...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_file = f.name
    
    try:
        # Create config manager
        config_manager = PerCoreAgentConfigManager(temp_config_file)
        
        # Load default configuration
        config = config_manager.load_config(create_default=True)
        
        # Validate configuration
        errors = config.validate()
        if errors:
            print(f"❌ Default configuration validation failed: {errors}")
            return False
        
        print("✅ Default configuration created and validated successfully")
        
        # Test configuration saving
        if config_manager.save_config(config):
            print("✅ Configuration saved successfully")
        else:
            print("❌ Configuration save failed")
            return False
        
        # Test configuration loading
        loaded_config = config_manager.load_config(create_default=False)
        if loaded_config.enabled == config.enabled:
            print("✅ Configuration loaded successfully")
        else:
            print("❌ Configuration load failed")
            return False
        
        return True
        
    finally:
        # Clean up
        if os.path.exists(temp_config_file):
            os.unlink(temp_config_file)


def test_openrouter_api_key_validation():
    """Test OpenRouter API key validation"""
    print("\n🧪 Testing OpenRouter API key validation...")
    
    validator = get_per_core_config_validator()
    
    # Test valid API key format
    valid_key = "sk-or-abcdefghijklmnopqrstuvwxyz1234567890"
    is_valid, errors = validator.validate_openrouter_api_key(valid_key)
    if is_valid:
        print("✅ Valid API key format accepted")
    else:
        print(f"❌ Valid API key rejected: {errors}")
        return False
    
    # Test invalid API key formats
    invalid_keys = [
        "",  # Empty
        "invalid-key",  # Wrong format
        "sk-or-short",  # Too short
        "your_api_key_here",  # Placeholder
        "sk-or-placeholder"  # Placeholder
    ]
    
    for invalid_key in invalid_keys:
        is_valid, errors = validator.validate_openrouter_api_key(invalid_key)
        if not is_valid:
            print(f"✅ Invalid API key '{invalid_key}' correctly rejected")
        else:
            print(f"❌ Invalid API key '{invalid_key}' incorrectly accepted")
            return False
    
    return True


def test_role_configuration():
    """Test role-specific configuration"""
    print("\n🧪 Testing role-specific configuration...")
    
    # Create a test OpenRouter configuration
    test_api_key = "sk-or-test1234567890abcdefghijklmnopqrstuvwxyz"
    
    role_config = OpenRouterAgentConfig(
        api_key=test_api_key,
        preferred_models=["openai/o3", "anthropic/claude-3-opus"],
        rate_limit_per_minute=60,
        max_concurrent_calls=3,
        thinking_mode_default="high",
        temperature_range=(0.3, 0.8)
    )
    
    # Validate role configuration
    errors = role_config.validate()
    if errors:
        print(f"❌ Role configuration validation failed: {errors}")
        return False
    
    print("✅ Role configuration validated successfully")
    
    # Test serialization
    config_dict = role_config.to_dict()
    restored_config = OpenRouterAgentConfig.from_dict(config_dict)
    
    if restored_config.api_key == role_config.api_key:
        print("✅ Role configuration serialization works")
    else:
        print("❌ Role configuration serialization failed")
        return False
    
    return True


def test_configuration_updates():
    """Test configuration updates and validation"""
    print("\n🧪 Testing configuration updates...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_file = f.name
    
    try:
        config_manager = PerCoreAgentConfigManager(temp_config_file)
        
        # Create initial configuration
        config = config_manager.load_config(create_default=True)
        config.enabled = True
        config.max_agents = 4
        config_manager.save_config(config)
        
        # Test configuration update
        updates = {
            'max_agents': 8,
            'health_check_interval': 120,
            'agent_timeout': 600
        }
        
        if config_manager.update_config(updates):
            print("✅ Configuration update successful")
            
            # Verify updates were applied
            updated_config = config_manager.get_config()
            if (updated_config.max_agents == 8 and 
                updated_config.health_check_interval == 120 and
                updated_config.agent_timeout == 600):
                print("✅ Configuration updates applied correctly")
            else:
                print("❌ Configuration updates not applied correctly")
                return False
        else:
            print("❌ Configuration update failed")
            return False
        
        # Test invalid update
        invalid_updates = {
            'max_agents': -1,  # Invalid value
            'fallback_mode': 'invalid_mode'  # Invalid mode
        }
        
        if not config_manager.update_config(invalid_updates):
            print("✅ Invalid configuration update correctly rejected")
        else:
            print("❌ Invalid configuration update incorrectly accepted")
            return False
        
        return True
        
    finally:
        if os.path.exists(temp_config_file):
            os.unlink(temp_config_file)


def test_hot_reload():
    """Test configuration hot-reload functionality"""
    print("\n🧪 Testing configuration hot-reload...")
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_file = f.name
    
    try:
        config_manager = PerCoreAgentConfigManager(temp_config_file)
        
        # Create initial configuration
        config = config_manager.load_config(create_default=True)
        config.enabled = True
        config.max_agents = 4
        config_manager.save_config(config)
        
        # Track configuration changes
        change_detected = False
        old_config_ref = None
        new_config_ref = None
        
        def config_change_callback(old_config, new_config):
            nonlocal change_detected, old_config_ref, new_config_ref
            change_detected = True
            old_config_ref = old_config
            new_config_ref = new_config
        
        config_manager.add_change_callback(config_change_callback)
        
        # Start hot-reload monitoring
        config_manager.start_hot_reload()
        
        # Wait a moment for monitoring to start
        time.sleep(0.5)
        
        # Modify configuration file directly
        config_dict = config.to_dict()
        config_dict['max_agents'] = 6
        config_dict['health_check_interval'] = 90
        
        with open(temp_config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Wait for hot-reload to detect change
        time.sleep(6)  # Hot-reload interval is 5 seconds
        
        # Stop hot-reload monitoring
        config_manager.stop_hot_reload()
        
        if change_detected:
            print("✅ Configuration hot-reload detected changes")
            
            # Verify the new configuration
            current_config = config_manager.get_config()
            if current_config.max_agents == 6 and current_config.health_check_interval == 90:
                print("✅ Hot-reload applied changes correctly")
            else:
                print("❌ Hot-reload did not apply changes correctly")
                return False
        else:
            print("❌ Configuration hot-reload did not detect changes")
            return False
        
        return True
        
    finally:
        if os.path.exists(temp_config_file):
            os.unlink(temp_config_file)


def test_environment_variable_integration():
    """Test environment variable integration"""
    print("\n🧪 Testing environment variable integration...")
    
    # Set test environment variables
    test_env_vars = {
        'ENABLE_PER_CORE_AGENTS': 'true',
        'PER_CORE_MAX_AGENTS': '6',
        'PER_CORE_HEALTH_CHECK_INTERVAL': '90',
        'OPENROUTER_API_KEY': 'sk-or-test1234567890abcdefghijklmnopqrstuvwxyz123456789012345'
    }
    
    # Save original environment
    original_env = {}
    for key in test_env_vars:
        original_env[key] = os.environ.get(key)
    
    try:
        # Set test environment variables
        for key, value in test_env_vars.items():
            os.environ[key] = value
        
        # Create config manager (should pick up environment variables)
        temp_config_file = tempfile.mktemp(suffix='.json')  # Create temp filename but don't create file
        
        try:
            # Create config manager after setting environment variables
            config_manager = PerCoreAgentConfigManager(temp_config_file)
            config = config_manager.load_config(create_default=True)
            
            # Verify environment variables were applied
            if (config.enabled and 
                config.max_agents == 6 and 
                config.health_check_interval == 90 and
                config.openrouter_api_key == test_env_vars['OPENROUTER_API_KEY']):
                print("✅ Environment variables integrated correctly")
            else:
                print(f"❌ Environment variables not integrated correctly")
                return False
            
            return True
            
        finally:
            if os.path.exists(temp_config_file):
                os.unlink(temp_config_file)
        
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    """Run all configuration management tests"""
    print("🚀 Starting per-core agent configuration management tests...\n")
    
    tests = [
        test_basic_configuration,
        test_openrouter_api_key_validation,
        test_role_configuration,
        test_configuration_updates,
        test_hot_reload,
        test_environment_variable_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed\n")
            else:
                failed += 1
                print("❌ Test failed\n")
        except Exception as e:
            failed += 1
            print(f"❌ Test failed with exception: {e}\n")
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All configuration management tests passed!")
        return 0
    else:
        print("💥 Some configuration management tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())