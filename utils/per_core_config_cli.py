#!/usr/bin/env python3
"""
Per-Core Agent Configuration CLI

Command-line utility for managing per-core agent configurations,
including API key setup, validation, and configuration management.

Usage:
    python -m utils.per_core_config_cli --help
    python -m utils.per_core_config_cli validate
    python -m utils.per_core_config_cli setup --api-key sk-or-...
    python -m utils.per_core_config_cli show
    python -m utils.per_core_config_cli test-connection
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# Add the parent directory to the path so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.per_core_agent_config import (
    PerCoreAgentConfigManager,
    PerCoreAgentSystemConfig,
    OpenRouterAgentConfig
)
from utils.per_core_config_validator import get_per_core_config_validator
from utils.agent_core import AgentRole


def setup_logging(verbose: bool = False) -> None:
    """Setup logging for the CLI"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def validate_config(config_file: Optional[str] = None, verbose: bool = False) -> int:
    """
    Validate per-core agent configuration
    
    Args:
        config_file: Path to configuration file (optional)
        verbose: Enable verbose output
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config_manager = PerCoreAgentConfigManager(config_file)
        config = config_manager.load_config(create_default=False)
        
        # Validate configuration
        errors = config.validate()
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  • {error}")
            return 1
        
        print("✅ Configuration validation passed")
        
        # Validate environment variables
        validator = get_per_core_config_validator()
        env_valid, env_warnings = validator.validate_environment_variables()
        
        if env_warnings:
            print("\n⚠️  Environment variable warnings:")
            for warning in env_warnings:
                print(f"  • {warning}")
        
        if verbose:
            print(f"\nConfiguration details:")
            print(f"  • Enabled: {config.enabled}")
            print(f"  • Max agents: {config.max_agents or 'auto-detect'}")
            print(f"  • OpenRouter required: {config.openrouter_required}")
            print(f"  • Fallback mode: {config.fallback_mode}")
            print(f"  • Health check interval: {config.health_check_interval}s")
            print(f"  • Agent timeout: {config.agent_timeout}s")
            print(f"  • Role configurations: {len(config.role_configs)}")
            print(f"  • Multi-agent tools: {len(config.multi_agent_tools)}")
        
        return 0
        
    except FileNotFoundError:
        print("❌ Configuration file not found")
        print("   Run 'setup' command to create a default configuration")
        return 1
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def setup_config(api_key: str, config_file: Optional[str] = None, 
                max_agents: Optional[int] = None, verbose: bool = False) -> int:
    """
    Setup per-core agent configuration with OpenRouter API key
    
    Args:
        api_key: OpenRouter API key
        config_file: Path to configuration file (optional)
        max_agents: Maximum number of agents (optional)
        verbose: Enable verbose output
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Validate API key first
        validator = get_per_core_config_validator()
        is_valid, errors = validator.validate_openrouter_api_key(api_key)
        
        if not is_valid:
            print("❌ Invalid OpenRouter API key:")
            for error in errors:
                print(f"  • {error}")
            return 1
        
        # Create configuration manager
        config_manager = PerCoreAgentConfigManager(config_file)
        
        # Load existing config or create default
        try:
            config = config_manager.load_config(create_default=False)
            print("📝 Updating existing configuration...")
        except FileNotFoundError:
            config = config_manager.load_config(create_default=True)
            print("📝 Creating new configuration...")
        
        # Update configuration with provided values
        config.openrouter_api_key = api_key
        if max_agents is not None:
            config.max_agents = max_agents
        
        # Create role-specific configurations
        config.role_configs = config_manager._create_default_role_configs(api_key)
        
        # Save configuration
        if config_manager.save_config(config):
            print("✅ Configuration saved successfully")
            
            if verbose:
                print(f"\nConfiguration saved to: {config_manager.config_file}")
                print(f"Role configurations created for {len(config.role_configs)} roles")
            
            return 0
        else:
            print("❌ Failed to save configuration")
            return 1
            
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1


def show_config(config_file: Optional[str] = None, format_json: bool = False) -> int:
    """
    Show current per-core agent configuration
    
    Args:
        config_file: Path to configuration file (optional)
        format_json: Output in JSON format
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config_manager = PerCoreAgentConfigManager(config_file)
        config = config_manager.load_config(create_default=False)
        
        if format_json:
            # Output as JSON (mask API key)
            config_dict = config.to_dict()
            if 'openrouter_api_key' in config_dict and config_dict['openrouter_api_key']:
                config_dict['openrouter_api_key'] = config_dict['openrouter_api_key'][:10] + "..."
            
            # Mask API keys in role configs
            for role_name, role_config in config_dict.get('role_configs', {}).items():
                if 'api_key' in role_config and role_config['api_key']:
                    role_config['api_key'] = role_config['api_key'][:10] + "..."
            
            print(json.dumps(config_dict, indent=2))
        else:
            # Human-readable output
            print("📋 Per-Core Agent Configuration")
            print("=" * 40)
            print(f"Enabled: {config.enabled}")
            print(f"Max agents: {config.max_agents or 'auto-detect'}")
            print(f"OpenRouter required: {config.openrouter_required}")
            print(f"Fallback mode: {config.fallback_mode}")
            print(f"Health check interval: {config.health_check_interval}s")
            print(f"Agent timeout: {config.agent_timeout}s")
            
            # Show API key status (masked)
            if config.openrouter_api_key:
                masked_key = config.openrouter_api_key[:10] + "..." + config.openrouter_api_key[-4:]
                print(f"OpenRouter API key: {masked_key}")
            else:
                print("OpenRouter API key: Not configured")
            
            # Show role configurations
            print(f"\nRole Configurations ({len(config.role_configs)}):")
            for role_name, role_config in config.role_configs.items():
                print(f"  • {role_name}:")
                print(f"    - Models: {', '.join(role_config.preferred_models[:3])}{'...' if len(role_config.preferred_models) > 3 else ''}")
                print(f"    - Rate limit: {role_config.rate_limit_per_minute}/min")
                print(f"    - Concurrent: {role_config.max_concurrent_calls}")
                print(f"    - Thinking mode: {role_config.thinking_mode_default}")
            
            # Show multi-agent tools
            print(f"\nMulti-agent tools ({len(config.multi_agent_tools)}):")
            print(f"  {', '.join(sorted(config.multi_agent_tools))}")
        
        return 0
        
    except FileNotFoundError:
        print("❌ Configuration file not found")
        print("   Run 'setup' command to create a configuration")
        return 1
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1


def test_connection(config_file: Optional[str] = None, verbose: bool = False) -> int:
    """
    Test OpenRouter API connection
    
    Args:
        config_file: Path to configuration file (optional)
        verbose: Enable verbose output
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        config_manager = PerCoreAgentConfigManager(config_file)
        config = config_manager.load_config(create_default=False)
        
        if not config.openrouter_api_key:
            print("❌ No OpenRouter API key configured")
            print("   Run 'setup' command to configure API key")
            return 1
        
        print("🔍 Testing OpenRouter API connection...")
        
        # Test API key validation
        validator = get_per_core_config_validator()
        is_valid, errors = validator.validate_openrouter_api_key(config.openrouter_api_key)
        
        if not is_valid:
            print("❌ API key validation failed:")
            for error in errors:
                print(f"  • {error}")
            return 1
        
        print("✅ API key format validation passed")
        
        # Test provider availability
        try:
            from providers.registry import ModelProviderRegistry
            from providers.base import ProviderType
            
            registry = ModelProviderRegistry()
            openrouter_provider = registry.get_provider(ProviderType.OPENROUTER)
            
            if openrouter_provider:
                print("✅ OpenRouter provider is available")
                
                # Test model availability
                available_models = registry.get_available_model_names(ProviderType.OPENROUTER)
                print(f"✅ Found {len(available_models)} available models")
                
                if verbose:
                    print("Available models:")
                    for model in sorted(available_models)[:10]:  # Show first 10
                        print(f"  • {model}")
                    if len(available_models) > 10:
                        print(f"  ... and {len(available_models) - 10} more")
                
            else:
                print("⚠️  OpenRouter provider not available in registry")
                
        except Exception as e:
            print(f"⚠️  Could not test provider availability: {e}")
        
        # Test role configurations
        valid_roles = 0
        for role_name, role_config in config.role_configs.items():
            role_errors = role_config.validate()
            if not role_errors:
                valid_roles += 1
            elif verbose:
                print(f"⚠️  Role {role_name} has configuration issues:")
                for error in role_errors:
                    print(f"    • {error}")
        
        print(f"✅ {valid_roles}/{len(config.role_configs)} role configurations are valid")
        
        return 0
        
    except FileNotFoundError:
        print("❌ Configuration file not found")
        return 1
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Per-Core Agent Configuration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate                           # Validate current configuration
  %(prog)s setup --api-key sk-or-...         # Setup with OpenRouter API key
  %(prog)s show                              # Show current configuration
  %(prog)s show --json                       # Show configuration as JSON
  %(prog)s test-connection                   # Test OpenRouter connection
        """
    )
    
    parser.add_argument(
        '--config-file', '-c',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup configuration')
    setup_parser.add_argument(
        '--api-key', '-k',
        required=True,
        help='OpenRouter API key'
    )
    setup_parser.add_argument(
        '--max-agents', '-m',
        type=int,
        help='Maximum number of agents'
    )
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration')
    show_parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output in JSON format'
    )
    
    # Test connection command
    test_parser = subparsers.add_parser('test-connection', help='Test OpenRouter connection')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'validate':
        return validate_config(args.config_file, args.verbose)
    elif args.command == 'setup':
        return setup_config(args.api_key, args.config_file, args.max_agents, args.verbose)
    elif args.command == 'show':
        return show_config(args.config_file, args.json)
    elif args.command == 'test-connection':
        return test_connection(args.config_file, args.verbose)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())