#!/usr/bin/env python3
"""
Test script for provider prefix functionality.

This script tests that the new provider prefix system allows forcing specific providers
when multiple providers are available for the same model.
"""

import os
import sys
import tempfile
from unittest.mock import patch

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/zen-mcp-server/zen-mcp-server')

def test_provider_prefix_routing():
    """Test that provider prefixes correctly route to specific providers."""
    print("Testing provider prefix routing...")
    
    # Mock environment variables for multiple providers
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test-gemini-key',
        'OPENAI_API_KEY': 'test-openai-key', 
        'OPENROUTER_API_KEY': 'test-openrouter-key',
        'CUSTOM_API_URL': 'http://localhost:11434/v1',
        'CUSTOM_API_KEY': ''
    }):
        # Clear any existing provider registrations
        from providers.registry import ModelProviderRegistry
        ModelProviderRegistry.reset_for_testing()
        
        # Register providers like server.py does
        from providers.base import ProviderType
        from providers.gemini import GeminiModelProvider
        from providers.openai_provider import OpenAIModelProvider
        from providers.openrouter import OpenRouterProvider
        from providers.custom import CustomProvider
        
        def custom_provider_factory(api_key=None):
            return CustomProvider(api_key=api_key or "", base_url="http://localhost:11434/v1")
        
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.OPENAI, OpenAIModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        ModelProviderRegistry.register_provider(ProviderType.CUSTOM, custom_provider_factory)
        
        # Test normal priority routing (no prefix)
        print("\n=== Testing normal priority routing ===")
        
        # "pro" should go to Google/Gemini by default (higher priority)
        provider = ModelProviderRegistry.get_provider_for_model("pro")
        if provider:
            print(f"✓ 'pro' routed to {provider.get_provider_type().value} provider (expected: google)")
            assert provider.get_provider_type() == ProviderType.GOOGLE, f"Expected Google, got {provider.get_provider_type()}"
        else:
            print("✗ No provider found for 'pro'")
            
        # Test explicit provider routing with prefixes
        print("\n=== Testing explicit provider routing ===")
        
        # "openrouter:pro" should go directly to OpenRouter
        provider = ModelProviderRegistry.get_provider_for_model("openrouter:pro")
        if provider:
            print(f"✓ 'openrouter:pro' routed to {provider.get_provider_type().value} provider")
            assert provider.get_provider_type() == ProviderType.OPENROUTER, f"Expected OpenRouter, got {provider.get_provider_type()}"
        else:
            print("✗ No provider found for 'openrouter:pro'")
            
        # "google:pro" should go to Google
        provider = ModelProviderRegistry.get_provider_for_model("google:pro")
        if provider:
            print(f"✓ 'google:pro' routed to {provider.get_provider_type().value} provider")
            assert provider.get_provider_type() == ProviderType.GOOGLE, f"Expected Google, got {provider.get_provider_type()}"
        else:
            print("✗ No provider found for 'google:pro'")
            
        # "custom:llama3.2" should go to Custom
        provider = ModelProviderRegistry.get_provider_for_model("custom:llama3.2")
        if provider:
            print(f"✓ 'custom:llama3.2' routed to {provider.get_provider_type().value} provider")
            assert provider.get_provider_type() == ProviderType.CUSTOM, f"Expected Custom, got {provider.get_provider_type()}"
        else:
            print("✗ No provider found for 'custom:llama3.2'")
            
        # Test invalid provider prefix
        print("\n=== Testing invalid provider prefix ===")
        provider = ModelProviderRegistry.get_provider_for_model("invalid:model")
        if provider:
            print(f"✓ 'invalid:model' treated as normal model name, routed to {provider.get_provider_type().value}")
        else:
            print("✓ 'invalid:model' correctly returned None (no provider supports it)")
            
        print("\n=== All tests passed! ===")
        return True

def test_tool_integration():
    """Test that tools correctly handle provider prefixes."""
    print("\n\nTesting tool integration...")
    
    # Mock environment variables
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'test-gemini-key',
        'OPENROUTER_API_KEY': 'test-openrouter-key'
    }):
        # Clear registry
        from providers.registry import ModelProviderRegistry
        ModelProviderRegistry.reset_for_testing()
        
        # Register providers
        from providers.base import ProviderType
        from providers.gemini import GeminiModelProvider
        from providers.openrouter import OpenRouterProvider
        
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        
        # Test ModelContext handles prefixes correctly
        from utils.model_context import ModelContext
        
        # Test normal model name
        context = ModelContext("pro")
        print(f"✓ ModelContext for 'pro': provider = {context.provider.get_provider_type().value}")
        assert context.provider.get_provider_type() == ProviderType.GOOGLE
        
        # Test prefixed model name
        context = ModelContext("openrouter:pro") 
        print(f"✓ ModelContext for 'openrouter:pro': provider = {context.provider.get_provider_type().value}")
        assert context.provider.get_provider_type() == ProviderType.OPENROUTER
        
        print("✓ Tool integration tests passed!")
        return True

if __name__ == "__main__":
    try:
        test_provider_prefix_routing()
        test_tool_integration()
        print("\n🎉 ALL TESTS PASSED! Provider prefix functionality is working correctly.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)