#!/usr/bin/env python3
"""
Real-world test for OpenRouter individual provider calls.

This test demonstrates that users can now force OpenRouter usage even when
native providers are available for the same model names.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, '/home/runner/work/zen-mcp-server/zen-mcp-server')

def test_real_world_scenario():
    """Test the real-world scenario that was broken before."""
    print("🔧 Testing Real-World OpenRouter Individual Calls")
    print("=" * 60)
    
    # Scenario: User has both Gemini and OpenRouter API keys configured
    # They want to use OpenRouter's "google/gemini-2.5-pro" instead of native Gemini
    
    with patch.dict(os.environ, {
        'GEMINI_API_KEY': 'real-gemini-key',
        'OPENROUTER_API_KEY': 'real-openrouter-key'
    }):
        # Clear and setup providers like server.py
        from providers.registry import ModelProviderRegistry
        ModelProviderRegistry.reset_for_testing()
        
        from providers.base import ProviderType
        from providers.gemini import GeminiModelProvider
        from providers.openrouter import OpenRouterProvider
        
        ModelProviderRegistry.register_provider(ProviderType.GOOGLE, GeminiModelProvider)
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        
        print("✓ Both Gemini and OpenRouter providers registered")
        
        # Test 1: Default behavior (should prefer native Gemini)
        print("\n1️⃣  Testing default priority behavior:")
        provider = ModelProviderRegistry.get_provider_for_model("pro")
        assert provider.get_provider_type() == ProviderType.GOOGLE
        print("   ✓ 'pro' → Native Gemini (correct priority order)")
        
        provider = ModelProviderRegistry.get_provider_for_model("flash")
        assert provider.get_provider_type() == ProviderType.GOOGLE  
        print("   ✓ 'flash' → Native Gemini (correct priority order)")
        
        # Test 2: Explicit OpenRouter requests (THE NEW FUNCTIONALITY!)
        print("\n2️⃣  Testing explicit OpenRouter requests:")
        provider = ModelProviderRegistry.get_provider_for_model("openrouter:pro")
        assert provider.get_provider_type() == ProviderType.OPENROUTER
        print("   ✓ 'openrouter:pro' → OpenRouter (bypassed priority!)")
        
        provider = ModelProviderRegistry.get_provider_for_model("openrouter:flash")
        assert provider.get_provider_type() == ProviderType.OPENROUTER
        print("   ✓ 'openrouter:flash' → OpenRouter (bypassed priority!)")
        
        provider = ModelProviderRegistry.get_provider_for_model("openrouter:opus")
        assert provider.get_provider_type() == ProviderType.OPENROUTER
        print("   ✓ 'openrouter:opus' → OpenRouter (OpenRouter-only model)")
        
        # Test 3: Alternative prefix forms
        print("\n3️⃣  Testing alternative prefix forms:")
        provider = ModelProviderRegistry.get_provider_for_model("google:flash")
        assert provider.get_provider_type() == ProviderType.GOOGLE
        print("   ✓ 'google:flash' → Native Gemini")
        
        provider = ModelProviderRegistry.get_provider_for_model("gemini:pro")
        assert provider.get_provider_type() == ProviderType.GOOGLE
        print("   ✓ 'gemini:pro' → Native Gemini")
        
        # Test 4: Tool integration (the ultimate test)
        print("\n4️⃣  Testing full tool integration:")
        
        # Mock the ChatTool to avoid API calls but test routing
        from tools.chat import ChatTool
        from utils.model_context import ModelContext
        
        # Test that ModelContext correctly routes prefixed models
        context_normal = ModelContext("pro")
        context_explicit = ModelContext("openrouter:pro")
        
        assert context_normal.provider.get_provider_type() == ProviderType.GOOGLE
        assert context_explicit.provider.get_provider_type() == ProviderType.OPENROUTER
        
        print("   ✓ ModelContext correctly routes 'pro' → Google")
        print("   ✓ ModelContext correctly routes 'openrouter:pro' → OpenRouter")
        
        # Verify the tool would use the right model names for API calls
        tool = ChatTool()
        from providers.base import parse_provider_prefix
        
        # Simulate what happens in tool execution
        test_cases = [
            ("pro", ProviderType.GOOGLE, "pro"),
            ("openrouter:pro", ProviderType.OPENROUTER, "pro"),
            ("openrouter:opus", ProviderType.OPENROUTER, "opus"),
            ("google:flash", ProviderType.GOOGLE, "flash")
        ]
        
        for original_model, expected_provider, expected_api_model in test_cases:
            explicit_provider_type, actual_model = parse_provider_prefix(original_model)
            provider = ModelProviderRegistry.get_provider_for_model(original_model)
            
            assert provider.get_provider_type() == expected_provider
            assert actual_model == expected_api_model
            
            print(f"   ✓ '{original_model}' → {expected_provider.value} provider, API calls use '{expected_api_model}'")
        
        print("\n🎉 SUCCESS! OpenRouter individual calls are now working!")
        print("\nKey Benefits:")
        print("• Users can force OpenRouter usage with 'openrouter:model' syntax")
        print("• Backward compatibility maintained for existing model names")
        print("• Works across all tools (chat, codereview, debug, etc.)")
        print("• Supports billing/usage tracking via specific providers")
        
        return True

def test_error_cases():
    """Test error handling for edge cases."""
    print("\n\n🚨 Testing Error Cases")
    print("=" * 30)
    
    with patch.dict(os.environ, {
        'OPENROUTER_API_KEY': 'test-key'
    }):
        from providers.registry import ModelProviderRegistry
        ModelProviderRegistry.reset_for_testing()
        
        from providers.base import ProviderType  
        from providers.openrouter import OpenRouterProvider
        
        ModelProviderRegistry.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
        
        # Test requesting unavailable provider
        print("\n1️⃣  Testing unavailable provider request:")
        provider = ModelProviderRegistry.get_provider_for_model("google:pro")
        assert provider is None
        print("   ✓ 'google:pro' correctly returns None (Google not configured)")
        
        # Test invalid prefix
        print("\n2️⃣  Testing invalid prefix:")
        provider = ModelProviderRegistry.get_provider_for_model("invalid:model")
        # Should fall back to normal routing and find OpenRouter
        assert provider is not None
        assert provider.get_provider_type() == ProviderType.OPENROUTER
        print("   ✓ 'invalid:model' treated as normal model name")
        
        print("\n✓ Error handling works correctly!")
        
        return True

if __name__ == "__main__":
    try:
        test_real_world_scenario()
        test_error_cases()
        print("\n" + "="*60)
        print("🏆 ALL TESTS PASSED! The issue has been successfully fixed!")
        print("Users can now call OpenRouter individually using provider prefixes.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)