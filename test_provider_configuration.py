#!/usr/bin/env python3
"""
Test script to verify that the parallel thinking tool works correctly
when providers ARE configured (but may still have API failures).

This complements the provider fix test by showing the tool still works
in normal circumstances.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from server import configure_providers
from tools.parallelthink import ParallelThinkTool
from providers.registry import ModelProviderRegistry


async def test_with_configured_providers():
    """Test parallel thinking with providers configured"""
    print("🧠 Testing Parallel Thinking Tool - With Providers")
    print("=" * 50)
    
    # Configure providers first
    try:
        configure_providers()
        registry = ModelProviderRegistry()
        available_providers = registry.get_available_providers()
        available_models = registry.get_available_models()
        
        print(f"📡 Provider Configuration:")
        print(f"  - Available providers: {[p.value for p in available_providers]}")
        print(f"  - Available models: {len(available_models)}")
        
        if not available_providers:
            print("⚠️  No providers configured - this will test the fallback scenario")
        
    except Exception as e:
        print(f"⚠️  Provider configuration failed: {e}")
        print("   This will test the fallback scenario")
    
    # Create the parallel thinking tool
    tool = ParallelThinkTool()
    
    # Test request with realistic parameters
    test_request = {
        "prompt": "What are the key considerations for designing a scalable web API?",
        "thinking_paths": 2,
        "enable_agent_mode": True,
        "cpu_cores": 2,
        "time_limit": 10,
        "synthesis_style": "comprehensive"
    }
    
    print(f"\n🔧 Test Configuration:")
    print(f"  - Thinking paths: {test_request['thinking_paths']}")
    print(f"  - Agent mode: {test_request['enable_agent_mode']}")
    print(f"  - CPU cores: {test_request['cpu_cores']}")
    print(f"  - Time limit: {test_request['time_limit']}s")
    print(f"  - Synthesis style: {test_request['synthesis_style']}")
    
    print("\n🚀 Executing parallel thinking...")
    
    try:
        # Execute the parallel thinking
        result = await tool.execute(test_request)
        
        print("\n✅ SUCCESS: Tool executed without crashing!")
        
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if isinstance(first_result, dict):
                # Check execution summary
                if 'execution_summary' in first_result:
                    summary = first_result['execution_summary']
                    print(f"\n📊 Execution Summary:")
                    print(f"  - Total paths: {summary.get('total_paths', 'unknown')}")
                    print(f"  - Successful paths: {summary.get('successful_paths', 'unknown')}")
                    print(f"  - Models used: {summary.get('models_used', 'unknown')}")
                    print(f"  - Execution time: {summary.get('total_execution_time', 0):.3f}s")
                    print(f"  - CPU cores used: {summary.get('cpu_cores_used', 'unknown')}")
                
                # Check individual paths
                if 'individual_paths' in first_result:
                    paths = first_result['individual_paths']
                    successful_paths = [p for p in paths if p.get('success', False)]
                    failed_paths = [p for p in paths if not p.get('success', False)]
                    
                    print(f"\n🔍 Path Results:")
                    print(f"  - Successful: {len(successful_paths)}")
                    print(f"  - Failed: {len(failed_paths)}")
                    
                    for i, path in enumerate(paths[:2]):  # Show first 2 paths
                        success = path.get('success', False)
                        error = path.get('error', 'none')
                        approach = path.get('approach', 'unknown')
                        print(f"  Path {i+1}:")
                        print(f"    • Approach: {approach}")
                        print(f"    • Success: {success}")
                        if not success:
                            print(f"    • Error: {error}")
                
                # Check final analysis (only show preview to avoid spam)
                if 'parallel_thinking_analysis' in first_result:
                    analysis = first_result['parallel_thinking_analysis']
                    if analysis and len(analysis) > 100:
                        print(f"\n💭 Analysis Preview:")
                        print(f"  {analysis[:200]}...")
                    else:
                        print(f"\n💭 Analysis:")
                        print(f"  {analysis}")
        
        print("\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Tool crashed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_with_configured_providers())
    sys.exit(0 if success else 1)