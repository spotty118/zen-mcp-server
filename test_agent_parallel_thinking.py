#!/usr/bin/env python3
"""
Test Agent-Enhanced Parallel Thinking

Test script to validate that the enhanced parallel thinking tool works
correctly with agent communication and collaboration features.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.parallelthink import ParallelThinkTool, ParallelThinkRequest


async def test_agent_parallel_thinking():
    """Test parallel thinking with agent communication"""
    print("🧠 Testing Agent-Enhanced Parallel Thinking")
    print("=" * 50)
    
    # Create the parallel thinking tool
    tool = ParallelThinkTool()
    
    # Test request with agent mode enabled
    test_request = {
        "prompt": "Analyze the security implications of storing user passwords in plain text",
        "thinking_paths": 3,
        "enable_agent_mode": True,
        "agent_roles": ["security_analyst", "architecture_reviewer", "code_quality_inspector"],
        "enable_agent_communication": True,
        "cpu_cores": 3,
        "time_limit": 30
    }
    
    print("🔧 Test Configuration:")
    print(f"  - Thinking paths: {test_request['thinking_paths']}")
    print(f"  - Agent mode: {test_request['enable_agent_mode']}")
    print(f"  - Agent roles: {test_request['agent_roles']}")
    print(f"  - Agent communication: {test_request['enable_agent_communication']}")
    print(f"  - CPU cores: {test_request['cpu_cores']}")
    
    print("\n🚀 Executing agent-enhanced parallel thinking...")
    
    try:
        # Execute the parallel thinking with agents
        result = await tool.execute(test_request)
        
        if result and len(result) > 0:
            response = result[0]
            
            print("\n✅ Execution completed successfully!")
            
            # Check for agent-specific data in response
            if "metadata" in response:
                metadata = response["metadata"]
                print(f"\n📊 Execution Metadata:")
                print(f"  - Paths completed: {metadata.get('paths_completed', 'N/A')}")
                print(f"  - Total execution time: {metadata.get('total_execution_time', 'N/A'):.2f}s")
                print(f"  - CPU cores used: {metadata.get('cpu_cores_used', 'N/A')}")
                print(f"  - Execution strategy: {metadata.get('execution_strategy', 'N/A')}")
                print(f"  - Core context enabled: {metadata.get('core_context_enabled', 'N/A')}")
                print(f"  - Insights sharing enabled: {metadata.get('insights_sharing_enabled', 'N/A')}")
            
            # Check for synthesis
            if "synthesis" in response:
                synthesis = response["synthesis"]
                print(f"\n🔍 Synthesis Preview:")
                print(f"  - Length: {len(synthesis)} characters")
                print(f"  - First 100 chars: {synthesis[:100]}...")
            
            # Check for individual paths (agent results)
            if "individual_paths" in response:
                paths = response["individual_paths"]
                print(f"\n🤖 Agent Results:")
                print(f"  - Number of agent paths: {len(paths)}")
                
                for i, path in enumerate(paths):
                    approach = path.get("approach", "Unknown")
                    success = path.get("success", False)
                    exec_time = path.get("execution_time", 0)
                    core_id = path.get("cpu_core", "N/A")
                    
                    print(f"  - Agent {i+1}: {approach}")
                    print(f"    • Success: {success}")
                    print(f"    • Execution time: {exec_time:.2f}s")
                    print(f"    • CPU core: {core_id}")
                    
                    if "result" in path:
                        result_preview = path["result"][:100] if path["result"] else "No result"
                        print(f"    • Result preview: {result_preview}...")
            
            print("\n🎉 Agent-enhanced parallel thinking test completed successfully!")
            return True
            
        else:
            print("\n❌ No result returned from parallel thinking execution")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_parallel_thinking():
    """Test basic parallel thinking without agents for comparison"""
    print("\n🔄 Testing Basic Parallel Thinking (no agents)")
    print("=" * 50)
    
    tool = ParallelThinkTool()
    
    # Test request without agent mode
    test_request = {
        "prompt": "Analyze the security implications of storing user passwords in plain text",
        "thinking_paths": 3,
        "enable_agent_mode": False,  # Disable agent mode
        "cpu_cores": 3,
        "time_limit": 30
    }
    
    print("🔧 Test Configuration:")
    print(f"  - Thinking paths: {test_request['thinking_paths']}")
    print(f"  - Agent mode: {test_request['enable_agent_mode']}")
    print(f"  - CPU cores: {test_request['cpu_cores']}")
    
    try:
        result = await tool.execute(test_request)
        
        if result and len(result) > 0:
            print("\n✅ Basic parallel thinking completed successfully!")
            
            response = result[0]
            if "individual_paths" in response:
                paths = response["individual_paths"]
                print(f"  - Number of paths: {len(paths)}")
                for i, path in enumerate(paths):
                    approach = path.get("approach", "Unknown")
                    success = path.get("success", False)
                    print(f"  - Path {i+1}: {approach} (Success: {success})")
            
            return True
        else:
            print("\n❌ No result returned from basic parallel thinking")
            return False
            
    except Exception as e:
        print(f"\n💥 Basic test failed with error: {e}")
        return False


if __name__ == "__main__":
    async def run_all_tests():
        print("🧪 Running Nexus Agent-Enhanced Parallel Thinking Tests")
        print("=" * 60)
        
        # Test 1: Agent-enhanced parallel thinking
        agent_test_result = await test_agent_parallel_thinking()
        
        # Test 2: Basic parallel thinking for comparison
        basic_test_result = await test_basic_parallel_thinking()
        
        print("\n" + "=" * 60)
        print("📋 Test Results Summary:")
        print(f"  - Agent-enhanced test: {'✅ PASSED' if agent_test_result else '❌ FAILED'}")
        print(f"  - Basic parallel test: {'✅ PASSED' if basic_test_result else '❌ FAILED'}")
        
        if agent_test_result and basic_test_result:
            print("\n🎉 All tests passed! Nexus agent system is working correctly.")
            return True
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            return False
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)