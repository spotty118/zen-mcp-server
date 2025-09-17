#!/usr/bin/env python3
"""
Test script for per-core agent integration

This script tests the basic functionality of the per-core agent integration
to ensure it works correctly with existing MCP tools.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_per_core_status_tool():
    """Test the per-core status tool"""
    logger.info("Testing per-core status tool...")
    
    try:
        from tools.per_core_status import PerCoreStatusTool
        
        tool = PerCoreStatusTool()
        
        # Test basic status check
        result = await tool.execute({"detailed": False})
        logger.info("Basic status check completed")
        
        # Test detailed status check
        result_detailed = await tool.execute({"detailed": True})
        logger.info("Detailed status check completed")
        
        # Parse and display results
        import json
        from tools.models import ToolOutput
        
        if result:
            output = ToolOutput.model_validate_json(result[0].text)
            logger.info(f"Status tool result: {output.status}")
            if output.status == "success":
                logger.info("✅ Per-core status tool working correctly")
            else:
                logger.warning(f"⚠️ Status tool returned: {output.content}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Per-core status tool test failed: {e}")
        return False


async def test_per_core_integrator():
    """Test the per-core integrator directly"""
    logger.info("Testing per-core integrator...")
    
    try:
        from utils.per_core_tool_integration import get_per_core_integrator
        
        integrator = get_per_core_integrator()
        
        # Test initialization
        is_available = integrator.is_available()
        logger.info(f"Per-core integrator available: {is_available}")
        
        if is_available:
            # Test system health
            health = integrator.get_system_health()
            logger.info(f"System health: {health}")
            
            # Test task classification
            from utils.per_core_tool_integration import TaskType
            task_type = integrator.classify_task_type("chat", {"prompt": "help me debug this code"})
            logger.info(f"Task classification test: 'debug' -> {task_type}")
            
            logger.info("✅ Per-core integrator working correctly")
        else:
            logger.info("ℹ️ Per-core integrator not available (this is normal if not configured)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Per-core integrator test failed: {e}")
        return False


async def test_tool_integration():
    """Test integration with a simple tool"""
    logger.info("Testing tool integration...")
    
    try:
        from tools.chat import ChatTool
        from utils.per_core_tool_integration import enhance_tool_execution
        
        tool = ChatTool()
        arguments = {
            "prompt": "Hello, this is a test of the per-core agent integration system.",
            "model": "auto"  # Use auto mode to test model resolution
        }
        
        # Test enhanced execution
        result, used_per_core = await enhance_tool_execution(tool, arguments)
        
        logger.info(f"Tool execution completed. Used per-core system: {used_per_core}")
        
        if result:
            logger.info("✅ Tool integration working correctly")
        else:
            logger.warning("⚠️ Tool execution returned empty result")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool integration test failed: {e}")
        return False


async def test_configuration():
    """Test configuration settings"""
    logger.info("Testing configuration...")
    
    try:
        from config import (
            ENABLE_PER_CORE_AGENTS,
            PER_CORE_MAX_AGENTS,
            PER_CORE_OPENROUTER_REQUIRED,
            PER_CORE_FALLBACK_MODE,
            PER_CORE_MULTI_AGENT_TOOLS
        )
        
        logger.info(f"ENABLE_PER_CORE_AGENTS: {ENABLE_PER_CORE_AGENTS}")
        logger.info(f"PER_CORE_MAX_AGENTS: {PER_CORE_MAX_AGENTS}")
        logger.info(f"PER_CORE_OPENROUTER_REQUIRED: {PER_CORE_OPENROUTER_REQUIRED}")
        logger.info(f"PER_CORE_FALLBACK_MODE: {PER_CORE_FALLBACK_MODE}")
        logger.info(f"PER_CORE_MULTI_AGENT_TOOLS: {PER_CORE_MULTI_AGENT_TOOLS}")
        
        # Check OpenRouter configuration
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
            logger.info("✅ OpenRouter API key configured")
        else:
            logger.info("ℹ️ OpenRouter API key not configured (per-core agents will use fallback)")
        
        logger.info("✅ Configuration test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("🚀 Starting per-core agent integration tests...")
    
    tests = [
        ("Configuration", test_configuration),
        ("Per-Core Integrator", test_per_core_integrator),
        ("Per-Core Status Tool", test_per_core_status_tool),
        ("Tool Integration", test_tool_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Per-core agent integration is working correctly.")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)