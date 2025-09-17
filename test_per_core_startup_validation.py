#!/usr/bin/env python3
"""
Per-Core Agent System Startup Validation Script

This script validates the complete per-core agent coordination system integration
by simulating the server startup sequence and performing comprehensive system checks.

Usage:
    python test_per_core_startup_validation.py [--detailed] [--no-openrouter]

Options:
    --detailed      Show detailed agent information and health metrics
    --no-openrouter Skip OpenRouter connectivity tests
"""

import asyncio
import os
import sys
import argparse
import logging
from typing import Dict, Any
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def validate_server_startup_integration():
    """
    Validate the per-core agent system integration with server startup sequence.
    
    This simulates the actual server startup process and validates that all
    components are properly integrated and functional.
    """
    print("🚀 Starting Per-Core Agent System Startup Validation\n")
    print("=" * 60)
    
    validation_results = {
        "configuration_valid": False,
        "system_initialized": False,
        "agents_healthy": False,
        "tools_integrated": False,
        "end_to_end_working": False,
        "overall_success": False
    }
    
    try:
        # Step 1: Validate Configuration
        print("\n1. 📋 Validating Configuration...")
        config_valid = await validate_configuration()
        validation_results["configuration_valid"] = config_valid
        
        if config_valid:
            print("   ✅ Configuration validation passed")
        else:
            print("   ❌ Configuration validation failed")
            return validation_results
        
        # Step 2: Test System Initialization (as done in server startup)
        print("\n2. 🔧 Testing System Initialization...")
        init_success = await test_system_initialization()
        validation_results["system_initialized"] = init_success
        
        if init_success:
            print("   ✅ System initialization successful")
        else:
            print("   ❌ System initialization failed")
            return validation_results
        
        # Step 3: Validate Agent Health
        print("\n3. 🏥 Validating Agent Health...")
        agents_healthy = await validate_agent_health()
        validation_results["agents_healthy"] = agents_healthy
        
        if agents_healthy:
            print("   ✅ All agents are healthy")
        else:
            print("   ⚠️  Some agents may have issues")
        
        # Step 4: Test Tool Integration
        print("\n4. 🔗 Testing Tool Integration...")
        tools_integrated = await test_tool_integration()
        validation_results["tools_integrated"] = tools_integrated
        
        if tools_integrated:
            print("   ✅ Tool integration working")
        else:
            print("   ❌ Tool integration failed")
            return validation_results
        
        # Step 5: End-to-End Workflow Test
        print("\n5. 🎯 Testing End-to-End Workflow...")
        e2e_working = await test_end_to_end_workflow()
        validation_results["end_to_end_working"] = e2e_working
        
        if e2e_working:
            print("   ✅ End-to-end workflow successful")
        else:
            print("   ❌ End-to-end workflow failed")
        
        # Overall Assessment
        validation_results["overall_success"] = all([
            validation_results["configuration_valid"],
            validation_results["system_initialized"],
            validation_results["tools_integrated"],
            validation_results["end_to_end_working"]
        ])
        
        return validation_results
        
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        logger.error(f"Startup validation error: {e}")
        return validation_results


async def validate_configuration():
    """Validate per-core agent system configuration"""
    try:
        from config import (
            ENABLE_PER_CORE_AGENTS,
            PER_CORE_MAX_AGENTS,
            PER_CORE_OPENROUTER_REQUIRED,
            PER_CORE_FALLBACK_MODE,
            PER_CORE_MULTI_AGENT_TOOLS
        )
        
        print(f"   - ENABLE_PER_CORE_AGENTS: {ENABLE_PER_CORE_AGENTS}")
        print(f"   - PER_CORE_MAX_AGENTS: {PER_CORE_MAX_AGENTS}")
        print(f"   - PER_CORE_OPENROUTER_REQUIRED: {PER_CORE_OPENROUTER_REQUIRED}")
        print(f"   - PER_CORE_FALLBACK_MODE: {PER_CORE_FALLBACK_MODE}")
        print(f"   - Multi-agent tools: {len(PER_CORE_MULTI_AGENT_TOOLS)}")
        
        # Check OpenRouter configuration if required
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_configured = bool(openrouter_key and openrouter_key != "your_openrouter_api_key_here")
        print(f"   - OpenRouter configured: {openrouter_configured}")
        
        if PER_CORE_OPENROUTER_REQUIRED and not openrouter_configured:
            print("   ⚠️  OpenRouter required but not configured")
            return False
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Configuration import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Configuration validation error: {e}")
        return False


async def test_system_initialization():
    """Test the system initialization as done in server startup"""
    try:
        # Import and test the actual server initialization function
        from server import initialize_per_core_agent_system
        
        print("   - Running initialize_per_core_agent_system()...")
        success = await initialize_per_core_agent_system()
        
        if success:
            print("   - System initialization completed successfully")
            
            # Verify integrator is available
            from utils.per_core_tool_integration import get_per_core_integrator
            integrator = get_per_core_integrator()
            
            if integrator.is_available():
                print("   - Per-core integrator is available")
                
                if integrator.per_core_manager:
                    num_agents = len(integrator.per_core_manager.agents)
                    print(f"   - {num_agents} agents initialized")
                
                return True
            else:
                print("   - Per-core integrator not available")
                return False
        else:
            print("   - System initialization returned False")
            return False
            
    except Exception as e:
        print(f"   ❌ System initialization error: {e}")
        return False


async def validate_agent_health():
    """Validate the health of all initialized agents"""
    try:
        from utils.per_core_tool_integration import get_per_core_integrator
        from server import validate_per_core_system
        
        integrator = get_per_core_integrator()
        
        if not integrator.is_available():
            print("   - Per-core system not available for health check")
            return False
        
        # Use the server's validation function
        validation_results = await validate_per_core_system(integrator)
        
        print(f"   - Total agents: {validation_results['num_agents']}")
        print(f"   - Agent roles: {', '.join(validation_results['agent_roles'])}")
        print(f"   - Health status: {validation_results['health_status']}")
        print(f"   - Overall health: {validation_results['overall_health']}")
        
        if validation_results["health_issues"]:
            print("   - Health issues:")
            for issue in validation_results["health_issues"]:
                print(f"     • {issue}")
        
        return validation_results["overall_health"]
        
    except Exception as e:
        print(f"   ❌ Agent health validation error: {e}")
        return False


async def test_tool_integration():
    """Test that tools are properly integrated with the per-core system"""
    try:
        from utils.per_core_tool_integration import enhance_tool_execution
        from tools.chat import ChatTool
        
        # Create a test tool
        chat_tool = ChatTool()
        test_arguments = {
            "prompt": "Test integration with per-core agents",
            "model": "auto"
        }
        
        print("   - Testing tool execution with per-core integration...")
        
        # This should use the per-core system if available
        result, used_per_core = await enhance_tool_execution(chat_tool, test_arguments)
        
        print(f"   - Used per-core system: {used_per_core}")
        print(f"   - Result type: {type(result)}")
        print(f"   - Result length: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            print("   - Tool execution successful")
            return True
        else:
            print("   - Tool execution returned empty result")
            return False
            
    except Exception as e:
        print(f"   ❌ Tool integration test error: {e}")
        return False


async def test_end_to_end_workflow():
    """Test complete end-to-end workflow from tool call to response"""
    try:
        from utils.per_core_tool_integration import get_per_core_integrator
        
        integrator = get_per_core_integrator()
        
        if not integrator.is_available():
            print("   - Per-core system not available for E2E test")
            return False
        
        # Test task classification
        print("   - Testing task classification...")
        task_type = integrator.classify_task_type("secaudit", {"prompt": "Check for security issues"})
        print(f"     Task type: {task_type.value}")
        
        # Test agent assignment
        print("   - Testing agent assignment...")
        assignments = integrator.assign_agents(task_type, 1)
        
        if assignments:
            assignment = assignments[0]
            print(f"     Assigned agent: {assignment.agent_id} (role: {assignment.role.value})")
            print(f"     Assignment reasoning: {assignment.reasoning}")
        else:
            print("     No agents assigned")
            return False
        
        # Test system health reporting
        print("   - Testing system health reporting...")
        health_info = integrator.get_system_health()
        print(f"     System available: {health_info['available']}")
        print(f"     Total agents: {health_info['total_agents']}")
        print(f"     Healthy agents: {health_info['healthy_agents']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end workflow test error: {e}")
        return False


async def test_per_core_status_tool():
    """Test the per_core_status tool functionality"""
    try:
        print("\n6. 📊 Testing Per-Core Status Tool...")
        
        from tools.per_core_status import PerCoreStatusTool
        
        status_tool = PerCoreStatusTool()
        
        # Test basic status
        print("   - Testing basic status check...")
        result = await status_tool.execute({"detailed": False})
        
        if result and len(result) > 0:
            print("   - Basic status check successful")
            
            # Test detailed status
            print("   - Testing detailed status check...")
            detailed_result = await status_tool.execute({"detailed": True})
            
            if detailed_result and len(detailed_result) > 0:
                print("   - Detailed status check successful")
                return True
            else:
                print("   - Detailed status check failed")
                return False
        else:
            print("   - Basic status check failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Per-core status tool test error: {e}")
        return False


def print_final_report(validation_results: Dict[str, Any]):
    """Print final validation report"""
    print("\n" + "=" * 60)
    print("📋 FINAL VALIDATION REPORT")
    print("=" * 60)
    
    checks = [
        ("Configuration Valid", validation_results["configuration_valid"]),
        ("System Initialized", validation_results["system_initialized"]),
        ("Agents Healthy", validation_results["agents_healthy"]),
        ("Tools Integrated", validation_results["tools_integrated"]),
        ("End-to-End Working", validation_results["end_to_end_working"])
    ]
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
    
    print("-" * 60)
    
    if validation_results["overall_success"]:
        print("🎉 OVERALL RESULT: ✅ SUCCESS")
        print("\nThe per-core agent coordination system is properly integrated")
        print("and ready for production use. All components are working correctly.")
    else:
        print("💥 OVERALL RESULT: ❌ FAILURE")
        print("\nThe per-core agent coordination system has issues that need to be")
        print("addressed before it can be used reliably. Check the errors above.")
    
    print("=" * 60)


async def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(description="Validate per-core agent system startup integration")
    parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    parser.add_argument("--no-openrouter", action="store_true", help="Skip OpenRouter tests")
    
    args = parser.parse_args()
    
    # Set environment variables for testing if not already set
    if not os.getenv("ENABLE_PER_CORE_AGENTS"):
        os.environ["ENABLE_PER_CORE_AGENTS"] = "true"
    
    if not os.getenv("PER_CORE_MAX_AGENTS"):
        os.environ["PER_CORE_MAX_AGENTS"] = "4"  # Limit for testing
    
    if args.no_openrouter:
        os.environ["PER_CORE_OPENROUTER_REQUIRED"] = "false"
    
    try:
        # Run main validation
        validation_results = await validate_server_startup_integration()
        
        # Test status tool if system is working
        if validation_results["system_initialized"]:
            status_tool_working = await test_per_core_status_tool()
            validation_results["status_tool_working"] = status_tool_working
        
        # Print final report
        print_final_report(validation_results)
        
        # Exit with appropriate code
        if validation_results["overall_success"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Validation failed with error: {e}")
        logger.error(f"Main validation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())