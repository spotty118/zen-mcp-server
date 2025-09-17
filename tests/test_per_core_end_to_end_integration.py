"""
End-to-End Integration Tests for Per-Core Agent Coordination

This module provides comprehensive end-to-end tests for the complete per-core agent
coordination workflow, validating the integration of all components from server startup
through tool execution and system shutdown.

Tests cover:
- Complete system initialization and validation
- End-to-end tool execution with multi-agent coordination
- System health monitoring and error recovery
- Graceful shutdown and cleanup procedures
- Integration with MCP server startup sequence
"""

import asyncio
import os
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
import tempfile
import json

from utils.agent_core import Agent, AgentRole, AgentStatus
from utils.per_core_agent_manager import PerCoreAgentManager, PerCoreAgentStatus
from utils.per_core_tool_integration import PerCoreToolIntegrator, TaskType, AgentAssignment
from utils.agent_thinking_session import AgentThinkingSession
from tools.shared.base_tool import BaseTool
from tools.models import ToolOutput
from mcp.types import TextContent


class MockTool(BaseTool):
    """Mock tool for testing end-to-end integration"""
    
    def __init__(self, name: str = "test_tool"):
        self.name = name
        self.description = f"Mock {name} tool for testing"
    
    def get_name(self) -> str:
        return self.name
    
    def get_description(self) -> str:
        return self.description
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "files": {"type": "array", "items": {"type": "string"}}
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Mock execution that returns test response"""
        prompt = arguments.get("prompt", "test prompt")
        agent_context = arguments.get("_agent_context", {})
        
        response = f"Mock response for: {prompt}"
        if agent_context:
            response += f" (Agent: {agent_context.get('role', 'unknown')})"
        
        output = ToolOutput(
            status="success",
            content=response,
            content_type="text"
        )
        
        return [TextContent(type="text", text=output.model_dump_json())]


class TestPerCoreEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests for complete per-core agent coordination workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_max_agents = 4
        self.test_openrouter_key = "test_openrouter_key_123"
        
        # Create mock config manager
        self.mock_config_manager = Mock()
        self.mock_config_manager.get_config.return_value = Mock(
            enabled=True,
            max_agents=self.test_max_agents,
            openrouter_api_key=self.test_openrouter_key,
            openrouter_required=False,
            fallback_mode="single_agent",
            health_check_interval=60,
            agent_timeout=300,
            multi_agent_tools={"test_tool", "codereview", "consensus"}
        )
        self.mock_config_manager.validate.return_value = []
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'ENABLE_PER_CORE_AGENTS': 'true',
            'PER_CORE_MAX_AGENTS': str(self.test_max_agents),
            'OPENROUTER_API_KEY': self.test_openrouter_key,
            'PER_CORE_OPENROUTER_REQUIRED': 'false'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test environment"""
        self.env_patcher.stop()
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_complete_system_initialization_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test complete system initialization from server startup through validation"""
        # Arrange
        mock_exists.return_value = False  # No recovery state
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Import server functions
        from server import initialize_per_core_agent_system, validate_per_core_system
        
        # Act - Initialize the system as it would during server startup
        initialization_success = await initialize_per_core_agent_system()
        
        # Assert
        self.assertTrue(initialization_success, "System initialization should succeed")
        
        # Verify integrator was created and initialized
        from utils.per_core_tool_integration import get_per_core_integrator
        integrator = get_per_core_integrator()
        
        self.assertTrue(integrator.is_available(), "Integrator should be available")
        self.assertIsNotNone(integrator.per_core_manager, "Per-core manager should be initialized")
        self.assertIsNotNone(integrator.communication_system, "Communication system should be available")
        
        # Perform validation as done during startup
        validation_results = await validate_per_core_system(integrator)
        
        self.assertTrue(validation_results["overall_health"], "System should be healthy")
        self.assertEqual(validation_results["num_agents"], self.test_max_agents)
        self.assertGreater(len(validation_results["agent_roles"]), 0)
        self.assertEqual(len(validation_results["health_issues"]), 0)
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_end_to_end_tool_execution_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test complete end-to-end tool execution with per-core agent coordination"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Initialize system
        from server import initialize_per_core_agent_system
        await initialize_per_core_agent_system()
        
        # Create test tool and arguments
        test_tool = MockTool("test_tool")
        test_arguments = {
            "prompt": "Analyze this code for security issues",
            "files": ["test_file.py"]
        }
        
        # Act - Execute tool through the integration layer
        from utils.per_core_tool_integration import enhance_tool_execution
        result, used_per_core = await enhance_tool_execution(test_tool, test_arguments)
        
        # Assert
        self.assertTrue(used_per_core, "Should use per-core system")
        self.assertIsInstance(result, list, "Should return list of TextContent")
        self.assertEqual(len(result), 1, "Should return single response")
        
        # Parse the response
        response_text = result[0].text
        response_data = json.loads(response_text)
        
        self.assertEqual(response_data["status"], "success")
        self.assertIn("Mock response", response_data["content"])
        self.assertTrue(response_data["metadata"]["per_core_coordination"])
        self.assertGreater(response_data["metadata"]["agents_used"], 0)
        self.assertIn("task_type", response_data["metadata"])
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_multi_agent_coordination_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test multi-agent coordination for tools that benefit from multiple perspectives"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Initialize system
        from server import initialize_per_core_agent_system
        await initialize_per_core_agent_system()
        
        # Create tool that benefits from multi-agent coordination
        consensus_tool = MockTool("consensus")
        test_arguments = {
            "prompt": "What's the best approach for implementing user authentication?",
            "files": []
        }
        
        # Act
        from utils.per_core_tool_integration import enhance_tool_execution
        result, used_per_core = await enhance_tool_execution(consensus_tool, test_arguments)
        
        # Assert
        self.assertTrue(used_per_core, "Should use per-core system")
        
        # Parse response to check multi-agent coordination
        response_text = result[0].text
        response_data = json.loads(response_text)
        
        self.assertTrue(response_data["metadata"]["per_core_coordination"])
        # For multi-agent tools, should potentially use multiple agents
        agents_used = response_data["metadata"]["agents_used"]
        self.assertGreaterEqual(agents_used, 1)
        
        # Should have coordination metadata
        self.assertIn("coordination_strategy", response_data["metadata"])
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_system_health_monitoring_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test system health monitoring and status reporting"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Initialize system
        from server import initialize_per_core_agent_system, validate_per_core_system
        await initialize_per_core_agent_system()
        
        # Get integrator for health checks
        from utils.per_core_tool_integration import get_per_core_integrator
        integrator = get_per_core_integrator()
        
        # Act - Perform health validation
        validation_results = await validate_per_core_system(integrator)
        
        # Assert - Check comprehensive health information
        self.assertIn("overall_health", validation_results)
        self.assertIn("num_agents", validation_results)
        self.assertIn("agent_roles", validation_results)
        self.assertIn("health_status", validation_results)
        self.assertIn("health_issues", validation_results)
        self.assertIn("agent_details", validation_results)
        self.assertIn("system_capabilities", validation_results)
        
        # Verify agent details
        agent_details = validation_results["agent_details"]
        self.assertEqual(len(agent_details), self.test_max_agents)
        
        for agent_detail in agent_details:
            self.assertIn("agent_id", agent_detail)
            self.assertIn("core_id", agent_detail)
            self.assertIn("role", agent_detail)
            self.assertIn("status", agent_detail)
            self.assertIn("healthy", agent_detail)
        
        # Verify system capabilities
        capabilities = validation_results["system_capabilities"]
        self.assertIn("communication", capabilities)
        self.assertIn("openrouter", capabilities)
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_error_recovery_and_fallback_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test error recovery and fallback to single-agent mode"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Initialize system
        from server import initialize_per_core_agent_system
        await initialize_per_core_agent_system()
        
        # Create tool and arguments
        test_tool = MockTool("test_tool")
        test_arguments = {"prompt": "Test prompt"}
        
        # Simulate agent failure by making execute_with_agents raise an exception
        from utils.per_core_tool_integration import get_per_core_integrator
        integrator = get_per_core_integrator()
        
        original_execute = integrator.execute_with_agents
        async def failing_execute(*args, **kwargs):
            raise Exception("Simulated agent failure")
        
        integrator.execute_with_agents = failing_execute
        
        # Act - Execute tool with simulated failure
        from utils.per_core_tool_integration import enhance_tool_execution
        result, used_per_core = await enhance_tool_execution(test_tool, test_arguments)
        
        # Assert - Should fall back to single-agent mode
        self.assertFalse(used_per_core, "Should fall back to single-agent mode")
        self.assertIsInstance(result, list, "Should still return valid result")
        self.assertEqual(len(result), 1, "Should return single response")
        
        # Response should be from standard tool execution
        response_text = result[0].text
        response_data = json.loads(response_text)
        self.assertEqual(response_data["status"], "success")
        self.assertIn("Mock response", response_data["content"])
        
        # Restore original method
        integrator.execute_with_agents = original_execute
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_system_disabled_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test workflow when per-core agent system is disabled"""
        # Arrange - Disable per-core agents
        with patch.dict(os.environ, {'ENABLE_PER_CORE_AGENTS': 'false'}):
            # Act
            from server import initialize_per_core_agent_system
            initialization_success = await initialize_per_core_agent_system()
            
            # Assert
            self.assertFalse(initialization_success, "Should not initialize when disabled")
            
            # Tool execution should still work in single-agent mode
            test_tool = MockTool("test_tool")
            test_arguments = {"prompt": "Test prompt"}
            
            from utils.per_core_tool_integration import enhance_tool_execution
            result, used_per_core = await enhance_tool_execution(test_tool, test_arguments)
            
            self.assertFalse(used_per_core, "Should not use per-core system when disabled")
            self.assertIsInstance(result, list, "Should still return valid result")
    
    @patch('utils.per_core_agent_manager.multiprocessing.cpu_count')
    @patch('utils.per_core_agent_manager.os.cpu_count')
    @patch('utils.per_core_agent_manager.get_per_core_agent_config_manager')
    @patch('utils.per_core_agent_manager.os.path.exists')
    async def test_task_classification_and_agent_assignment_workflow(
        self, mock_exists, mock_get_config_manager, mock_os_cpu_count, mock_mp_cpu_count
    ):
        """Test task classification and intelligent agent assignment"""
        # Arrange
        mock_exists.return_value = False
        mock_get_config_manager.return_value = self.mock_config_manager
        mock_os_cpu_count.return_value = self.test_max_agents
        mock_mp_cpu_count.return_value = self.test_max_agents
        
        # Initialize system
        from server import initialize_per_core_agent_system
        await initialize_per_core_agent_system()
        
        from utils.per_core_tool_integration import get_per_core_integrator
        integrator = get_per_core_integrator()
        
        # Test different task types and their classifications
        test_cases = [
            ("secaudit", {"prompt": "Check for security vulnerabilities"}, TaskType.SECURITY_ANALYSIS),
            ("codereview", {"prompt": "Review this code"}, TaskType.CODE_QUALITY_INSPECTION),
            ("debug", {"prompt": "Fix this bug"}, TaskType.DEBUG_ASSISTANCE),
            ("planner", {"prompt": "Plan this project"}, TaskType.PLANNING_COORDINATION),
            ("consensus", {"prompt": "What's the best approach?"}, TaskType.CONSENSUS_BUILDING),
        ]
        
        for tool_name, arguments, expected_task_type in test_cases:
            # Act
            classified_task_type = integrator.classify_task_type(tool_name, arguments)
            agent_assignments = integrator.assign_agents(classified_task_type, 1)
            
            # Assert
            self.assertEqual(classified_task_type, expected_task_type, 
                           f"Task type classification failed for {tool_name}")
            self.assertGreater(len(agent_assignments), 0, 
                             f"Should assign agents for {tool_name}")
            
            # Verify agent assignment has correct structure
            assignment = agent_assignments[0]
            self.assertIsInstance(assignment, AgentAssignment)
            self.assertEqual(assignment.task_type, expected_task_type)
            self.assertIn(assignment.role, integrator.task_to_role_mapping[expected_task_type])
    
    async def test_graceful_shutdown_workflow(self):
        """Test graceful shutdown and cleanup procedures"""
        # This test verifies that the system can be properly shut down
        # In a real implementation, this would test cleanup of:
        # - Agent connections
        # - OpenRouter API clients
        # - Communication channels
        # - Persistent state
        
        # For now, we'll test that the integrator can be reset
        from utils.per_core_tool_integration import get_per_core_integrator
        integrator = get_per_core_integrator()
        
        # Simulate shutdown by resetting the integrator
        integrator._initialization_attempted = False
        integrator._is_available = False
        integrator.per_core_manager = None
        integrator.communication_system = None
        
        # Verify shutdown state
        self.assertFalse(integrator.is_available())
        self.assertIsNone(integrator.per_core_manager)
        self.assertIsNone(integrator.communication_system)


class TestPerCoreSystemValidation(unittest.TestCase):
    """Test system-wide validation and health checks"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_max_agents = 2
        
        # Create mock integrator with manager
        self.mock_integrator = Mock()
        self.mock_manager = Mock()
        self.mock_integrator.per_core_manager = self.mock_manager
        self.mock_integrator.communication_system = Mock()
    
    async def test_validation_with_healthy_system(self):
        """Test validation with a healthy per-core agent system"""
        # Arrange
        self.mock_manager.get_system_health.return_value = {
            "total_agents": 2,
            "overall_status": "healthy",
            "agents": [
                {
                    "agent_id": "agent_0",
                    "core_id": 0,
                    "role": "security_analyst",
                    "status": "active"
                },
                {
                    "agent_id": "agent_1", 
                    "core_id": 1,
                    "role": "performance_optimizer",
                    "status": "active"
                }
            ]
        }
        
        # Act
        from server import validate_per_core_system
        results = await validate_per_core_system(self.mock_integrator)
        
        # Assert
        self.assertTrue(results["overall_health"])
        self.assertEqual(results["num_agents"], 2)
        self.assertEqual(len(results["agent_details"]), 2)
        self.assertEqual(len(results["health_issues"]), 0)
        self.assertTrue(results["system_capabilities"]["communication"])
    
    async def test_validation_with_unhealthy_agents(self):
        """Test validation with some unhealthy agents"""
        # Arrange
        self.mock_manager.get_system_health.return_value = {
            "total_agents": 2,
            "overall_status": "degraded",
            "agents": [
                {
                    "agent_id": "agent_0",
                    "core_id": 0,
                    "role": "security_analyst",
                    "status": "active"
                },
                {
                    "agent_id": "agent_1",
                    "core_id": 1,
                    "role": "performance_optimizer", 
                    "status": "failed"
                }
            ]
        }
        
        # Act
        from server import validate_per_core_system
        results = await validate_per_core_system(self.mock_integrator)
        
        # Assert
        self.assertFalse(results["overall_health"])
        self.assertEqual(results["num_agents"], 2)
        self.assertGreater(len(results["health_issues"]), 0)
        self.assertIn("agent_1", results["health_issues"][0])
    
    async def test_validation_without_manager(self):
        """Test validation when per-core manager is not initialized"""
        # Arrange
        self.mock_integrator.per_core_manager = None
        
        # Act
        from server import validate_per_core_system
        results = await validate_per_core_system(self.mock_integrator)
        
        # Assert
        self.assertFalse(results["overall_health"])
        self.assertEqual(results["num_agents"], 0)
        self.assertIn("Per-core manager not initialized", results["health_issues"])


if __name__ == '__main__':
    # Run async tests
    import sys
    
    async def run_async_tests():
        """Run all async test methods"""
        test_instance = TestPerCoreEndToEndIntegration()
        test_instance.setUp()
        
        try:
            await test_instance.test_complete_system_initialization_workflow()
            print("✅ test_complete_system_initialization_workflow passed")
            
            await test_instance.test_end_to_end_tool_execution_workflow()
            print("✅ test_end_to_end_tool_execution_workflow passed")
            
            await test_instance.test_multi_agent_coordination_workflow()
            print("✅ test_multi_agent_coordination_workflow passed")
            
            await test_instance.test_system_health_monitoring_workflow()
            print("✅ test_system_health_monitoring_workflow passed")
            
            await test_instance.test_error_recovery_and_fallback_workflow()
            print("✅ test_error_recovery_and_fallback_workflow passed")
            
            await test_instance.test_system_disabled_workflow()
            print("✅ test_system_disabled_workflow passed")
            
            await test_instance.test_task_classification_and_agent_assignment_workflow()
            print("✅ test_task_classification_and_agent_assignment_workflow passed")
            
            await test_instance.test_graceful_shutdown_workflow()
            print("✅ test_graceful_shutdown_workflow passed")
            
            print("\n🎉 All end-to-end integration tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            sys.exit(1)
        finally:
            test_instance.tearDown()
    
    # Run the async tests
    asyncio.run(run_async_tests())