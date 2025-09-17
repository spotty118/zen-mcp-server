#!/usr/bin/env python3
"""
Integration test for per-core agent error handling and logging

This test verifies that the comprehensive error handling and logging
system works correctly with the per-core agent coordination system.
"""

import asyncio
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Set up test environment
os.environ['TESTING'] = '1'

from utils.per_core_error_handling import (
    PerCoreErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    OpenRouterRetryStrategy, AgentRestartStrategy, WorkloadRedistributionStrategy,
    get_per_core_error_handler
)
from utils.per_core_logging import (
    PerCoreLogger, LogCategory, AgentActivityLog, OpenRouterAPILog, SystemEventLog,
    get_per_core_logger, log_agent_activity, log_openrouter_api_call, log_system_event
)
from utils.agent_core import Agent, AgentRole, AgentStatus


class TestPerCoreErrorHandlingIntegration(unittest.TestCase):
    """Test comprehensive error handling and logging integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_agent_id = "test_agent_001"
        self.test_core_id = 0
        
        # Create test logger with temporary directory
        self.logger = PerCoreLogger(log_dir=self.temp_dir, use_json=True)
        
        # Create test error handler
        self.error_handler = PerCoreErrorHandler()
        
        # Create mock agent
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.agent_id = self.test_agent_id
        self.mock_agent.core_id = self.test_core_id
        self.mock_agent.role = AgentRole.GENERALIST
        self.mock_agent.status = AgentStatus.ACTIVE
    
    def tearDown(self):
        """Clean up test environment"""
        # Shutdown logger and error handler
        self.logger.close_all_logs()
        self.error_handler.shutdown()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_categorization(self):
        """Test error categorization functionality"""
        test_cases = [
            (Exception("OpenRouter API key invalid"), ErrorCategory.OPENROUTER_AUTH_ERROR),
            (Exception("Rate limit exceeded"), ErrorCategory.OPENROUTER_RATE_LIMIT),
            (Exception("Connection timeout"), ErrorCategory.OPENROUTER_TIMEOUT),
            (Exception("Server error 500"), ErrorCategory.OPENROUTER_SERVER_ERROR),
            (Exception("Agent communication failed"), ErrorCategory.AGENT_COMMUNICATION_ERROR),
            (Exception("Network connection failed"), ErrorCategory.NETWORK_ERROR),
            (Exception("Memory allocation failed"), ErrorCategory.SYSTEM_RESOURCE_ERROR),
            (Exception("Configuration invalid"), ErrorCategory.CONFIGURATION_ERROR),
            (Exception("Unknown error"), ErrorCategory.UNKNOWN_ERROR)
        ]
        
        for error, expected_category in test_cases:
            with self.subTest(error=str(error)):
                category = self.error_handler.categorize_error(error)
                self.assertEqual(category, expected_category)
    
    def test_severity_determination(self):
        """Test error severity determination"""
        test_cases = [
            (ErrorCategory.SYSTEM_RESOURCE_ERROR, ErrorSeverity.CRITICAL),
            (ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.CRITICAL),
            (ErrorCategory.AGENT_INITIALIZATION_ERROR, ErrorSeverity.HIGH),
            (ErrorCategory.OPENROUTER_AUTH_ERROR, ErrorSeverity.HIGH),
            (ErrorCategory.AGENT_COMMUNICATION_ERROR, ErrorSeverity.MEDIUM),
            (ErrorCategory.OPENROUTER_API_ERROR, ErrorSeverity.MEDIUM),
            (ErrorCategory.OPENROUTER_RATE_LIMIT, ErrorSeverity.LOW),
            (ErrorCategory.OPENROUTER_TIMEOUT, ErrorSeverity.LOW)
        ]
        
        for category, expected_severity in test_cases:
            with self.subTest(category=category):
                error = Exception("Test error")
                severity = self.error_handler.determine_severity(error, category)
                self.assertEqual(severity, expected_severity)
    
    async def test_openrouter_retry_strategy(self):
        """Test OpenRouter retry strategy with exponential backoff"""
        strategy = OpenRouterRetryStrategy()
        
        # Create error context for retryable error
        error_context = ErrorContext(
            error_id="test_001",
            category=ErrorCategory.OPENROUTER_API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            agent_id=self.test_agent_id,
            core_id=self.test_core_id,
            error_message="API call failed",
            retry_count=0,
            max_retries=3
        )
        
        # Mock API client
        mock_api_client = Mock()
        mock_api_client.openrouter_circuit_half_open = False
        
        # Test retry strategy
        start_time = time.time()
        success = await strategy.recover(error_context, api_client=mock_api_client)
        end_time = time.time()
        
        # Should succeed (setup successful)
        self.assertTrue(success)
        
        # Should have waited for backoff
        self.assertGreater(end_time - start_time, 0.5)  # At least some delay
        
        # Retry count should be incremented
        self.assertEqual(error_context.retry_count, 1)
    
    async def test_agent_restart_strategy(self):
        """Test agent restart strategy"""
        strategy = AgentRestartStrategy()
        
        # Create error context for agent error
        error_context = ErrorContext(
            error_id="test_002",
            category=ErrorCategory.AGENT_COMMUNICATION_ERROR,
            severity=ErrorSeverity.MEDIUM,
            agent_id=self.test_agent_id,
            core_id=self.test_core_id,
            error_message="Agent communication failed"
        )
        
        # Mock agent manager
        mock_agent_manager = Mock()
        mock_agent_manager.get_agent_by_id.return_value = self.mock_agent
        mock_agent_manager.restart_agent = AsyncMock(return_value=True)
        
        # Test restart strategy
        success = await strategy.recover(error_context, agent_manager=mock_agent_manager)
        
        # Should succeed
        self.assertTrue(success)
        
        # Should have called restart_agent
        mock_agent_manager.restart_agent.assert_called_once_with(self.test_agent_id)
    
    async def test_workload_redistribution_strategy(self):
        """Test workload redistribution strategy"""
        strategy = WorkloadRedistributionStrategy()
        
        # Create error context for system error
        error_context = ErrorContext(
            error_id="test_003",
            category=ErrorCategory.SYSTEM_RESOURCE_ERROR,
            severity=ErrorSeverity.CRITICAL,
            agent_id=self.test_agent_id,
            core_id=self.test_core_id,
            error_message="System resource exhausted"
        )
        
        # Mock agent manager
        mock_agent_manager = Mock()
        mock_agent_manager.redistribute_workload.return_value = True
        
        # Test redistribution strategy
        success = await strategy.recover(error_context, agent_manager=mock_agent_manager)
        
        # Should succeed
        self.assertTrue(success)
        
        # Should have called redistribute_workload
        mock_agent_manager.redistribute_workload.assert_called_once_with(self.test_agent_id)
    
    async def test_comprehensive_error_handling(self):
        """Test comprehensive error handling workflow"""
        # Create a test error
        test_error = Exception("OpenRouter API timeout")
        
        # Handle the error
        success = await self.error_handler.handle_error(
            error=test_error,
            agent_id=self.test_agent_id,
            core_id=self.test_core_id,
            context={"operation": "api_call", "model": "gpt-4"}
        )
        
        # Should attempt recovery
        self.assertIsInstance(success, bool)
        
        # Check error statistics
        stats = self.error_handler.get_error_statistics()
        self.assertGreater(stats["total_errors"], 0)
        self.assertIn("openrouter_timeout", [cat for cat in stats["error_categories"].keys()])
    
    def test_structured_logging(self):
        """Test structured logging functionality"""
        # Test agent activity logging
        activity_log = AgentActivityLog(
            agent_id=self.test_agent_id,
            activity_type="api_call",
            activity_details={"model": "gpt-4", "tokens": 150},
            duration_ms=1500.0,
            success=True
        )
        
        self.logger.log_agent_activity(activity_log)
        
        # Test OpenRouter API logging
        api_log = OpenRouterAPILog(
            agent_id=self.test_agent_id,
            api_call_id="call_001",
            model_used="gpt-4",
            request_type="thinking_session",
            response_time_ms=2000.0,
            tokens_used=150,
            success=True
        )
        
        self.logger.log_openrouter_api_call(api_log)
        
        # Test system event logging
        event_log = SystemEventLog(
            event_type="agent_initialization",
            event_details={"total_agents": 4, "success": True},
            severity="INFO",
            affected_agents=[self.test_agent_id]
        )
        
        self.logger.log_system_event(event_log)
        
        # Flush logs to ensure they're written
        self.logger.flush_all_logs()
        
        # Verify log files were created
        log_files = os.listdir(self.temp_dir)
        self.assertIn("per_core_agents.log", log_files)
        self.assertIn("openrouter_api.log", log_files)
        self.assertIn("system_events.log", log_files)
    
    def test_convenience_logging_functions(self):
        """Test convenience logging functions"""
        # Test agent activity logging
        log_agent_activity(
            agent_id=self.test_agent_id,
            activity_type="test_activity",
            activity_details={"test": "data"},
            duration_ms=100.0,
            success=True
        )
        
        # Test OpenRouter API call logging
        log_openrouter_api_call(
            agent_id=self.test_agent_id,
            api_call_id="test_call",
            model_used="test_model",
            request_type="test",
            response_time_ms=200.0,
            success=True
        )
        
        # Test system event logging
        log_system_event(
            event_type="test_event",
            event_details={"test": "event"},
            severity="INFO"
        )
        
        # Should not raise any exceptions
        self.assertTrue(True)
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker functionality"""
        # Create multiple failures to trigger circuit breaker
        test_error = Exception("OpenRouter server error")
        
        async def run_multiple_failures():
            for i in range(6):  # Exceed threshold of 5
                await self.error_handler.handle_error(
                    error=test_error,
                    agent_id=self.test_agent_id,
                    core_id=self.test_core_id,
                    context={"attempt": i}
                )
        
        # Run the test
        asyncio.run(run_multiple_failures())
        
        # Check circuit breaker status
        stats = self.error_handler.get_error_statistics()
        circuit_breakers = stats.get("circuit_breakers", {})
        
        # Should have circuit breaker information
        self.assertIsInstance(circuit_breakers, dict)
    
    def test_error_recovery_metrics(self):
        """Test error recovery metrics collection"""
        # Get initial statistics
        initial_stats = self.error_handler.get_error_statistics()
        
        # Create and handle an error
        test_error = Exception("Test error for metrics")
        
        async def handle_test_error():
            await self.error_handler.handle_error(
                error=test_error,
                agent_id=self.test_agent_id,
                core_id=self.test_core_id
            )
        
        asyncio.run(handle_test_error())
        
        # Get updated statistics
        updated_stats = self.error_handler.get_error_statistics()
        
        # Should have more errors
        self.assertGreater(updated_stats["total_errors"], initial_stats["total_errors"])
        
        # Should have recovery strategy information
        self.assertIn("recovery_strategies", updated_stats)
        recovery_strategies = updated_stats["recovery_strategies"]
        self.assertIsInstance(recovery_strategies, dict)
        
        # Should have strategy names
        expected_strategies = ["openrouter_retry", "agent_restart", "workload_redistribution"]
        for strategy_name in expected_strategies:
            self.assertIn(strategy_name, recovery_strategies)
    
    def test_log_statistics(self):
        """Test log statistics collection"""
        # Generate some log entries
        for i in range(5):
            log_agent_activity(
                agent_id=f"agent_{i}",
                activity_type="test_activity",
                activity_details={"iteration": i}
            )
        
        # Get log statistics
        stats = self.logger.get_log_statistics()
        
        # Verify statistics structure
        self.assertIn("log_directory", stats)
        self.assertIn("active_loggers", stats)
        self.assertIn("log_files", stats)
        self.assertIn("total_log_size_bytes", stats)
        
        # Should have log files
        self.assertGreater(len(stats["log_files"]), 0)


class TestErrorHandlingWithMockAgentManager(unittest.TestCase):
    """Test error handling integration with mock agent manager"""
    
    def setUp(self):
        """Set up test with mock agent manager"""
        self.mock_agent_manager = Mock()
        self.mock_agent = Mock(spec=Agent)
        self.mock_agent.agent_id = "test_agent"
        self.mock_agent.core_id = 0
        self.mock_agent.role = AgentRole.GENERALIST
        
        self.mock_agent_manager.get_agent_by_id.return_value = self.mock_agent
        self.mock_agent_manager.restart_agent = AsyncMock(return_value=True)
        self.mock_agent_manager.redistribute_workload.return_value = True
    
    async def test_agent_restart_integration(self):
        """Test agent restart through error handler"""
        error_handler = PerCoreErrorHandler()
        
        # Create agent initialization error
        init_error = Exception("Agent initialization failed")
        
        # Handle the error with agent manager
        success = await error_handler.handle_error(
            error=init_error,
            agent_id="test_agent",
            core_id=0,
            context={"operation": "initialization"},
            agent_manager=self.mock_agent_manager
        )
        
        # Should attempt recovery
        self.assertIsInstance(success, bool)
        
        # Should have called restart_agent if recovery was attempted
        if success:
            self.mock_agent_manager.restart_agent.assert_called()
    
    async def test_workload_redistribution_integration(self):
        """Test workload redistribution through error handler"""
        error_handler = PerCoreErrorHandler()
        
        # Create system resource error
        resource_error = Exception("System resource exhausted")
        
        # Handle the error with agent manager
        success = await error_handler.handle_error(
            error=resource_error,
            agent_id="test_agent",
            core_id=0,
            context={"operation": "resource_allocation"},
            agent_manager=self.mock_agent_manager
        )
        
        # Should attempt recovery
        self.assertIsInstance(success, bool)
        
        # Should have called redistribute_workload if recovery was attempted
        if success:
            self.mock_agent_manager.redistribute_workload.assert_called()


def run_integration_tests():
    """Run all integration tests"""
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPerCoreErrorHandlingIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingWithMockAgentManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)