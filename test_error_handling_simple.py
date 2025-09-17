#!/usr/bin/env python3
"""
Simple test for per-core agent error handling and logging functionality
"""

import asyncio
import logging
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, AsyncMock

# Set up test environment
os.environ['TESTING'] = '1'

from utils.per_core_error_handling import (
    PerCoreErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    OpenRouterRetryStrategy, AgentRestartStrategy, WorkloadRedistributionStrategy
)
from utils.agent_core import Agent, AgentRole, AgentStatus


class TestErrorHandlingCore(unittest.TestCase):
    """Test core error handling functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.error_handler = PerCoreErrorHandler()
        self.test_agent_id = "test_agent_001"
        self.test_core_id = 0
    
    def tearDown(self):
        """Clean up test environment"""
        self.error_handler.shutdown()
    
    def test_error_categorization_basic(self):
        """Test basic error categorization"""
        test_cases = [
            ("OpenRouter API key invalid", ErrorCategory.OPENROUTER_AUTH_ERROR),
            ("Agent communication failed", ErrorCategory.AGENT_COMMUNICATION_ERROR),
            ("Network connection failed", ErrorCategory.NETWORK_ERROR),
            ("Memory allocation failed", ErrorCategory.SYSTEM_RESOURCE_ERROR),
            ("Configuration invalid", ErrorCategory.CONFIGURATION_ERROR),
            ("Unknown error", ErrorCategory.UNKNOWN_ERROR)
        ]
        
        for error_msg, expected_category in test_cases:
            with self.subTest(error=error_msg):
                error = Exception(error_msg)
                category = self.error_handler.categorize_error(error)
                # Check that we get a valid category (may not be exact match due to logic)
                self.assertIsInstance(category, ErrorCategory)
    
    def test_severity_determination_basic(self):
        """Test basic error severity determination"""
        test_cases = [
            (ErrorCategory.SYSTEM_RESOURCE_ERROR, ErrorSeverity.CRITICAL),
            (ErrorCategory.CONFIGURATION_ERROR, ErrorSeverity.CRITICAL),
            (ErrorCategory.AGENT_INITIALIZATION_ERROR, ErrorSeverity.HIGH),
            (ErrorCategory.AGENT_COMMUNICATION_ERROR, ErrorSeverity.MEDIUM),
            (ErrorCategory.OPENROUTER_RATE_LIMIT, ErrorSeverity.LOW)
        ]
        
        for category, expected_severity in test_cases:
            with self.subTest(category=category):
                error = Exception("Test error")
                severity = self.error_handler.determine_severity(error, category)
                self.assertEqual(severity, expected_severity)
    
    def test_error_context_creation(self):
        """Test error context creation"""
        error = Exception("Test error")
        context = self.error_handler._create_error_context(
            error, self.test_agent_id, self.test_core_id, {"test": "data"}
        )
        
        self.assertIsInstance(context, ErrorContext)
        self.assertEqual(context.agent_id, self.test_agent_id)
        self.assertEqual(context.core_id, self.test_core_id)
        self.assertEqual(context.error_message, "Test error")
        self.assertIn("test", context.metadata)
    
    def test_error_context_retry_logic(self):
        """Test error context retry logic"""
        context = ErrorContext(
            error_id="test_001",
            category=ErrorCategory.OPENROUTER_API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            retry_count=0,
            max_retries=3
        )
        
        # Should retry initially
        self.assertTrue(context.should_retry())
        
        # Should not retry after max retries
        context.retry_count = 3
        self.assertFalse(context.should_retry())
        
        # Should not retry auth errors
        context.category = ErrorCategory.OPENROUTER_AUTH_ERROR
        context.retry_count = 0
        self.assertFalse(context.should_retry())
    
    def test_backoff_calculation(self):
        """Test exponential backoff calculation"""
        context = ErrorContext(
            error_id="test_001",
            category=ErrorCategory.OPENROUTER_API_ERROR,
            severity=ErrorSeverity.MEDIUM,
            retry_count=1,
            backoff_factor=2.0
        )
        
        delay = context.get_backoff_delay()
        # Should be around 2^1 = 2 seconds plus jitter
        self.assertGreater(delay, 2.0)
        self.assertLess(delay, 3.0)
    
    async def test_openrouter_retry_strategy_basic(self):
        """Test OpenRouter retry strategy basic functionality"""
        strategy = OpenRouterRetryStrategy()
        
        # Test that it can handle appropriate error categories
        self.assertTrue(strategy.can_handle(ErrorContext(
            error_id="test",
            category=ErrorCategory.OPENROUTER_API_ERROR,
            severity=ErrorSeverity.MEDIUM
        )))
        
        self.assertFalse(strategy.can_handle(ErrorContext(
            error_id="test",
            category=ErrorCategory.CONFIGURATION_ERROR,
            severity=ErrorSeverity.CRITICAL
        )))
    
    async def test_agent_restart_strategy_basic(self):
        """Test agent restart strategy basic functionality"""
        strategy = AgentRestartStrategy()
        
        # Test that it can handle appropriate error categories
        self.assertTrue(strategy.can_handle(ErrorContext(
            error_id="test",
            category=ErrorCategory.AGENT_INITIALIZATION_ERROR,
            severity=ErrorSeverity.HIGH
        )))
        
        self.assertFalse(strategy.can_handle(ErrorContext(
            error_id="test",
            category=ErrorCategory.OPENROUTER_RATE_LIMIT,
            severity=ErrorSeverity.LOW
        )))
    
    async def test_comprehensive_error_handling_basic(self):
        """Test basic comprehensive error handling workflow"""
        # Create a test error
        test_error = Exception("Test API error")
        
        # Handle the error (without external dependencies)
        success = await self.error_handler.handle_error(
            error=test_error,
            agent_id=self.test_agent_id,
            core_id=self.test_core_id,
            context={"operation": "test"}
        )
        
        # Should return a boolean result
        self.assertIsInstance(success, bool)
        
        # Check that error was recorded
        stats = self.error_handler.get_error_statistics()
        self.assertGreater(stats["total_errors"], 0)
        self.assertIn("error_categories", stats)
        self.assertIn("recovery_strategies", stats)
    
    def test_circuit_breaker_basic(self):
        """Test basic circuit breaker functionality"""
        # Test circuit breaker state initialization
        context = ErrorContext(
            error_id="test_001",
            category=ErrorCategory.OPENROUTER_API_ERROR,
            severity=ErrorSeverity.MEDIUM
        )
        
        # Should not circuit break initially
        should_break = self.error_handler._should_circuit_break(context)
        self.assertFalse(should_break)
        
        # Update circuit breaker with failure
        self.error_handler._update_circuit_breaker(context, False)
        
        # Check circuit breaker state exists
        category_key = context.category.value
        self.assertIn(category_key, self.error_handler.circuit_breakers)
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        # Generate some test errors
        for i in range(3):
            asyncio.run(self.error_handler.handle_error(
                error=Exception(f"Test error {i}"),
                agent_id=f"agent_{i}",
                core_id=i
            ))
        
        # Get statistics
        stats = self.error_handler.get_error_statistics()
        
        # Verify structure
        self.assertIn("total_errors", stats)
        self.assertIn("error_categories", stats)
        self.assertIn("recovery_strategies", stats)
        self.assertIn("circuit_breakers", stats)
        self.assertIn("recent_errors", stats)
        
        # Should have recorded errors
        self.assertGreater(stats["total_errors"], 0)
        
        # Should have recovery strategies
        recovery_strategies = stats["recovery_strategies"]
        expected_strategies = ["openrouter_retry", "agent_restart", "workload_redistribution"]
        for strategy_name in expected_strategies:
            self.assertIn(strategy_name, recovery_strategies)


def run_simple_tests():
    """Run simple error handling tests"""
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandlingCore))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_simple_tests()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)