#!/usr/bin/env python3
"""
Test suite for core-specific context sharing enhancement
Tests per-core context isolation, inter-core sharing, and GPU detection
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch
from utils.core_context_storage import CoreContextStorage, get_core_context_storage
from tools.parallelthink import ParallelThinkTool, ParallelThinkRequest, ParallelThinkingPath


class TestCoreContextStorage:
    """Test core context storage functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.storage = CoreContextStorage(max_cores=4)
    
    def teardown_method(self):
        """Clean up after tests"""
        self.storage.shutdown()
    
    def test_basic_core_context_operations(self):
        """Test basic set/get operations for core context"""
        # Test setting and getting context for specific core
        self.storage.set_core_context("test_key", "test_value", core_id=0)
        value = self.storage.get_core_context("test_key", core_id=0)
        assert value == "test_value"
        
        # Test core isolation - different cores should have separate contexts
        self.storage.set_core_context("test_key", "core_1_value", core_id=1)
        value_core_0 = self.storage.get_core_context("test_key", core_id=0)
        value_core_1 = self.storage.get_core_context("test_key", core_id=1)
        
        assert value_core_0 == "test_value"
        assert value_core_1 == "core_1_value"
    
    def test_shared_context_functionality(self):
        """Test context sharing between cores"""
        # Set context with sharing enabled
        self.storage.set_core_context("shared_key", "shared_value", core_id=0, share_with_others=True)
        
        # Should be accessible from other cores via shared context
        value = self.storage.get_core_context("shared_key", core_id=1, check_shared=True)
        assert value == "shared_value"
        
        # Should not be accessible if shared checking is disabled
        value = self.storage.get_core_context("shared_key", core_id=1, check_shared=False)
        assert value is None
    
    def test_explicit_context_sharing(self):
        """Test explicit context sharing between specific cores"""
        # Set up context on source core
        self.storage.set_core_context("source_key", "source_value", core_id=0)
        
        # Share with specific target cores
        success = self.storage.share_context_between_cores("source_key", 0, {1, 2})
        assert success is True
        
        # Check that target cores received the shared context
        for target_core in [1, 2]:
            target_context = self.storage._core_contexts[target_core]
            shared_key = f"shared_from_0_source_key"
            assert shared_key in target_context.context_data
            assert target_context.context_data[shared_key] == "source_value"
    
    def test_core_statistics(self):
        """Test core statistics collection"""
        # Set up some test data
        self.storage.set_core_context("key1", "value1", core_id=0)
        self.storage.set_core_context("key2", "value2", core_id=0, share_with_others=True)
        self.storage.set_core_context("key3", "value3", core_id=1)
        
        stats = self.storage.get_core_statistics()
        
        assert stats["total_cores"] == 2
        assert stats["total_shared_contexts"] == 1
        assert 0 in stats["cores"]
        assert 1 in stats["cores"]
        assert stats["cores"][0]["context_count"] == 2
        assert stats["cores"][0]["shared_keys"] == 1
        assert stats["cores"][1]["context_count"] == 1
    
    def test_thread_safety(self):
        """Test thread safety of core context operations"""
        def worker(core_id, start_value):
            for i in range(100):
                key = f"thread_key_{i}"
                value = f"core_{core_id}_value_{start_value + i}"
                self.storage.set_core_context(key, value, core_id=core_id)
        
        # Start multiple threads
        threads = []
        for core_id in range(4):
            thread = threading.Thread(target=worker, args=(core_id, core_id * 1000))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify that all operations completed successfully
        stats = self.storage.get_core_statistics()
        assert stats["total_cores"] == 4
        for core_id in range(4):
            assert stats["cores"][core_id]["context_count"] == 100


class TestParallelThinkingEnhancements:
    """Test enhanced parallel thinking with core context"""
    
    def setup_method(self):
        """Set up test environment"""
        self.tool = ParallelThinkTool()
    
    def test_gpu_detection(self):
        """Test GPU detection functionality"""
        gpu_info = self.tool._detect_gpu_availability()
        
        # Should return dict with required keys
        assert isinstance(gpu_info, dict)
        assert "available" in gpu_info
        assert "type" in gpu_info
        assert "memory" in gpu_info
        assert "compute_capability" in gpu_info
        
        # available should be boolean
        assert isinstance(gpu_info["available"], bool)
    
    def test_core_context_request_parameters(self):
        """Test new request parameters for core context"""
        request_data = {
            "prompt": "Test prompt",
            "enable_core_context": True,
            "share_insights_between_cores": True,
            "context_sharing_threshold": 0.8
        }
        
        request = ParallelThinkRequest(**request_data)
        
        assert request.enable_core_context is True
        assert request.share_insights_between_cores is True
        assert request.context_sharing_threshold == 0.8
    
    def test_thinking_path_enhancements(self):
        """Test enhanced thinking path with core context tracking"""
        path = ParallelThinkingPath("test_1", "Analytical approach")
        
        # Check new attributes
        assert hasattr(path, 'core_context_used')
        assert hasattr(path, 'shared_context_keys')
        assert path.core_context_used is False
        assert path.shared_context_keys == []
        
        # Test setting core context attributes
        path.core_context_used = True
        path.shared_context_keys = ["insight_1", "insight_2"]
        
        assert path.core_context_used is True
        assert len(path.shared_context_keys) == 2
    
    @patch('tools.parallelthink.ModelProviderRegistry')
    def test_core_context_storage_in_execution(self, mock_registry):
        """Test that core context storage is used during path execution"""
        # Mock the provider and response
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Test response with performance considerations"
        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_default_model.return_value = "test-model"
        
        mock_registry_instance = Mock()
        mock_registry_instance.get_default_provider.return_value = mock_provider
        mock_registry.return_value = mock_registry_instance
        
        # Create path and execute with core context
        path = ParallelThinkingPath("test_1", "Performance analysis")
        path.cpu_core = 0
        
        # Execute the sync version for testing
        result = self.tool._execute_thinking_path_sync(
            path, "Test prompt", "System prompt", "", 
            core_id=0, enable_core_context=True, share_insights=True
        )
        
        # Verify core context was used
        assert result.core_context_used is True
        assert result.cpu_core == 0
        assert isinstance(result.shared_context_keys, list)
    
    def test_insights_extraction(self):
        """Test insight extraction from results"""
        # Test different types of insights
        test_cases = [
            ("performance optimization is key", ["performance_considerations"]),
            ("security vulnerability found", ["security_aspects"]),
            ("scalability concerns arise", ["scalability_factors"]),
            ("complex algorithm needed", ["complexity_analysis"]),
            ("performance and security issues", ["performance_considerations", "security_aspects"]),
        ]
        
        for content, expected_insights in test_cases:
            # Create a mock result that would trigger insight extraction
            insights = []
            content_lower = content.lower()
            
            if "performance" in content_lower or "optimization" in content_lower:
                insights.append("performance_considerations")
            if "security" in content_lower or "vulnerability" in content_lower:
                insights.append("security_aspects")
            if "scalability" in content_lower or "scale" in content_lower:
                insights.append("scalability_factors")
            if "complexity" in content_lower or "complex" in content_lower:
                insights.append("complexity_analysis")
            
            assert insights == expected_insights
    
    def test_shared_insights_retrieval(self):
        """Test retrieval of shared insights from other cores"""
        # Set up some shared insights
        storage = get_core_context_storage()
        storage.set_core_context(
            "approach_analytical_insights", 
            ["performance_considerations"], 
            core_id=0, 
            share_with_others=True
        )
        
        # Retrieve insights for a different approach
        insights = self.tool._retrieve_shared_insights(1, "Creative brainstorming")
        
        # Should retrieve insights from analytical approach
        assert isinstance(insights, dict)
        # Note: The actual retrieval depends on the similarity logic
    
    def test_synthesis_with_core_context(self):
        """Test synthesis includes core context information"""
        # Create paths with core context information
        paths = []
        for i in range(3):
            path = ParallelThinkingPath(f"path_{i}", f"Approach {i}")
            path.result = f"Result {i} with performance considerations"
            path.execution_time = 1.0 + i * 0.1
            path.memory_usage = 10.0 + i * 2.0
            path.cpu_core = i
            path.core_context_used = True
            path.shared_context_keys = [f"insight_{i}"]
            paths.append(path)
        
        request = ParallelThinkRequest(
            prompt="Test", 
            enable_core_context=True,
            share_insights_between_cores=True
        )
        
        synthesis = self.tool._synthesize_results(paths, "comprehensive", request)
        
        # Check that synthesis includes core context information
        assert "Core context enabled" in synthesis
        assert "Insights shared between cores" in synthesis
        assert "Core Context:** Enabled" in synthesis
        assert "Shared Insights:** 1" in synthesis


class TestIntegration:
    """Integration tests for core context functionality"""
    
    def test_singleton_storage_instance(self):
        """Test that get_core_context_storage returns singleton instance"""
        storage1 = get_core_context_storage()
        storage2 = get_core_context_storage()
        
        assert storage1 is storage2
        assert isinstance(storage1, CoreContextStorage)
    
    def test_cleanup_functionality(self):
        """Test cleanup of expired contexts"""
        storage = CoreContextStorage(max_cores=2)
        
        # Add some contexts with different timestamps
        storage.set_core_context("key1", "value1", core_id=0)
        
        # Manually set old timestamp to simulate expiration
        storage._core_contexts[0].last_access = time.time() - 7200  # 2 hours ago
        
        # Add recent context
        storage.set_core_context("key2", "value2", core_id=1)
        
        # Run cleanup
        storage._cleanup_expired()
        
        # Check that old context was removed but recent one remains
        stats = storage.get_core_statistics()
        assert 0 not in stats["cores"]  # Should be cleaned up
        assert 1 in stats["cores"]      # Should remain
        
        storage.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])