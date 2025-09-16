"""
Tests for enhanced parallel thinking tool with CPU optimization
"""

import os
import threading
import unittest

from tools.parallelthink import ParallelThinkingPath, ParallelThinkRequest, ParallelThinkTool


class TestParallelThinkTool(unittest.TestCase):
    """Test the enhanced ParallelThinkTool with CPU optimization"""

    def test_tool_initialization(self):
        """Test that the tool initializes correctly"""
        tool = ParallelThinkTool()
        self.assertEqual(tool.get_name(), "parallelthink")
        self.assertIn("parallel", tool.get_description().lower())
        self.assertIn("concurrent", tool.get_description().lower())

    def test_tool_schema(self):
        """Test that the tool has a valid input schema"""
        tool = ParallelThinkTool()
        schema = tool.get_input_schema()

        # Check required fields exist
        self.assertIn("properties", schema)
        properties = schema["properties"]

        self.assertIn("prompt", properties)
        self.assertIn("thinking_paths", properties)
        self.assertIn("approach_diversity", properties)
        self.assertIn("model_diversity", properties)
        self.assertIn("hypothesis_testing", properties)
        self.assertIn("cpu_cores", properties)
        self.assertIn("execution_strategy", properties)
        self.assertIn("enable_cpu_affinity", properties)
        self.assertIn("batch_size", properties)

    def test_request_model_validation(self):
        """Test that the request model validates correctly"""
        # Valid request
        request = ParallelThinkRequest(
            prompt="Test prompt", thinking_paths=3, approach_diversity=True, model_diversity=False
        )
        self.assertEqual(request.prompt, "Test prompt")
        self.assertEqual(request.thinking_paths, 3)
        self.assertTrue(request.approach_diversity)
        self.assertFalse(request.model_diversity)

    def test_request_model_constraints(self):
        """Test that the request model enforces constraints"""
        # Test thinking_paths constraint (min 2, max 8)
        with self.assertRaises(ValueError):
            ParallelThinkRequest(prompt="Test", thinking_paths=1)  # Too low

        with self.assertRaises(ValueError):
            ParallelThinkRequest(prompt="Test", thinking_paths=9)  # Too high

        # Valid boundaries
        request_min = ParallelThinkRequest(prompt="Test", thinking_paths=2)
        self.assertEqual(request_min.thinking_paths, 2)

        request_max = ParallelThinkRequest(prompt="Test", thinking_paths=8)
        self.assertEqual(request_max.thinking_paths, 8)

    def test_thinking_path_creation(self):
        """Test creation of thinking paths"""
        path = ParallelThinkingPath(
            path_id="test_1", approach="Analytical approach", model="test-model", prompt_variation="Custom prompt"
        )

        self.assertEqual(path.path_id, "test_1")
        self.assertEqual(path.approach, "Analytical approach")
        self.assertEqual(path.model, "test-model")
        self.assertEqual(path.prompt_variation, "Custom prompt")
        self.assertIsNone(path.result)
        self.assertIsNone(path.error)
        self.assertEqual(path.execution_time, 0.0)
        self.assertIsNone(path.cpu_core)
        self.assertIsNone(path.thread_id)
        self.assertEqual(path.memory_usage, 0.0)

    def test_generate_thinking_approaches(self):
        """Test generation of diverse thinking approaches"""
        tool = ParallelThinkTool()

        # Test basic approach generation
        approaches = tool._generate_thinking_approaches(3)
        self.assertEqual(len(approaches), 3)
        self.assertTrue(all(isinstance(approach, str) for approach in approaches))
        self.assertTrue(all(len(approach) > 10 for approach in approaches))  # Should be meaningful descriptions

    def test_generate_thinking_approaches_with_focus(self):
        """Test generation of approaches with focus areas"""
        tool = ParallelThinkTool()

        focus_areas = ["security", "performance"]
        approaches = tool._generate_thinking_approaches(3, focus_areas)
        self.assertEqual(len(approaches), 3)

        # First approaches should be focused on the specified areas
        self.assertTrue("security" in approaches[0].lower() or "Security" in approaches[0])
        self.assertTrue("performance" in approaches[1].lower() or "Performance" in approaches[1])

    def test_generate_hypothesis_prompts(self):
        """Test generation of hypothesis-testing prompts"""
        tool = ParallelThinkTool()

        base_prompt = "What is the best approach?"
        hypothesis_prompts = tool._generate_hypothesis_prompts(base_prompt, 3)

        self.assertEqual(len(hypothesis_prompts), 3)
        self.assertTrue(all(base_prompt in prompt for prompt in hypothesis_prompts))
        self.assertTrue(all("Hypothesis:" in prompt for prompt in hypothesis_prompts))

    def test_get_available_models(self):
        """Test getting available models"""
        tool = ParallelThinkTool()

        # This should not raise an error even if no providers are configured
        models = tool._get_available_models()
        self.assertIsInstance(models, list)
        # In test environment, might be empty, which is fine

    def test_synthesize_results_best_path(self):
        """Test synthesis of results using best_path strategy"""
        tool = ParallelThinkTool()

        # Create test paths
        path1 = ParallelThinkingPath("path1", "Approach 1")
        path1.result = "Result 1"
        path1.execution_time = 2.0

        path2 = ParallelThinkingPath("path2", "Approach 2")
        path2.result = "Result 2"
        path2.execution_time = 1.0  # Faster

        paths = [path1, path2]
        request = ParallelThinkRequest(prompt="Test", synthesis_style="best_path")

        synthesis = tool._synthesize_results(paths, "best_path", request)

        self.assertIn("Result 2", synthesis)  # Should pick faster path
        self.assertIn("Approach 2", synthesis)

    def test_synthesize_results_comprehensive(self):
        """Test synthesis of results using comprehensive strategy"""
        tool = ParallelThinkTool()

        # Create test paths
        path1 = ParallelThinkingPath("path1", "Approach 1")
        path1.result = "Result 1"
        path1.execution_time = 1.0

        path2 = ParallelThinkingPath("path2", "Approach 2")
        path2.result = "Result 2"
        path2.execution_time = 2.0

        paths = [path1, path2]
        request = ParallelThinkRequest(prompt="Test", synthesis_style="comprehensive")

        synthesis = tool._synthesize_results(paths, "comprehensive", request)

        self.assertIn("Result 1", synthesis)
        self.assertIn("Result 2", synthesis)
        self.assertIn("Approach 1", synthesis)
        self.assertIn("Approach 2", synthesis)

    def test_synthesize_results_with_errors(self):
        """Test synthesis when some paths have errors"""
        tool = ParallelThinkTool()

        # Create test paths with one error
        path1 = ParallelThinkingPath("path1", "Approach 1")
        path1.result = "Result 1"
        path1.execution_time = 1.0

        path2 = ParallelThinkingPath("path2", "Approach 2")
        path2.error = "Test error"
        path2.execution_time = 2.0

        paths = [path1, path2]
        request = ParallelThinkRequest(prompt="Test", synthesis_style="comprehensive")

        synthesis = tool._synthesize_results(paths, "comprehensive", request)

        # Should only include successful path
        self.assertIn("Result 1", synthesis)
        self.assertNotIn("Result 2", synthesis)
        self.assertIn("Successful: 1", synthesis)  # Updated to match new format

    def test_synthesize_results_all_failed(self):
        """Test synthesis when all paths fail"""
        tool = ParallelThinkTool()

        # Create test paths with all errors
        path1 = ParallelThinkingPath("path1", "Approach 1")
        path1.error = "Error 1"

        path2 = ParallelThinkingPath("path2", "Approach 2")
        path2.error = "Error 2"

        paths = [path1, path2]
        request = ParallelThinkRequest(prompt="Test", synthesis_style="comprehensive")

        synthesis = tool._synthesize_results(paths, "comprehensive", request)

        self.assertIn("failed", synthesis.lower())

    def test_default_values(self):
        """Test default values in request model"""
        request = ParallelThinkRequest(prompt="Test prompt")

        self.assertEqual(request.thinking_paths, 3)  # Default
        self.assertTrue(request.approach_diversity)  # Default
        self.assertFalse(request.model_diversity)  # Default
        self.assertFalse(request.hypothesis_testing)  # Default
        self.assertEqual(request.time_limit, 60)  # Default
        self.assertEqual(request.synthesis_style, "comprehensive")  # Default
        self.assertTrue(request.include_individual_paths)  # Default

    def test_model_category(self):
        """Test that the tool returns the correct model category"""
        tool = ParallelThinkTool()
        from tools.models import ToolModelCategory

        category = tool.get_model_category()
        self.assertEqual(category, ToolModelCategory.EXTENDED_REASONING)

    def test_system_prompt(self):
        """Test that the tool has a system prompt"""
        tool = ParallelThinkTool()
        prompt = tool.get_system_prompt()

        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)  # Should be substantial
        self.assertIn("parallel", prompt.lower())

    def test_default_temperature(self):
        """Test default temperature setting"""
        tool = ParallelThinkTool()
        temp = tool.get_default_temperature()

        self.assertIsInstance(temp, float)
        self.assertGreaterEqual(temp, 0.0)
        self.assertLessEqual(temp, 1.0)

    def test_default_thinking_mode(self):
        """Test default thinking mode"""
        tool = ParallelThinkTool()
        mode = tool.get_default_thinking_mode()

        self.assertEqual(mode, "high")

    def test_cpu_core_detection(self):
        """Test CPU core detection and optimization"""
        tool = ParallelThinkTool()

        # Test auto-detection
        cores = tool._get_optimal_cpu_cores()
        self.assertIsInstance(cores, int)
        self.assertGreater(cores, 0)
        self.assertLessEqual(cores, os.cpu_count() or 4)

        # Test with explicit values
        cores_2 = tool._get_optimal_cpu_cores(2)
        self.assertEqual(cores_2, 2)

        # Test capping at available cores
        cores_high = tool._get_optimal_cpu_cores(100)
        self.assertLessEqual(cores_high, os.cpu_count() or 4)

    def test_batch_size_calculation(self):
        """Test batch size optimization"""
        tool = ParallelThinkTool()

        # Test various combinations
        batch_2_2 = tool._get_optimal_batch_size(2, 2)
        self.assertEqual(batch_2_2, 1)

        batch_8_4 = tool._get_optimal_batch_size(8, 4)
        self.assertGreaterEqual(batch_8_4, 1)
        self.assertLessEqual(batch_8_4, 3)

        batch_12_4 = tool._get_optimal_batch_size(12, 4)
        self.assertGreaterEqual(batch_12_4, 1)
        self.assertLessEqual(batch_12_4, 3)

    def test_execution_strategy_selection(self):
        """Test execution strategy selection logic"""
        tool = ParallelThinkTool()

        # Test specific strategies pass through
        self.assertEqual(tool._choose_execution_strategy("asyncio", 4, 4), "asyncio")
        self.assertEqual(tool._choose_execution_strategy("threads", 4, 4), "threads")
        self.assertEqual(tool._choose_execution_strategy("hybrid", 4, 4), "hybrid")

        # Test adaptive strategy logic
        strategy_small = tool._choose_execution_strategy("adaptive", 2, 4)
        self.assertEqual(strategy_small, "asyncio")

        strategy_large = tool._choose_execution_strategy("adaptive", 8, 4)
        self.assertIn(strategy_large, ["threads", "hybrid"])

    def test_memory_monitoring(self):
        """Test memory usage monitoring"""
        tool = ParallelThinkTool()

        memory = tool._get_memory_usage()
        self.assertIsInstance(memory, float)
        self.assertGreaterEqual(memory, 0)

    def test_cpu_affinity(self):
        """Test CPU affinity setting"""
        tool = ParallelThinkTool()

        # Test affinity setting (may not work on all systems)
        result = tool._set_cpu_affinity(0)
        self.assertIsInstance(result, bool)

    def test_enhanced_thinking_path(self):
        """Test enhanced thinking path functionality"""
        path = ParallelThinkingPath("test_1", "Enhanced approach")

        # Test initial state
        self.assertIsNone(path.cpu_core)
        self.assertIsNone(path.thread_id)
        self.assertEqual(path.memory_usage, 0.0)

        # Test setting enhanced attributes
        path.cpu_core = 2
        path.thread_id = threading.get_ident()
        path.memory_usage = 15.5

        self.assertEqual(path.cpu_core, 2)
        self.assertIsNotNone(path.thread_id)
        self.assertEqual(path.memory_usage, 15.5)

    def test_enhanced_request_model(self):
        """Test enhanced request model with new parameters"""
        request = ParallelThinkRequest(
            prompt="Enhanced test",
            thinking_paths=4,
            cpu_cores=2,
            execution_strategy="hybrid",
            enable_cpu_affinity=True,
            batch_size=2
        )

        self.assertEqual(request.cpu_cores, 2)
        self.assertEqual(request.execution_strategy, "hybrid")
        self.assertTrue(request.enable_cpu_affinity)
        self.assertEqual(request.batch_size, 2)

    def test_enhanced_default_values(self):
        """Test enhanced default values in request model"""
        request = ParallelThinkRequest(prompt="Test prompt")

        # Original defaults
        self.assertEqual(request.thinking_paths, 3)
        self.assertTrue(request.approach_diversity)
        self.assertFalse(request.model_diversity)
        self.assertFalse(request.hypothesis_testing)
        self.assertEqual(request.time_limit, 60)
        self.assertEqual(request.synthesis_style, "comprehensive")
        self.assertTrue(request.include_individual_paths)

        # Enhanced defaults
        self.assertIsNone(request.cpu_cores)
        self.assertEqual(request.execution_strategy, "adaptive")
        self.assertTrue(request.enable_cpu_affinity)
        self.assertIsNone(request.batch_size)

    def test_execute_basic_functionality(self):
        """Test basic execute functionality (without real API calls)"""
        tool = ParallelThinkTool()

        # Test that the method exists and has the right signature
        self.assertTrue(hasattr(tool, "execute"))
        self.assertTrue(callable(tool.execute))


if __name__ == "__main__":
    unittest.main()
