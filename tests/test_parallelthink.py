"""
Tests for parallel thinking tool
"""

import pytest

from tools.parallelthink import ParallelThinkingPath, ParallelThinkRequest, ParallelThinkTool


class TestParallelThinkTool:
    """Test the ParallelThinkTool"""

    def test_tool_initialization(self):
        """Test that the tool initializes correctly"""
        tool = ParallelThinkTool()
        assert tool.get_name() == "parallelthink"
        assert "parallel" in tool.get_description().lower()
        assert "concurrent" in tool.get_description().lower()

    def test_tool_schema(self):
        """Test that the tool has a valid input schema"""
        tool = ParallelThinkTool()
        schema = tool.get_input_schema()

        # Check required fields exist
        assert "properties" in schema
        properties = schema["properties"]

        assert "prompt" in properties
        assert "thinking_paths" in properties
        assert "approach_diversity" in properties
        assert "model_diversity" in properties
        assert "hypothesis_testing" in properties

    def test_request_model_validation(self):
        """Test that the request model validates correctly"""
        # Valid request
        request = ParallelThinkRequest(
            prompt="Test prompt", thinking_paths=3, approach_diversity=True, model_diversity=False
        )
        assert request.prompt == "Test prompt"
        assert request.thinking_paths == 3
        assert request.approach_diversity is True
        assert request.model_diversity is False

    def test_request_model_constraints(self):
        """Test that the request model enforces constraints"""
        # Test thinking_paths constraint (min 2, max 8)
        with pytest.raises(ValueError):
            ParallelThinkRequest(prompt="Test", thinking_paths=1)  # Too low

        with pytest.raises(ValueError):
            ParallelThinkRequest(prompt="Test", thinking_paths=9)  # Too high

        # Valid boundaries
        request_min = ParallelThinkRequest(prompt="Test", thinking_paths=2)
        assert request_min.thinking_paths == 2

        request_max = ParallelThinkRequest(prompt="Test", thinking_paths=8)
        assert request_max.thinking_paths == 8

    def test_thinking_path_creation(self):
        """Test creation of thinking paths"""
        path = ParallelThinkingPath(
            path_id="test_1", approach="Analytical approach", model="test-model", prompt_variation="Custom prompt"
        )

        assert path.path_id == "test_1"
        assert path.approach == "Analytical approach"
        assert path.model == "test-model"
        assert path.prompt_variation == "Custom prompt"
        assert path.result is None
        assert path.error is None
        assert path.execution_time == 0.0

    def test_generate_thinking_approaches(self):
        """Test generation of diverse thinking approaches"""
        tool = ParallelThinkTool()

        # Test basic approach generation
        approaches = tool._generate_thinking_approaches(3)
        assert len(approaches) == 3
        assert all(isinstance(approach, str) for approach in approaches)
        assert all(len(approach) > 10 for approach in approaches)  # Should be meaningful descriptions

    def test_generate_thinking_approaches_with_focus(self):
        """Test generation of approaches with focus areas"""
        tool = ParallelThinkTool()

        focus_areas = ["security", "performance"]
        approaches = tool._generate_thinking_approaches(3, focus_areas)
        assert len(approaches) == 3

        # First approaches should be focused on the specified areas
        assert "security" in approaches[0].lower() or "Security" in approaches[0]
        assert "performance" in approaches[1].lower() or "Performance" in approaches[1]

    def test_generate_hypothesis_prompts(self):
        """Test generation of hypothesis-testing prompts"""
        tool = ParallelThinkTool()

        base_prompt = "What is the best approach?"
        hypothesis_prompts = tool._generate_hypothesis_prompts(base_prompt, 3)

        assert len(hypothesis_prompts) == 3
        assert all(base_prompt in prompt for prompt in hypothesis_prompts)
        assert all("Hypothesis:" in prompt for prompt in hypothesis_prompts)

    def test_get_available_models(self):
        """Test getting available models"""
        tool = ParallelThinkTool()

        # This should not raise an error even if no providers are configured
        models = tool._get_available_models()
        assert isinstance(models, list)
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

        assert "Result 2" in synthesis  # Should pick faster path
        assert "Approach 2" in synthesis

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

        assert "Result 1" in synthesis
        assert "Result 2" in synthesis
        assert "Approach 1" in synthesis
        assert "Approach 2" in synthesis

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
        assert "Result 1" in synthesis
        assert "Result 2" not in synthesis
        assert "1/2" in synthesis  # Should show success ratio

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

        assert "failed" in synthesis.lower()

    def test_default_values(self):
        """Test default values in request model"""
        request = ParallelThinkRequest(prompt="Test prompt")

        assert request.thinking_paths == 3  # Default
        assert request.approach_diversity is True  # Default
        assert request.model_diversity is False  # Default
        assert request.hypothesis_testing is False  # Default
        assert request.time_limit == 60  # Default
        assert request.synthesis_style == "comprehensive"  # Default
        assert request.include_individual_paths is True  # Default

    def test_model_category(self):
        """Test that the tool returns the correct model category"""
        tool = ParallelThinkTool()
        from tools.models import ToolModelCategory

        category = tool.get_model_category()
        assert category == ToolModelCategory.EXTENDED_REASONING

    def test_system_prompt(self):
        """Test that the tool has a system prompt"""
        tool = ParallelThinkTool()
        prompt = tool.get_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert "parallel" in prompt.lower()

    def test_default_temperature(self):
        """Test default temperature setting"""
        tool = ParallelThinkTool()
        temp = tool.get_default_temperature()

        assert isinstance(temp, float)
        assert 0.0 <= temp <= 1.0

    def test_default_thinking_mode(self):
        """Test default thinking mode"""
        tool = ParallelThinkTool()
        mode = tool.get_default_thinking_mode()

        assert mode == "high"

    @pytest.mark.asyncio
    async def test_execute_basic_functionality(self):
        """Test basic execute functionality (without real API calls)"""
        tool = ParallelThinkTool()

        # This test would require mocking the provider registry and API calls
        # For now, we test that the method exists and has the right signature
        assert hasattr(tool, "execute")
        assert callable(tool.execute)
