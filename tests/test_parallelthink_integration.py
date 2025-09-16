"""
Integration test for parallel thinking tool execution
"""

from unittest.mock import Mock, patch

import pytest

from tools.parallelthink import ParallelThinkTool


class TestParallelThinkIntegration:
    """Integration tests for parallel thinking tool execution"""

    @pytest.mark.asyncio
    async def test_execute_with_mocked_providers(self):
        """Test parallel thinking execution with mocked providers"""
        tool = ParallelThinkTool()

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()

        # Mock the model response
        mock_response = Mock()
        mock_response.content = "Mock AI analysis result for this thinking path"

        # Mock the provider methods
        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_default_model.return_value = "mock-model"

        mock_registry.get_default_provider.return_value = mock_provider
        mock_registry.get_provider_for_model.return_value = mock_provider

        with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
            # Test basic parallel thinking execution
            result = await tool.execute(
                {
                    "prompt": "What's the best approach to solve this problem?",
                    "thinking_paths": 2,
                    "approach_diversity": True,
                    "model_diversity": False,
                    "synthesis_style": "comprehensive",
                }
            )

            # Verify the result structure
            assert len(result) == 1
            response = result[0]

            assert "parallel_thinking_analysis" in response
            assert "execution_summary" in response

            summary = response["execution_summary"]
            assert summary["total_paths"] == 2
            assert summary["successful_paths"] == 2  # Both should succeed with mock
            assert len(summary["approaches_used"]) == 2
            assert summary["synthesis_style"] == "comprehensive"

            # Verify provider was called for each path
            assert mock_provider.generate_content.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_hypothesis_testing(self):
        """Test parallel thinking with hypothesis testing mode"""
        tool = ParallelThinkTool()

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()

        # Mock the model response
        mock_response = Mock()
        mock_response.content = "Hypothesis analysis result"

        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_default_model.return_value = "mock-model"
        mock_registry.get_default_provider.return_value = mock_provider

        with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
            result = await tool.execute(
                {
                    "prompt": "Is this the right approach?",
                    "thinking_paths": 3,
                    "hypothesis_testing": True,
                    "synthesis_style": "consensus",
                }
            )

            # Verify the result
            assert len(result) == 1
            response = result[0]

            summary = response["execution_summary"]
            assert summary["total_paths"] == 3
            assert summary["synthesis_style"] == "consensus"

            # Should have called provider for each hypothesis
            assert mock_provider.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_model_diversity(self):
        """Test parallel thinking with multiple models"""
        tool = ParallelThinkTool()

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()

        # Mock different model responses
        mock_response1 = Mock()
        mock_response1.content = "Analysis from model 1"
        mock_response2 = Mock()
        mock_response2.content = "Analysis from model 2"

        # Alternate responses for different calls
        mock_provider.generate_content.side_effect = [mock_response1, mock_response2]
        mock_provider.get_default_model.return_value = "default-model"
        mock_registry.get_default_provider.return_value = mock_provider
        mock_registry.get_provider_for_model.return_value = mock_provider

        # Mock available models
        with patch.object(tool, "_get_available_models", return_value=["model1", "model2"]):
            with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
                result = await tool.execute(
                    {
                        "prompt": "Analyze this problem",
                        "thinking_paths": 2,
                        "model_diversity": True,
                        "include_individual_paths": True,
                    }
                )

                # Verify the result
                assert len(result) == 1
                response = result[0]

                assert "individual_paths" in response
                assert len(response["individual_paths"]) == 2

                # Check individual path results
                paths = response["individual_paths"]
                assert paths[0]["success"] is True
                assert paths[1]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_partial_failures(self):
        """Test parallel thinking when some paths fail"""
        tool = ParallelThinkTool()

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()

        # Mock one success and one failure
        mock_success_response = Mock()
        mock_success_response.content = "Successful analysis"

        # First call succeeds, second raises exception
        mock_provider.generate_content.side_effect = [mock_success_response, Exception("API Error")]
        mock_provider.get_default_model.return_value = "mock-model"
        mock_registry.get_default_provider.return_value = mock_provider

        with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
            result = await tool.execute(
                {"prompt": "Test prompt", "thinking_paths": 2, "include_individual_paths": True}
            )

            # Should still return a result even with partial failure
            assert len(result) == 1
            response = result[0]

            summary = response["execution_summary"]
            assert summary["total_paths"] == 2
            assert summary["successful_paths"] == 1  # Only one succeeded

            # Check individual paths show the failure
            paths = response["individual_paths"]
            assert len(paths) == 2

            # One should be successful, one should have error
            success_count = sum(1 for p in paths if p["success"])
            assert success_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_files(self):
        """Test parallel thinking with file input"""
        tool = ParallelThinkTool()

        # Mock file reading
        mock_file_data = [
            {"path": "/test/file1.py", "content": "print('hello')"},
            {"path": "/test/file2.py", "content": "def test(): pass"},
        ]

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()
        mock_response = Mock()
        mock_response.content = "Analysis with file context"
        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_default_model.return_value = "mock-model"
        mock_registry.get_default_provider.return_value = mock_provider

        with patch("tools.parallelthink.read_files", return_value=mock_file_data):
            with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
                result = await tool.execute(
                    {
                        "prompt": "Analyze these files",
                        "files": ["/test/file1.py", "/test/file2.py"],
                        "thinking_paths": 2,
                    }
                )

                # Should include file context in the analysis
                assert len(result) == 1
                response = result[0]
                assert "parallel_thinking_analysis" in response

                # Verify that generate_content was called with file context
                call_args = mock_provider.generate_content.call_args_list
                assert len(call_args) == 2  # Two thinking paths

                # Check that file content was included in prompts
                for call in call_args:
                    prompt_arg = call[1]["prompt"]  # keyword argument
                    assert "file1.py" in prompt_arg
                    assert "print('hello')" in prompt_arg

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test parallel thinking with timeout settings"""
        tool = ParallelThinkTool()

        # Mock the ModelProviderRegistry
        mock_registry = Mock()
        mock_provider = Mock()

        # Mock a response that returns directly (not async)
        mock_response = Mock()
        mock_response.content = "Analysis result"

        mock_provider.generate_content.return_value = mock_response
        mock_provider.get_default_model.return_value = "mock-model"
        mock_registry.get_default_provider.return_value = mock_provider

        with patch("tools.parallelthink.ModelProviderRegistry", return_value=mock_registry):
            result = await tool.execute(
                {"prompt": "Test with timeout", "thinking_paths": 2, "time_limit": 10}  # Should be enough time
            )

            # Should complete successfully
            assert len(result) == 1
            response = result[0]

            summary = response["execution_summary"]
            assert summary["successful_paths"] == 2

    @pytest.mark.asyncio
    async def test_execute_error_handling(self):
        """Test error handling in execute method"""
        tool = ParallelThinkTool()

        # Test with invalid arguments
        result = await tool.execute({"prompt": "Test", "thinking_paths": 10})  # Invalid - too many paths

        # Should return error response
        assert len(result) == 1
        response = result[0]
        assert "error" in response
        assert "tool" in response
        assert response["tool"] == "parallelthink"
