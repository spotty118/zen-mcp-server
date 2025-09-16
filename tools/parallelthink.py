"""
Parallel Thinking Tool - Concurrent Multi-Path Reasoning

This tool enables parallel thinking by running multiple reasoning processes concurrently.
It can explore different approaches, test multiple hypotheses, or gather insights from
different AI models simultaneously, then synthesize the results.

Key Features:
- Concurrent execution of multiple thinking paths
- Multi-model parallel reasoning for consensus building
- Parallel hypothesis testing and validation
- Intelligent synthesis of diverse perspectives
- Token-efficient parallel processing
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_CREATIVE
from providers import ModelProviderRegistry
from systemprompts import PARALLELTHINK_PROMPT
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool
from utils.file_utils import read_files

logger = logging.getLogger(__name__)


class ParallelThinkingPath:
    """Represents a single thinking path in parallel processing"""

    def __init__(
        self, path_id: str, approach: str, model: Optional[str] = None, prompt_variation: Optional[str] = None
    ):
        self.path_id = path_id
        self.approach = approach
        self.model = model
        self.prompt_variation = prompt_variation
        self.result = None
        self.error = None
        self.execution_time = 0.0


class ParallelThinkRequest(ToolRequest):
    """Request model for parallel thinking tool"""

    # Core thinking parameters
    prompt: str = Field(description="The main problem or question to think about in parallel")
    thinking_paths: int = Field(
        default=3, description="Number of parallel thinking paths to execute (2-6 recommended)", ge=2, le=8
    )

    # File input support
    files: Optional[list[str]] = Field(
        default=None,
        description="Optional files for context (must be FULL absolute paths to real files / folders - DO NOT SHORTEN)",
    )

    # Parallel thinking strategies
    approach_diversity: bool = Field(
        default=True, description="Use different thinking approaches (analytical, creative, systematic, etc.)"
    )
    model_diversity: bool = Field(
        default=False,
        description="Use different AI models for multi-perspective reasoning (requires multiple providers)",
    )
    hypothesis_testing: bool = Field(default=False, description="Generate and test multiple hypotheses in parallel")

    # Context and constraints
    focus_areas: Optional[list[str]] = Field(
        default=None, description="Specific aspects to focus on (architecture, performance, security, etc.)"
    )
    time_limit: Optional[int] = Field(
        default=60, description="Maximum execution time in seconds for all parallel paths", ge=10, le=300
    )

    # Synthesis options
    synthesis_style: str = Field(
        default="comprehensive",
        description="How to synthesize results: 'comprehensive', 'consensus', 'diverse', 'best_path'",
    )
    include_individual_paths: bool = Field(
        default=True, description="Include individual path results in addition to synthesis"
    )


class ParallelThinkTool(BaseTool):
    """
    Parallel Thinking Tool - Concurrent Multi-Path Reasoning

    Enables concurrent thinking by running multiple reasoning processes simultaneously.
    Supports different approaches, models, and synthesis strategies.
    """

    name = "parallelthink"
    description = (
        "PARALLEL MULTI-PATH REASONING - Execute multiple thinking processes concurrently to explore "
        "different approaches, test hypotheses, or gather diverse perspectives. Perfect for: complex "
        "problem-solving, architectural decisions, exploring trade-offs, consensus building, or when "
        "you want multiple angles on a challenging question. Synthesizes insights from parallel paths "
        "into comprehensive analysis. Choose 2-6 thinking paths based on problem complexity."
    )

    def get_name(self) -> str:
        """Return the tool name"""
        return self.name

    def get_description(self) -> str:
        """Return the tool description"""
        return self.description

    def get_model_category(self) -> "ToolModelCategory":
        """Return the model category for this tool"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for parallel thinking tool"""
        return ParallelThinkRequest.model_json_schema()

    def get_system_prompt(self) -> str:
        """Return the system prompt for this tool"""
        return PARALLELTHINK_PROMPT

    def get_default_temperature(self) -> float:
        """Return default temperature for parallel thinking"""
        return TEMPERATURE_CREATIVE

    def get_default_thinking_mode(self) -> str:
        """Return default thinking mode for parallel thinking"""
        return "high"  # Parallel thinking benefits from deeper reasoning

    def get_request_model(self):
        """Return the request model for parallel thinking"""
        return ParallelThinkRequest

    async def prepare_prompt(self, request: ParallelThinkRequest) -> str:
        """Prepare the prompt for parallel thinking - not used in execute()"""
        return request.prompt

    def _generate_thinking_approaches(self, count: int, focus_areas: Optional[list[str]] = None) -> list[str]:
        """Generate diverse thinking approaches for parallel execution"""
        base_approaches = [
            "Systematic analytical breakdown with logical reasoning",
            "Creative brainstorming with out-of-the-box thinking",
            "Risk-focused analysis identifying potential problems",
            "Solution-oriented approach focusing on practical implementation",
            "Historical perspective using past experiences and patterns",
            "Future-focused thinking considering long-term implications",
            "User-centered approach prioritizing human needs",
            "Technical deep-dive focusing on implementation details",
        ]

        if focus_areas:
            # Customize approaches based on focus areas
            focused_approaches = []
            for area in focus_areas[:count]:
                focused_approaches.append(f"{area.title()}-focused analysis with specialized expertise")
            base_approaches = focused_approaches + base_approaches

        return base_approaches[:count]

    def _generate_hypothesis_prompts(self, base_prompt: str, count: int) -> list[str]:
        """Generate different hypothesis-testing prompts"""
        hypothesis_frames = [
            f"Hypothesis: This problem is primarily about efficiency. {base_prompt}",
            f"Hypothesis: This problem is primarily about complexity. {base_prompt}",
            f"Hypothesis: This problem is primarily about scalability. {base_prompt}",
            f"Hypothesis: This problem is primarily about maintainability. {base_prompt}",
            f"Hypothesis: This problem is primarily about user experience. {base_prompt}",
            f"Hypothesis: This problem is primarily about technical debt. {base_prompt}",
        ]
        return hypothesis_frames[:count]

    def _get_available_models(self) -> list[str]:
        """Get list of available models from different providers"""
        registry = ModelProviderRegistry()
        available_models = []

        try:
            # Try to get models from different providers
            for provider_type in ["openai", "gemini", "xai", "openrouter"]:
                try:
                    provider = registry.get_provider(provider_type)
                    if provider:
                        # Get a representative model from this provider
                        models = provider.get_model_configurations()
                        if models:
                            # Get first available model
                            model_name = next(iter(models.keys()))
                            available_models.append(model_name)
                except Exception as e:
                    logger.debug(f"Could not get models from {provider_type}: {e}")

        except Exception as e:
            logger.debug(f"Error getting available models: {e}")

        return available_models[:4]  # Limit to 4 different models

    async def _execute_thinking_path(
        self, path: ParallelThinkingPath, prompt: str, system_prompt: str, files_content: str
    ) -> ParallelThinkingPath:
        """Execute a single thinking path asynchronously"""
        import time

        start_time = time.time()

        try:
            # Prepare the full prompt for this path
            if path.prompt_variation:
                full_prompt = path.prompt_variation
            else:
                approach_instruction = f"\n\nTHINKING APPROACH: {path.approach}\n"
                full_prompt = approach_instruction + prompt

            if files_content:
                full_prompt = f"{full_prompt}\n\nFILE CONTEXT:\n{files_content}"

            # Get provider and model
            registry = ModelProviderRegistry()
            if path.model:
                # Try to use specified model
                try:
                    provider = registry.get_provider_for_model(path.model)
                    model_name = path.model
                except Exception as e:
                    logger.warning(f"Could not use model {path.model}: {e}, falling back to default")
                    provider = registry.get_default_provider()
                    model_name = provider.get_default_model()
            else:
                provider = registry.get_default_provider()
                model_name = provider.get_default_model()

            # Execute the thinking
            response = provider.generate_content(
                prompt=full_prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=self.get_default_temperature(),
                thinking_mode=self.get_default_thinking_mode(),
            )

            path.result = response.content
            path.execution_time = time.time() - start_time

        except Exception as e:
            path.error = str(e)
            path.execution_time = time.time() - start_time
            logger.error(f"Error in thinking path {path.path_id}: {e}")

        return path

    def _synthesize_results(
        self, paths: list[ParallelThinkingPath], synthesis_style: str, request: ParallelThinkRequest
    ) -> str:
        """Synthesize results from multiple thinking paths"""
        successful_paths = [p for p in paths if p.result and not p.error]

        if not successful_paths:
            return "⚠️  All parallel thinking paths failed. Please check the logs for details."

        if synthesis_style == "best_path":
            # Return the result from the fastest successful path
            best_path = min(successful_paths, key=lambda p: p.execution_time)
            return f"**Best Path Result** (from {best_path.approach}):\n\n{best_path.result}"

        elif synthesis_style == "consensus":
            # Focus on common themes and agreements
            synthesis = "## Consensus Analysis\n\n"
            synthesis += f"Analyzed {len(successful_paths)} parallel thinking paths.\n\n"
            synthesis += "**Common Themes and Agreements:**\n"
            # Simple consensus detection (could be enhanced with NLP)
            all_results = " ".join([p.result for p in successful_paths])
            if "performance" in all_results.lower():
                synthesis += "- Performance considerations appear in multiple analyses\n"
            if "security" in all_results.lower():
                synthesis += "- Security aspects identified across thinking paths\n"
            if "scalability" in all_results.lower():
                synthesis += "- Scalability concerns noted in multiple approaches\n"

        elif synthesis_style == "diverse":
            # Highlight the diversity of perspectives
            synthesis = "## Diverse Perspectives Analysis\n\n"
            for i, path in enumerate(successful_paths, 1):
                synthesis += f"### Perspective {i}: {path.approach}\n"
                synthesis += f"{path.result[:200]}...\n\n"

        else:  # comprehensive
            synthesis = "## Comprehensive Parallel Thinking Analysis\n\n"
            synthesis += (
                f"Executed {len(paths)} parallel thinking paths in {max(p.execution_time for p in paths):.1f}s\n"
            )
            synthesis += f"Successful paths: {len(successful_paths)}/{len(paths)}\n\n"

            for i, path in enumerate(successful_paths, 1):
                synthesis += f"### Path {i}: {path.approach}\n"
                if path.model:
                    synthesis += f"**Model:** {path.model}\n"
                synthesis += f"**Execution time:** {path.execution_time:.1f}s\n\n"
                synthesis += f"{path.result}\n\n---\n\n"

        return synthesis

    async def execute(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute parallel thinking with multiple concurrent paths"""
        try:
            # Validate and parse request
            request = ParallelThinkRequest(**arguments)

            # Read files if provided
            files_content = ""
            if request.files:
                try:
                    files_data = read_files(request.files)
                    files_content = "\n\n".join(
                        [f"=== {file_info['path']} ===\n{file_info['content']}" for file_info in files_data]
                    )
                except Exception as e:
                    logger.warning(f"Could not read files: {e}")

            # Create thinking paths
            paths = []

            if request.hypothesis_testing:
                # Generate hypothesis-testing prompts
                hypothesis_prompts = self._generate_hypothesis_prompts(request.prompt, request.thinking_paths)
                for i, hypothesis_prompt in enumerate(hypothesis_prompts):
                    path = ParallelThinkingPath(
                        path_id=f"hypothesis_{i+1}",
                        approach=f"Hypothesis Testing {i+1}",
                        prompt_variation=hypothesis_prompt,
                    )
                    paths.append(path)
            else:
                # Generate diverse approaches
                approaches = self._generate_thinking_approaches(request.thinking_paths, request.focus_areas)

                if request.model_diversity:
                    # Use different models for different paths
                    available_models = self._get_available_models()
                    for i, approach in enumerate(approaches):
                        model = available_models[i % len(available_models)] if available_models else None
                        path = ParallelThinkingPath(path_id=f"path_{i+1}", approach=approach, model=model)
                        paths.append(path)
                else:
                    # Use same model with different approaches
                    for i, approach in enumerate(approaches):
                        path = ParallelThinkingPath(path_id=f"path_{i+1}", approach=approach)
                        paths.append(path)

            # Execute all thinking paths concurrently
            system_prompt = self.get_system_prompt()

            # Create tasks for concurrent execution
            tasks = [self._execute_thinking_path(path, request.prompt, system_prompt, files_content) for path in paths]

            # Execute with timeout
            try:
                completed_paths = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=request.time_limit or 60
                )
            except asyncio.TimeoutError:
                logger.warning(f"Parallel thinking timed out after {request.time_limit}s")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Use partial results
                completed_paths = [p for p in paths if p.result or p.error]

            # Synthesize results
            synthesis = self._synthesize_results(completed_paths, request.synthesis_style, request)

            # Prepare response
            response = {
                "parallel_thinking_analysis": synthesis,
                "execution_summary": {
                    "total_paths": len(paths),
                    "successful_paths": len(
                        [
                            p
                            for p in completed_paths
                            if hasattr(p, "result") and p.result and not getattr(p, "error", None)
                        ]
                    ),
                    "approaches_used": [p.approach for p in paths],
                    "models_used": list({p.model for p in paths if p.model}) or ["default"],
                    "total_execution_time": (
                        max([getattr(p, "execution_time", 0) for p in completed_paths]) if completed_paths else 0
                    ),
                    "synthesis_style": request.synthesis_style,
                },
            }

            # Include individual path results if requested
            if request.include_individual_paths:
                response["individual_paths"] = []
                for path in completed_paths:
                    if hasattr(path, "result"):
                        path_result = {
                            "path_id": path.path_id,
                            "approach": path.approach,
                            "execution_time": getattr(path, "execution_time", 0),
                            "success": bool(path.result and not getattr(path, "error", None)),
                        }
                        if path.model:
                            path_result["model"] = path.model
                        if path.result:
                            path_result["result"] = path.result
                        if getattr(path, "error", None):
                            path_result["error"] = path.error
                        response["individual_paths"].append(path_result)

            return [response]

        except Exception as e:
            logger.error(f"Error in parallel thinking execution: {e}")
            return [{"error": f"Parallel thinking failed: {str(e)}", "tool": self.name}]
