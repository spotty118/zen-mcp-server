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
import concurrent.futures
import logging
import os
import platform
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from config import TEMPERATURE_CREATIVE
from providers import ModelProviderRegistry
from systemprompts import PARALLELTHINK_PROMPT
from tools.shared.base_models import ToolRequest
from tools.shared.base_tool import BaseTool
from utils.core_context_storage import get_core_context_storage
from utils.file_utils import read_files
from utils.agent_core import AgentRole, AgentStatus
from utils.agent_communication import get_agent_communication_system

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
        self.cpu_core = None  # Track which CPU core processed this path
        self.thread_id = None  # Track thread ID for debugging
        self.memory_usage = 0.0  # Track memory usage for optimization
        self.core_context_used = False  # Track if core-specific context was utilized
        self.shared_context_keys = []  # Keys that were shared with other cores
        self.assigned_agent = None  # ID of the agent assigned to this path


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

    # Parallel execution optimization
    cpu_cores: Optional[int] = Field(
        default=None,
        description="Number of CPU cores to use (auto-detected if not specified), max recommended is system cores",
    )
    execution_strategy: str = Field(
        default="adaptive",
        description="Execution strategy: 'asyncio' (I/O focus), 'threads' (CPU focus), 'adaptive' (smart hybrid)",
    )
    enable_cpu_affinity: bool = Field(
        default=True, description="Enable CPU affinity optimization for better performance"
    )
    batch_size: Optional[int] = Field(
        default=None, description="Batch size for processing (auto-calculated if not specified)"
    )

    # Core-specific context options
    enable_core_context: bool = Field(
        default=True, description="Enable core-specific context isolation and sharing"
    )
    share_insights_between_cores: bool = Field(
        default=True, description="Allow cores to share relevant insights and discoveries"
    )
    context_sharing_threshold: float = Field(
        default=0.7, description="Confidence threshold for sharing insights between cores (0.0-1.0)"
    )
    
    # Agent-based execution options
    enable_agent_mode: bool = Field(
        default=True, description="Enable agent-based execution where each core acts as an autonomous agent"
    )
    auto_select_agents: bool = Field(
        default=True, description="Automatically select optimal agents based on task characteristics and CPU cores"
    )
    agent_roles: Optional[list[str]] = Field(
        default=None, description="Specific agent roles to use (security_analyst, performance_optimizer, etc.) - ignored if auto_select_agents is True"
    )
    enable_agent_communication: bool = Field(
        default=True, description="Enable communication between agents during parallel thinking"
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
        "CONCURRENT PARALLEL MULTI-AGENT REASONING - Execute multiple thinking processes using specialized AI agents "
        "that communicate and collaborate. Each CPU core acts as an autonomous agent with specific roles "
        "(Security Analyst, Performance Optimizer, Architecture Reviewer, etc.) that maintain individual "
        "thoughts and context while sharing insights. Perfect for: complex problem-solving, architectural "
        "decisions, multi-domain analysis, agent consensus building, or when you want specialized agents "
        "working together on challenging questions. Agents form teams, communicate insights, and synthesize "
        "collaborative analysis. Choose 2-6 agents based on problem complexity and required expertise areas."
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

    def _detect_cpu_architecture(self) -> dict:
        """Detect CPU architecture and capabilities for cross-platform optimization"""
        cpu_info = {
            "architecture": platform.machine().lower(),
            "system": platform.system().lower(),
            "processor": platform.processor(),
            "available_cores": os.cpu_count() or 4,
            "is_intel": False,
            "is_amd": False,
            "is_apple_silicon": False,
            "is_arm": False,
            "supports_affinity": False,
            "supports_performance_cores": False,
            "recommended_strategy": "adaptive"
        }

        # Detect CPU vendor/type
        machine = cpu_info["architecture"]
        processor = cpu_info["processor"].lower()
        
        # Apple Silicon detection (M1, M2, M3, etc.)
        if "arm" in machine or "aarch64" in machine:
            cpu_info["is_arm"] = True
            if cpu_info["system"] == "darwin":
                cpu_info["is_apple_silicon"] = True
                cpu_info["supports_performance_cores"] = True
                cpu_info["recommended_strategy"] = "hybrid"  # Apple Silicon benefits from hybrid approach
        
        # Intel/AMD x86_64 detection
        elif "x86_64" in machine or "amd64" in machine:
            if "intel" in processor or "genuine intel" in processor:
                cpu_info["is_intel"] = True
            elif "amd" in processor or "authentic amd" in processor:
                cpu_info["is_amd"] = True
                # AMD with 3D V-Cache benefits from specific optimizations
                if "3d" in processor or "v-cache" in processor:
                    cpu_info["supports_performance_cores"] = True

        # Check OS-specific features
        if cpu_info["system"] == "linux":
            cpu_info["supports_affinity"] = hasattr(os, 'sched_setaffinity')
        elif cpu_info["system"] == "windows":
            # Windows supports processor affinity through different APIs
            cpu_info["supports_affinity"] = True  # Will use psutil if available
        elif cpu_info["system"] == "darwin":
            # macOS has limited affinity support, but we can still optimize
            cpu_info["supports_affinity"] = False  # macOS doesn't expose sched_setaffinity

        logger.debug(f"Detected CPU: {cpu_info}")
        return cpu_info

    def _get_optimal_cpu_cores(self, requested_cores: Optional[int] = None) -> int:
        """Detect optimal number of CPU cores to use with cross-platform optimization"""
        cpu_info = self._detect_cpu_architecture()
        available_cores = cpu_info["available_cores"]

        if requested_cores:
            # Respect user preference but cap at available cores
            return min(requested_cores, available_cores)

        # Platform and architecture-specific optimization
        if cpu_info["is_apple_silicon"]:
            # Apple Silicon has performance and efficiency cores
            # Use most cores but leave some for system (efficiency cores handle background)
            if available_cores <= 4:
                return available_cores
            elif available_cores <= 8:
                return available_cores - 1
            else:
                return min(10, available_cores - 2)  # M1 Ultra/M2 Ultra can handle more
        
        elif cpu_info["is_amd"] and cpu_info["supports_performance_cores"]:
            # AMD with 3D V-Cache benefits from more aggressive core usage
            if available_cores <= 4:
                return available_cores
            elif available_cores <= 8:
                return available_cores - 1
            else:
                return min(12, available_cores - 2)  # Ryzen 7000X3D series
        
        elif cpu_info["is_intel"]:
            # Intel CPUs with E-cores (12th gen+) need different handling
            if available_cores <= 4:
                return available_cores
            elif available_cores <= 8:
                return available_cores - 1
            else:
                # For high core count Intel (with E-cores), be more conservative
                return min(8, available_cores - 2)
        
        else:
            # Generic optimization for unknown architectures
            if available_cores <= 2:
                return available_cores
            elif available_cores <= 4:
                return available_cores - 1
            elif available_cores <= 8:
                return min(6, available_cores - 1)
            else:
                return min(8, available_cores - 2)

    def _get_optimal_batch_size(self, total_paths: int, cores: int) -> int:
        """Calculate optimal batch size for processing"""
        if total_paths <= cores:
            return 1  # Each path gets its own processing slot

        # Aim for 2-3 batches per core for good utilization
        target_batches = cores * 2
        batch_size = max(1, total_paths // target_batches)
        return min(batch_size, 3)  # Cap batch size for memory efficiency

    def _set_cpu_affinity(self, core_id: int) -> bool:
        """Set CPU affinity for current thread with cross-platform support"""
        try:
            cpu_info = getattr(self, '_cached_cpu_info', None)
            if not cpu_info:
                cpu_info = self._detect_cpu_architecture()
                self._cached_cpu_info = cpu_info

            # Linux/Unix systems with sched_setaffinity
            if cpu_info["system"] == "linux" and hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, {core_id})
                logger.debug(f"Set CPU affinity to core {core_id} (Linux)")
                return True
            
            # Windows systems - try using psutil if available
            elif cpu_info["system"] == "windows":
                try:
                    import psutil
                    process = psutil.Process()
                    # Set affinity to specific CPU core
                    process.cpu_affinity([core_id])
                    logger.debug(f"Set CPU affinity to core {core_id} (Windows)")
                    return True
                except ImportError:
                    logger.debug("psutil not available for Windows CPU affinity")
                except Exception as e:
                    logger.debug(f"Windows CPU affinity failed: {e}")
            
            # macOS - no direct affinity, but we can provide hints
            elif cpu_info["system"] == "darwin":
                # macOS doesn't support CPU affinity, but we can use thread priorities
                # and let the OS scheduler handle core assignment
                try:
                    import threading
                    current_thread = threading.current_thread()
                    # Apple Silicon has performance and efficiency cores
                    # Higher priority threads tend to get scheduled on performance cores
                    if cpu_info["is_apple_silicon"] and core_id < (cpu_info["available_cores"] // 2):
                        # Hint for performance cores (lower core IDs typically performance cores)
                        logger.debug(f"macOS: Hinting performance core for thread {current_thread.name}")
                    return True  # Return True since we provided optimization hints
                except Exception as e:
                    logger.debug(f"macOS CPU optimization hint failed: {e}")
            
        except Exception as e:
            logger.debug(f"Could not set CPU affinity to core {core_id}: {e}")
        
        return False

    def _store_core_context(self, path: ParallelThinkingPath, core_id: int, context_data: dict,
                           share_with_others: bool = False) -> None:
        """Store context data for the specific core processing this path"""
        try:
            storage = get_core_context_storage()

            # Store path-specific context
            path_context_key = f"path_{path.path_id}_context"
            storage.set_core_context(
                key=path_context_key,
                value=context_data,
                core_id=core_id,
                share_with_others=share_with_others
            )

            # Store approach-specific insights if sharing is enabled
            if share_with_others and "insights" in context_data:
                insights_key = f"approach_{path.approach.replace(' ', '_').lower()}_insights"
                storage.set_core_context(
                    key=insights_key,
                    value=context_data["insights"],
                    core_id=core_id,
                    share_with_others=True
                )
                path.shared_context_keys.append(insights_key)

            path.core_context_used = True
            logger.debug(f"Stored context for path {path.path_id} on core {core_id}")

        except Exception as e:
            logger.warning(f"Could not store core context for path {path.path_id}: {e}")

    def _retrieve_shared_insights(self, core_id: int, approach: str) -> dict:
        """Retrieve insights shared by other cores that might be relevant"""
        try:
            storage = get_core_context_storage()
            shared_insights = {}

            # Look for insights from similar approaches
            similar_approaches = [
                "analytical", "creative", "systematic", "risk_focused",
                "solution_oriented", "historical", "future_focused", "technical"
            ]

            for similar_approach in similar_approaches:
                if similar_approach.lower() in approach.lower():
                    continue  # Skip same approach

                insights_key = f"approach_{similar_approach}_insights"
                insights = storage.get_core_context(insights_key, core_id=None, check_shared=True)

                if insights:
                    shared_insights[similar_approach] = insights
                    logger.debug(f"Retrieved shared insights from {similar_approach} approach for core {core_id}")

            return shared_insights

        except Exception as e:
            logger.warning(f"Could not retrieve shared insights for core {core_id}: {e}")
            return {}

    def _retrieve_agent_insights(self, agent_system, requesting_agent) -> dict:
        """Retrieve insights from other agents in the same team"""
        try:
            insights = {}
            
            # Get insights from agents in the same teams
            for team_id in requesting_agent.team_memberships:
                if team_id in agent_system.teams:
                    team = agent_system.teams[team_id]
                    for member_id in team.members:
                        if member_id != requesting_agent.agent_id and member_id in agent_system.agents:
                            member_agent = agent_system.agents[member_id]
                            
                            # Get recent insights from this agent
                            recent_thoughts = member_agent.get_recent_thoughts(limit=5)
                            insight_thoughts = [t for t in recent_thoughts if t.thought_type == "insight"]
                            
                            if insight_thoughts:
                                latest_insight = insight_thoughts[-1]  # Most recent insight
                                insights[member_id] = latest_insight.content
                                
            return insights
            
        except Exception as e:
            logger.warning(f"Could not retrieve agent insights: {e}")
            return {}

    def _detect_gpu_availability(self) -> dict:
        """Detect light GPU support availability (keeping it optional to avoid overkill)"""
        gpu_info = {
            "available": False,
            "type": None,
            "memory": None,
            "compute_capability": None
        }

        try:
            # Try to detect NVIDIA GPU
            import subprocess
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                gpu_info["available"] = True
                gpu_info["type"] = "nvidia"
                gpu_info["memory"] = result.stdout.strip().split(',')[1].strip()
                logger.debug(f"Detected NVIDIA GPU: {gpu_info['memory']}")
        except Exception:
            pass

        try:
            # Try to detect integrated GPU or other accelerators
            if os.path.exists("/sys/class/drm"):
                gpu_info["available"] = True
                gpu_info["type"] = "integrated"
                logger.debug("Detected integrated GPU support")
        except Exception:
            pass

        # Note: We detect GPU but don't use it by default to avoid overkill
        # This information can be used for future optimizations
        return gpu_info

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (basic implementation)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback: use basic resource monitoring
            try:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Convert to MB
            except Exception:
                return 0.0

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

    def _determine_agent_roles(self, requested_roles: Optional[list[str]], path_count: int) -> list[AgentRole]:
        """Determine appropriate agent roles for each thinking path"""
        if requested_roles:
            # Use explicitly requested roles
            role_mapping = {
                "security_analyst": AgentRole.SECURITY_ANALYST,
                "performance_optimizer": AgentRole.PERFORMANCE_OPTIMIZER,
                "architecture_reviewer": AgentRole.ARCHITECTURE_REVIEWER,
                "code_quality_inspector": AgentRole.CODE_QUALITY_INSPECTOR,
                "debug_specialist": AgentRole.DEBUG_SPECIALIST,
                "planning_coordinator": AgentRole.PLANNING_COORDINATOR,
                "consensus_facilitator": AgentRole.CONSENSUS_FACILITATOR,
                "generalist": AgentRole.GENERALIST
            }
            
            roles = []
            for role_str in requested_roles[:path_count]:
                role = role_mapping.get(role_str.lower(), AgentRole.GENERALIST)
                roles.append(role)
            
            # Fill remaining slots with generalists if needed
            while len(roles) < path_count:
                roles.append(AgentRole.GENERALIST)
                
            return roles
        else:
            # Auto-assign roles based on path count and general purpose
            default_roles = [
                AgentRole.ARCHITECTURE_REVIEWER,    # Strategic overview
                AgentRole.SECURITY_ANALYST,         # Risk assessment  
                AgentRole.PERFORMANCE_OPTIMIZER,    # Efficiency focus
                AgentRole.CODE_QUALITY_INSPECTOR,   # Quality focus
                AgentRole.DEBUG_SPECIALIST,         # Problem-solving
                AgentRole.PLANNING_COORDINATOR,     # Organization
                AgentRole.CONSENSUS_FACILITATOR,    # Integration
                AgentRole.GENERALIST               # Broad perspective
            ]
            
            # Cycle through default roles based on path count
            roles = []
            for i in range(path_count):
                role = default_roles[i % len(default_roles)]
                roles.append(role)
                
            return roles

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
        self, path: ParallelThinkingPath, prompt: str, system_prompt: str, files_content: str,
        enable_core_context: bool = True, share_insights: bool = True, agent_system=None
    ) -> ParallelThinkingPath:
        """Execute a single thinking path asynchronously with core context and agent communication support"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        # Record thread information for debugging
        path.thread_id = threading.get_ident()

        # Get agent if assigned
        agent = None
        agent_api_client = None
        resolved_model_name: Optional[str] = path.model
        if agent_system and path.assigned_agent:
            agent = agent_system.agents.get(path.assigned_agent)
            agent_api_client = agent_system.get_agent_api_client(path.assigned_agent)
            if agent:
                agent.update_status(AgentStatus.THINKING)
                agent.add_thought(
                    thought_type="analysis",
                    content=f"Starting analysis with approach: {path.approach}",
                    confidence=0.8
                )

        try:
            # Get shared insights from other cores if enabled
            shared_insights = {}
            if enable_core_context and share_insights and path.cpu_core is not None:
                shared_insights = self._retrieve_shared_insights(path.cpu_core, path.approach)

            # Also get agent insights if agent mode is enabled
            agent_insights = {}
            if agent_system and agent:
                agent_insights = self._retrieve_agent_insights(agent_system, agent)

            # Prepare the full prompt for this path
            if path.prompt_variation:
                full_prompt = path.prompt_variation
            else:
                approach_instruction = f"\n\nTHINKING APPROACH: {path.approach}\n"
                
                # Add agent personality and role context if available
                if agent:
                    role_context = f"AGENT ROLE: {agent.role.value.replace('_', ' ').title()}\n"
                    personality_context = f"COMMUNICATION STYLE: {agent.personality.communication_style}\n"
                    decision_style = f"DECISION MAKING: {agent.personality.decision_making_style}\n"
                    approach_instruction = f"\n\n{role_context}{personality_context}{decision_style}\n{approach_instruction}"
                
                full_prompt = approach_instruction + prompt

            # Add shared insights to prompt if available
            if shared_insights:
                insights_text = "\n\nSHARED INSIGHTS FROM OTHER CORES:\n"
                for approach, insights in shared_insights.items():
                    insights_text += f"- {approach.title()}: {insights}\n"
                full_prompt += insights_text

            # Add agent insights if available
            if agent_insights:
                agent_insights_text = "\n\nINSIGHTS FROM OTHER AGENTS:\n"
                for agent_id, insight in agent_insights.items():
                    agent_insights_text += f"- Agent {agent_id}: {insight}\n"
                full_prompt += agent_insights_text

            if files_content:
                full_prompt = f"{full_prompt}\n\nFILE CONTEXT:\n{files_content}"

            # Use agent's own API client if available, otherwise fall back to centralized approach
            agent_api_failed = False
            if agent_api_client:
                # Agent makes its own API call
                logger.debug(f"Agent {path.assigned_agent} making direct API call")
                try:
                    api_call = await agent_api_client.make_api_call(
                        prompt=full_prompt,
                        model_name=path.model,
                        parameters={
                            "system_prompt": system_prompt,
                            "temperature": self.get_default_temperature(),
                            "thinking_mode": self.get_default_thinking_mode()
                        }
                    )
                    
                    if api_call.status == "completed" and api_call.response:
                        path.result = api_call.response
                        resolved_model_name = api_call.model_name or resolved_model_name
                        logger.info(f"Agent {path.assigned_agent} completed API call successfully")
                    else:
                        # Check if this is a provider availability issue
                        if api_call.error and "No available providers" in api_call.error:
                            logger.warning(f"Agent {path.assigned_agent} has no available providers, falling back to centralized approach")
                            agent_api_failed = True
                        else:
                            raise Exception(f"Agent API call failed: {api_call.error}")
                except Exception as e:
                    logger.warning(f"Agent API call failed: {e}, falling back to centralized approach")
                    agent_api_failed = True
                    
            if not agent_api_client or agent_api_failed:
                # Fallback to centralized approach
                logger.debug(f"Using centralized API call for path {path.path_id}")

                # Get provider and model
                registry = ModelProviderRegistry()

                # Normalise provider list to handle mocks in unit tests gracefully
                available_providers: list[Any] = []
                try:
                    raw_providers = registry.get_available_providers()
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(f"Failed to obtain available providers: {exc}")
                    raw_providers = None

                if raw_providers:
                    if isinstance(raw_providers, (list, tuple, set)):
                        available_providers = list(raw_providers)
                    else:
                        try:
                            available_providers = list(raw_providers)
                        except TypeError:
                            # Some tests patch the registry with simple mocks; treat those as no providers
                            available_providers = []

                provider = None
                selected_model_name: Optional[str] = resolved_model_name

                if path.model:
                    # Try to use specified model
                    try:
                        provider = registry.get_provider_for_model(path.model)
                        selected_model_name = path.model
                    except Exception as e:
                        logger.warning(f"Could not use model {path.model}: {e}, trying first available provider")

                if not provider and available_providers:
                    # Get first available provider
                    provider_type = available_providers[0]
                    provider = registry.get_provider(provider_type)
                    available_models: list[str] = []

                    get_models = getattr(registry, "get_available_model_names", None)
                    if callable(get_models):
                        try:
                            available_models = get_models(provider_type)
                        except Exception as exc:  # pragma: no cover - defensive logging
                            logger.debug(f"Could not list models for provider {provider_type}: {exc}")

                    if available_models:
                        selected_model_name = available_models[0]

                if not provider:
                    # Fall back to registry default provider (used in unit tests)
                    default_provider_fn = getattr(registry, "get_default_provider", None)
                    if callable(default_provider_fn):
                        provider = default_provider_fn()
                        if provider and not selected_model_name:
                            default_model_getter = getattr(provider, "get_default_model", None)
                            if callable(default_model_getter):
                                selected_model_name = default_model_getter()
                            if not selected_model_name:
                                selected_model_name = getattr(provider, "default_model", None)

                if provider and selected_model_name:
                    # Execute the thinking
                    response = provider.generate_content(
                        prompt=full_prompt,
                        model_name=selected_model_name,
                        system_prompt=system_prompt,
                        temperature=self.get_default_temperature(),
                        thinking_mode=self.get_default_thinking_mode(),
                    )
                    path.result = getattr(response, "content", response)
                    resolved_model_name = selected_model_name
                elif not available_providers and not provider:
                    # No providers available at all - this is expected when running outside server context
                    logger.warning(
                        f"No providers available for centralized approach in path {path.path_id}. "
                        "This occurs when using tools outside the server context without configured providers."
                    )
                    path.result = "Error: No AI providers available. Please configure API keys and restart the server."
                    path.error = "No providers configured"
                else:
                    # Even the centralized approach failed
                    logger.error(f"No providers or models available for centralized approach in path {path.path_id}")
                    path.result = "Error: No AI providers or models available for processing."
                    path.error = "No providers or models available"

            path.execution_time = time.time() - start_time
            path.memory_usage = max(0, self._get_memory_usage() - start_memory)

            # Agent completes thinking and shares insights
            if agent:
                agent.update_status(AgentStatus.COMMUNICATING)
                agent.add_thought(
                    thought_type="insight",
                    content=f"Completed analysis in {path.execution_time:.2f}s. Key findings ready to share.",
                    confidence=0.9
                )

            # Store context and insights if enabled
            if enable_core_context and path.cpu_core is not None:
                # Extract key insights from the result for sharing
                context_data = {
                    "approach": path.approach,
                    "execution_time": path.execution_time,
                    "memory_usage": path.memory_usage,
                    "model_used": resolved_model_name or "unspecified",
                    "shared_insights_used": bool(shared_insights),
                    "agent_insights_used": bool(agent_insights),
                    "assigned_agent": path.assigned_agent
                }

                # Enhanced insight extraction
                raw_result = path.result
                if raw_result is None:
                    result_text = ""
                elif isinstance(raw_result, str):
                    result_text = raw_result.lower()
                else:
                    result_text = str(getattr(raw_result, "content", raw_result)).lower()

                insights: list[str] = []

                if "performance" in result_text or "optimization" in result_text:
                    insights.append("performance_considerations")
                if "security" in result_text or "vulnerability" in result_text:
                    insights.append("security_aspects")
                if "scalability" in result_text or "scale" in result_text:
                    insights.append("scalability_factors")
                if "complexity" in result_text or "complex" in result_text:
                    insights.append("complexity_analysis")
                if "architecture" in result_text or "design" in result_text:
                    insights.append("architectural_considerations")

                if insights:
                    context_data["insights"] = insights

                # Store with sharing enabled if insights were found
                should_share = share_insights and len(insights) > 0
                self._store_core_context(path, path.cpu_core, context_data, should_share)

                # Share insights through agent communication if enabled
                if agent_system and agent and insights:
                    insight_message = f"Discovered {len(insights)} key insights: {', '.join(insights)}"
                    agent_system.send_message(
                        from_agent=agent.agent_id,
                        to_agent="ALL",
                        message_type="insight",
                        content=insight_message,
                        priority=7
                    )

        except Exception as e:
            path.error = str(e)
            path.execution_time = time.time() - start_time
            path.memory_usage = max(0, self._get_memory_usage() - start_memory)
            logger.error(f"Error in thinking path {path.path_id}: {e}")
            
            if agent:
                agent.update_status(AgentStatus.ACTIVE)
                agent.add_thought(
                    thought_type="concern",
                    content=f"Encountered error during analysis: {str(e)}",
                    confidence=0.9
                )

        # Agent returns to active status
        if agent:
            agent.update_status(AgentStatus.ACTIVE)

        return path

    def _execute_thinking_path_sync(
        self, path: ParallelThinkingPath, prompt: str, system_prompt: str, files_content: str,
        core_id: Optional[int] = None, enable_core_context: bool = True, share_insights: bool = True, agent_system=None
    ) -> ParallelThinkingPath:
        """Execute a single thinking path synchronously (for thread pool execution)"""
        if core_id is not None:
            path.cpu_core = core_id
            # Try to set CPU affinity if enabled
            self._set_cpu_affinity(core_id)

        # Use asyncio.run to execute the async version in a thread
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            # We're in an async context, need to create a new event loop for this thread
            async def run_in_thread():
                return await self._execute_thinking_path(
                    path, prompt, system_prompt, files_content, enable_core_context, share_insights, agent_system
                )

            # Create a new event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result = new_loop.run_until_complete(run_in_thread())
                return result
            finally:
                new_loop.close()
        else:
            # Not in an async context, can use asyncio.run
            return asyncio.run(self._execute_thinking_path(
                path, prompt, system_prompt, files_content, enable_core_context, share_insights, agent_system
            ))

    async def _execute_paths_hybrid(
        self, paths: list[ParallelThinkingPath], prompt: str, system_prompt: str, files_content: str,
        cores: int, batch_size: int, enable_cpu_affinity: bool, enable_core_context: bool = True,
        share_insights: bool = True, agent_system=None
    ) -> list[ParallelThinkingPath]:
        """Execute thinking paths using hybrid concurrency strategy with core context and agent support"""
        logger.info(f"Executing {len(paths)} thinking paths using {cores} cores with batch size {batch_size}")

        # Create thread pool for CPU-bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=cores, thread_name_prefix="ThinkingPath") as executor:
            # Submit all paths to thread pool
            future_to_path = {}

            for i, path in enumerate(paths):
                # Assign core if CPU affinity is enabled
                core_id = i % cores if enable_cpu_affinity else None
                future = executor.submit(
                    self._execute_thinking_path_sync,
                    path, prompt, system_prompt, files_content, core_id, enable_core_context, share_insights, agent_system
                )
                future_to_path[future] = path

            # Collect results as they complete
            completed_paths = []
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    result_path = future.result()
                    completed_paths.append(result_path)
                except Exception as e:
                    path = future_to_path[future]
                    path.error = str(e)
                    path.execution_time = 0
                    completed_paths.append(path)
                    logger.error(f"Thread execution failed for path {path.path_id}: {e}")

        return completed_paths

    async def _execute_paths_asyncio(
        self, paths: list[ParallelThinkingPath], prompt: str, system_prompt: str, files_content: str,
        enable_core_context: bool = True, share_insights: bool = True, agent_system=None
    ) -> list[ParallelThinkingPath]:
        """Execute thinking paths using pure asyncio (original approach)"""
        tasks = [
            self._execute_thinking_path(path, prompt, system_prompt, files_content, enable_core_context, share_insights, agent_system)
            for path in paths
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _choose_execution_strategy(self, strategy: str, num_paths: int, cores: int) -> str:
        """Choose optimal execution strategy based on context and CPU architecture"""
        if strategy == "adaptive":
            cpu_info = getattr(self, '_cached_cpu_info', None)
            if not cpu_info:
                cpu_info = self._detect_cpu_architecture()
                self._cached_cpu_info = cpu_info

            # Architecture-specific strategy selection
            if cpu_info["is_apple_silicon"]:
                # Apple Silicon benefits from hybrid approach due to P/E core design
                if num_paths <= 2:
                    return "asyncio"
                else:
                    return "hybrid"  # Leverage both performance and efficiency cores
            
            elif cpu_info["is_amd"] and cpu_info["supports_performance_cores"]:
                # AMD 3D V-Cache chips benefit from threading due to large cache
                if num_paths <= 2:
                    return "asyncio"
                elif num_paths >= 4:
                    return "threads"  # Take advantage of 3D V-Cache
                else:
                    return "hybrid"
            
            elif cpu_info["is_intel"]:
                # Intel 12th gen+ with E-cores
                if cores > 8:  # Likely has E-cores
                    if num_paths <= 2:
                        return "asyncio"
                    else:
                        return "hybrid"  # Balance P and E cores
                else:
                    # Traditional Intel without E-cores
                    if num_paths <= 2:
                        return "asyncio"
                    elif cores >= 4 and num_paths >= cores:
                        return "threads"
                    else:
                        return "hybrid"
            
            else:
                # Generic strategy selection for unknown architectures
                if num_paths <= 2:
                    return "asyncio"  # Simple async for small workloads
                elif cores >= 4 and num_paths >= cores:
                    return "threads"  # Use threads for CPU-bound parallel work
                else:
                    return "hybrid"  # Hybrid approach for balanced workloads
        
        return strategy

    def _synthesize_results(
        self, paths: list[ParallelThinkingPath], synthesis_style: str, request: ParallelThinkRequest
    ) -> str:
        """Synthesize results from multiple thinking paths with core context information"""
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

            # Performance metrics
            total_time = max(p.execution_time for p in paths) if paths else 0
            total_memory = sum(getattr(p, 'memory_usage', 0) for p in successful_paths)
            avg_time = sum(p.execution_time for p in successful_paths) / len(successful_paths) if successful_paths else 0

            synthesis += "**Execution Summary:**\n"
            synthesis += f"- Total paths: {len(paths)} | Successful: {len(successful_paths)}\n"
            synthesis += f"- Total execution time: {total_time:.1f}s | Average per path: {avg_time:.1f}s\n"
            synthesis += f"- Memory usage: {total_memory:.1f}MB\n"

            # CPU core utilization summary
            cores_used = {getattr(p, 'cpu_core', None) for p in paths if getattr(p, 'cpu_core', None) is not None}
            if cores_used:
                synthesis += f"- CPU cores utilized: {len(cores_used)} ({sorted(cores_used)})\n"

            # Core context utilization summary
            context_enabled_paths = [p for p in successful_paths if getattr(p, 'core_context_used', False)]
            if context_enabled_paths and request.enable_core_context:
                synthesis += f"- Core context enabled: {len(context_enabled_paths)}/{len(successful_paths)} paths\n"

                # Count shared insights
                total_shared_keys = sum(len(getattr(p, 'shared_context_keys', [])) for p in context_enabled_paths)
                if total_shared_keys > 0:
                    synthesis += f"- Insights shared between cores: {total_shared_keys} instances\n"

            synthesis += "\n"

            for i, path in enumerate(successful_paths, 1):
                synthesis += f"### Path {i}: {path.approach}\n"
                if path.model:
                    synthesis += f"**Model:** {path.model}\n"
                synthesis += f"**Execution time:** {path.execution_time:.1f}s"
                if hasattr(path, 'cpu_core') and path.cpu_core is not None:
                    synthesis += f" | **CPU Core:** {path.cpu_core}"
                if hasattr(path, 'memory_usage') and path.memory_usage > 0:
                    synthesis += f" | **Memory:** {path.memory_usage:.1f}MB"
                if hasattr(path, 'core_context_used') and path.core_context_used:
                    synthesis += " | **Core Context:** Enabled"
                if hasattr(path, 'shared_context_keys') and path.shared_context_keys:
                    synthesis += f" | **Shared Insights:** {len(path.shared_context_keys)}"
                synthesis += "\n\n"
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

            # Smart CPU core and execution strategy selection
            optimal_cores = self._get_optimal_cpu_cores(request.cpu_cores)
            batch_size = request.batch_size or self._get_optimal_batch_size(len(paths), optimal_cores)
            execution_strategy = self._choose_execution_strategy(request.execution_strategy, len(paths), optimal_cores)

            # Detect GPU availability (optional, avoiding overkill)
            gpu_info = self._detect_gpu_availability()
            if gpu_info["available"]:
                logger.info(f"GPU detected ({gpu_info['type']}) but using CPU-focused approach to avoid overkill")

            logger.info(f"Parallel execution: {len(paths)} paths, {optimal_cores} cores, strategy: {execution_strategy}")

            # Get core context storage for statistics
            core_storage = None
            if request.enable_core_context:
                try:
                    core_storage = get_core_context_storage()
                except Exception as e:
                    logger.warning(f"Could not initialize core context storage: {e}")

            # Determine if any providers are available for agent mode; disable when none
            providers_available = True
            try:
                registry_check = ModelProviderRegistry()
                raw_available = registry_check.get_available_providers()
                if not raw_available:
                    providers_available = False
                elif isinstance(raw_available, (list, tuple, set)):
                    providers_available = len(raw_available) > 0
                else:
                    try:
                        providers_available = len(list(raw_available)) > 0
                    except TypeError:
                        providers_available = False
            except Exception:
                providers_available = False

            if not providers_available:
                request.enable_agent_mode = False

            # Initialize agent communication system if agent mode is enabled
            agent_system = None
            agents = []
            team_id = None
            if request.enable_agent_mode:
                try:
                    agent_system = get_agent_communication_system()

                    # Use automatic agent selection if enabled
                    if request.auto_select_agents:
                        from utils.automatic_agent_selector import get_automatic_agent_selector, TaskCharacteristics, TaskType, TaskComplexity
                        
                        # Get the automatic agent selector
                        agent_selector = get_automatic_agent_selector(agent_system)
                        
                        # Analyze the task to determine characteristics
                        task_characteristics = agent_selector.analyze_task_from_prompt(
                            request.prompt, request.files
                        )
                        
                        # Override task type for parallel thinking
                        task_characteristics.task_type = TaskType.PARALLEL_THINKING
                        
                        logger.info(f"Auto-selecting agents for {task_characteristics.task_type.value} "
                                   f"task with {task_characteristics.complexity.value} complexity")
                        
                        # Select optimal agents for this task
                        selected_agent_ids, coordinator_id = agent_selector.select_agents_for_task(task_characteristics)
                        
                        # If we have fewer agents than paths, register additional agents
                        if len(selected_agent_ids) < len(paths):
                            logger.info(f"Need {len(paths)} agents but only {len(selected_agent_ids)} selected, registering additional agents")
                            
                            # Get available agent roles for the additional agents
                            used_roles = [agent_system.agents[aid].role for aid in selected_agent_ids]
                            available_roles = [role for role in AgentRole if role not in used_roles]
                            
                            # Register additional agents
                            for i in range(len(selected_agent_ids), len(paths)):
                                core_id = i % optimal_cores
                                role = available_roles[i % len(available_roles)] if available_roles else AgentRole.GENERALIST
                                agent = agent_system.register_agent(core_id=core_id, role=role)
                                selected_agent_ids.append(agent.agent_id)
                        
                        # Use the selected agents
                        for i, agent_id in enumerate(selected_agent_ids[:len(paths)]):
                            agent = agent_system.agents[agent_id]
                            agents.append(agent)
                            paths[i].assigned_agent = agent_id
                            
                            # Add initial thought about the task
                            agent.add_thought(
                                thought_type="analysis",
                                content=f"Auto-selected for approach: {paths[i].approach}. Beginning parallel thinking analysis.",
                                confidence=0.8
                            )
                    else:
                        # Manual agent role assignment (original logic)
                        # Determine agent roles for each path
                        agent_roles = self._determine_agent_roles(request.agent_roles, len(paths))
                        
                        # Create agents for each core/path
                        for i, (path, role) in enumerate(zip(paths, agent_roles)):
                            core_id = i % optimal_cores
                            agent = agent_system.register_agent(core_id=core_id, role=role)
                            agents.append(agent)
                            path.assigned_agent = agent.agent_id
                            
                            # Add initial thought about the task
                            agent.add_thought(
                                thought_type="analysis",
                                content=f"Assigned to approach: {path.approach}. Beginning parallel thinking analysis.",
                                confidence=0.8
                            )
                    
                    # Create a team for this parallel thinking session
                    if len(agents) > 1:
                        team_id = agent_system.create_team(
                            team_name=f"ParallelThink_{int(time.time())}",
                            purpose=f"Collaborative analysis: {request.prompt[:100]}..."
                        )
                        
                        # Add all agents to the team
                        for agent in agents:
                            agent_system.add_agent_to_team(agent.agent_id, team_id)
                            
                        logger.info(f"Created agent team {team_id} with {len(agents)} agents")
                    
                except Exception as e:
                    logger.warning(f"Could not initialize agent communication system: {e}")
                    request.enable_agent_mode = False  # Fall back to non-agent mode

            # Execute all thinking paths concurrently
            system_prompt = self.get_system_prompt()
            start_time = time.time()

            try:
                if execution_strategy == "threads":
                    # Use thread pool for CPU-bound parallel execution
                    completed_paths = await self._execute_paths_hybrid(
                        paths, request.prompt, system_prompt, files_content,
                        optimal_cores, batch_size, request.enable_cpu_affinity,
                        request.enable_core_context, request.share_insights_between_cores, agent_system
                    )
                elif execution_strategy == "hybrid":
                    # Use hybrid approach
                    completed_paths = await self._execute_paths_hybrid(
                        paths, request.prompt, system_prompt, files_content,
                        optimal_cores, batch_size, request.enable_cpu_affinity,
                        request.enable_core_context, request.share_insights_between_cores, agent_system
                    )
                else:  # asyncio
                    # Use pure asyncio (original approach)
                    completed_paths = await asyncio.wait_for(
                        self._execute_paths_asyncio(
                            paths, request.prompt, system_prompt, files_content,
                            request.enable_core_context, request.share_insights_between_cores, agent_system
                        ),
                        timeout=request.time_limit or 60
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Parallel thinking timed out after {request.time_limit}s")
                # Use partial results
                completed_paths = [p for p in paths if p.result or p.error]
            except Exception as e:
                logger.error(f"Error in parallel execution: {e}")
                # Use any partial results available
                completed_paths = [p for p in paths if p.result or p.error]

            total_execution_time = time.time() - start_time

            # Synthesize results
            synthesis = self._synthesize_results(completed_paths, request.synthesis_style, request)

            # Prepare response with enhanced metrics
            successful_count = len([
                p for p in completed_paths
                if hasattr(p, "result") and p.result and not getattr(p, "error", None)
            ])

            # Get core context statistics
            core_context_stats = {}
            if request.enable_core_context and core_storage:
                try:
                    core_context_stats = core_storage.get_core_statistics()
                except Exception as e:
                    logger.warning(f"Could not get core context statistics: {e}")

            response = {
                "parallel_thinking_analysis": synthesis,
                "execution_summary": {
                    "total_paths": len(paths),
                    "successful_paths": successful_count,
                    "approaches_used": [p.approach for p in paths],
                    "models_used": list({p.model for p in paths if p.model}) or ["default"],
                    "total_execution_time": total_execution_time,
                    "average_path_time": (
                        sum([getattr(p, "execution_time", 0) for p in completed_paths]) / len(completed_paths)
                        if completed_paths else 0
                    ),
                    "synthesis_style": request.synthesis_style,
                    # Enhanced performance metrics
                    "cpu_cores_used": optimal_cores,
                    "execution_strategy": execution_strategy,
                    "batch_size": batch_size,
                    "total_memory_usage": sum([getattr(p, "memory_usage", 0) for p in completed_paths]),
                    "cores_utilized": sorted({
                        getattr(p, "cpu_core", None) for p in completed_paths
                        if getattr(p, "cpu_core", None) is not None
                    }),
                    # Core context metrics
                    "core_context_enabled": request.enable_core_context,
                    "insights_sharing_enabled": request.share_insights_between_cores,
                    "core_context_stats": core_context_stats,
                    "gpu_detected": gpu_info.get("available", False),
                    "gpu_type": gpu_info.get("type", None) if gpu_info.get("available") else None,
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
                            # Enhanced path metrics
                            "cpu_core": getattr(path, "cpu_core", None),
                            "thread_id": getattr(path, "thread_id", None),
                            "memory_usage": getattr(path, "memory_usage", 0),
                            "core_context_used": getattr(path, "core_context_used", False),
                            "shared_context_keys": getattr(path, "shared_context_keys", []),
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
