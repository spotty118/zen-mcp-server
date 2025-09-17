"""
Agent API Client - Direct API calling capabilities for agents

This module enables agents to make their own direct API calls to AI providers
instead of routing through the main server. This allows for true agent autonomy
and parallel processing capabilities.

Key Features:
- Direct API calls from individual agents
- Provider routing and selection per agent
- API rate limiting and error handling per agent
- Response caching and optimization
- Agent-specific model preferences and routing
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from providers.registry import ModelProviderRegistry
from providers.base import ProviderType
from utils.agent_core import Agent, AgentStatus
from utils.per_core_error_handling import get_per_core_error_handler, ErrorCategory
from utils.per_core_logging import get_per_core_logger, log_openrouter_api_call, log_agent_activity

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterConfig:
    """OpenRouter-specific configuration for an agent"""
    api_key: str
    preferred_models: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    max_concurrent_calls: int = 3
    thinking_mode_default: str = "high"
    temperature_range: Tuple[float, float] = (0.3, 0.8)
    fallback_enabled: bool = True
    circuit_breaker_threshold: int = 5  # Number of consecutive failures before circuit opens
    circuit_breaker_timeout: float = 300.0  # 5 minutes before trying again
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_models": self.preferred_models,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "max_concurrent_calls": self.max_concurrent_calls,
            "thinking_mode_default": self.thinking_mode_default,
            "temperature_range": self.temperature_range,
            "fallback_enabled": self.fallback_enabled,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout
        }


@dataclass
class AgentAPICall:
    """Represents an API call made by an agent"""
    call_id: str
    agent_id: str
    provider_type: ProviderType
    model_name: str
    prompt: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"  # pending, in_progress, completed, failed
    response: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    is_thinking_session: bool = False
    openrouter_usage: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "agent_id": self.agent_id,
            "provider_type": self.provider_type.value,
            "model_name": self.model_name,
            "prompt": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "status": self.status,
            "execution_time": self.execution_time,
            "retry_count": self.retry_count,
            "has_response": self.response is not None,
            "has_error": self.error is not None,
            "is_thinking_session": self.is_thinking_session,
            "openrouter_usage": self.openrouter_usage
        }


class AgentAPIClient:
    """
    API client for individual agents to make direct API calls
    """
    
    def __init__(self, agent: Agent, max_concurrent_calls: int = 3):
        self.agent = agent
        self.max_concurrent_calls = max_concurrent_calls
        self.active_calls: Dict[str, AgentAPICall] = {}
        self.call_history: List[AgentAPICall] = []
        # Don't initialize provider_registry here - access it lazily in _get_provider_registry()
        
        # Rate limiting per agent
        self.last_call_time = 0.0
        self.min_call_interval = 1.0  # Minimum seconds between calls
        self.openrouter_call_times: List[float] = []  # Track OpenRouter call times for rate limiting
        
        # OpenRouter-only configuration (must be set before determining preferred providers)
        self.openrouter_only = False
        self.openrouter_config: Optional[OpenRouterConfig] = None
        
        # Circuit breaker for OpenRouter failures
        self.openrouter_failure_count = 0
        self.openrouter_circuit_open = False
        self.openrouter_circuit_open_time = 0.0
        
        # Enhanced circuit breaker attributes
        self.openrouter_circuit_half_open = False
        self.openrouter_half_open_time = 0.0
        self.openrouter_half_open_attempts = 0
        
        # Preferred providers for this agent (depends on openrouter_only)
        self.preferred_providers = self._determine_preferred_providers()
        
        logger.info(f"Agent API client created for {agent.agent_id} with preferred providers: {[p.value for p in self.preferred_providers]}")
    
    def _get_provider_registry(self) -> ModelProviderRegistry:
        """Get the provider registry lazily to ensure it's properly configured"""
        registry = ModelProviderRegistry()

        # Some entry points (demos/tests) construct AgentAPIClient before the
        # server has called configure_providers(). In that situation the registry
        # will be empty and agents would fail their very first API call. Retry a
        # lazy provider configuration so agents can still function.
        try:
            if not registry.get_available_providers():
                from server import configure_providers

                configure_providers()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug(f"Provider auto-configuration skipped: {exc}")

        return registry
    
    def _determine_preferred_providers(self) -> List[ProviderType]:
        """Determine preferred providers based on agent role and available APIs."""
        from utils.agent_core import AgentRole

        # If OpenRouter-only mode is enabled, only use OpenRouter
        if self.openrouter_only:
            return [ProviderType.OPENROUTER]

        # Role-specific provider preferences for thinking sessions
        role_provider_preferences = {
            AgentRole.SECURITY_ANALYST: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.GOOGLE],
            AgentRole.PERFORMANCE_OPTIMIZER: [ProviderType.GOOGLE, ProviderType.OPENROUTER, ProviderType.OPENAI],
            AgentRole.ARCHITECTURE_REVIEWER: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.GOOGLE],
            AgentRole.CODE_QUALITY_INSPECTOR: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.GOOGLE],
            AgentRole.DEBUG_SPECIALIST: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.GOOGLE],
            AgentRole.PLANNING_COORDINATOR: [ProviderType.OPENAI, ProviderType.OPENROUTER, ProviderType.GOOGLE],
            AgentRole.CONSENSUS_FACILITATOR: [ProviderType.OPENROUTER, ProviderType.OPENAI, ProviderType.GOOGLE],
            AgentRole.GENERALIST: [ProviderType.OPENROUTER, ProviderType.GOOGLE, ProviderType.OPENAI]
        }
        
        preferred = role_provider_preferences.get(self.agent.role, [
            ProviderType.OPENROUTER,
            ProviderType.OPENAI,
            ProviderType.GOOGLE,
            ProviderType.XAI,
            ProviderType.DIAL,
            ProviderType.CUSTOM,
        ])
        
        return preferred
    
    async def make_api_call(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        parameters: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        is_thinking_session: bool = False
    ) -> AgentAPICall:
        """
        Make an API call directly from this agent
        
        Args:
            prompt: The prompt to send to the AI model
            model_name: Specific model to use (optional)
            provider_type: Specific provider to use (optional) 
            parameters: Additional parameters for the API call
            max_retries: Maximum number of retries on failure
            is_thinking_session: Whether this is a thinking session call (affects model selection)
            
        Returns:
            AgentAPICall object with the response
        """
        call_id = str(uuid.uuid4())
        parameters = parameters or {}
        
        # Rate limiting check
        current_time = time.time()
        if current_time - self.last_call_time < self.min_call_interval:
            await asyncio.sleep(self.min_call_interval - (current_time - self.last_call_time))
        
        # Check concurrent call limit
        if len(self.active_calls) >= self.max_concurrent_calls:
            logger.warning(f"Agent {self.agent.agent_id} has reached max concurrent calls limit ({self.max_concurrent_calls})")
            # Wait for oldest call to complete
            oldest_call = min(self.active_calls.values(), key=lambda c: c.timestamp)
            await self._wait_for_call(oldest_call.call_id)
        
        # Select provider and model
        selected_provider, selected_model = self._select_provider_and_model(provider_type, model_name, is_thinking_session)
        
        if not selected_provider:
            error_msg = (f"No available providers for agent {self.agent.agent_id}. "
                        "This may occur when using agent-based tools outside the server context. "
                        "Ensure API keys are configured and configure_providers() is called.")
            logger.error(error_msg)
            api_call = AgentAPICall(
                call_id=call_id,
                agent_id=self.agent.agent_id,
                provider_type=ProviderType.GOOGLE,  # Default fallback
                model_name="unavailable",
                prompt=prompt,
                parameters=parameters,
                status="failed",
                error=error_msg
            )
            return api_call
        
        # Create API call object
        api_call = AgentAPICall(
            call_id=call_id,
            agent_id=self.agent.agent_id,
            provider_type=selected_provider,
            model_name=selected_model,
            prompt=prompt,
            parameters=parameters,
            status="pending",
            is_thinking_session=is_thinking_session
        )
        
        # Apply OpenRouter-specific rate limiting
        if selected_provider == ProviderType.OPENROUTER:
            await self._apply_openrouter_rate_limiting()
        
        self.active_calls[call_id] = api_call
        self.last_call_time = current_time
        
        # Update agent status
        self.agent.update_status(AgentStatus.COMMUNICATING)
        
        # Add thought about making API call
        self.agent.add_thought(
            thought_type="api_call",
            content=f"Making API call to {selected_provider.value} model {selected_model}",
            confidence=0.8
        )
        
        try:
            # Log API call start
            log_agent_activity(
                agent_id=self.agent.agent_id,
                activity_type="api_call_start",
                activity_details={
                    "call_id": call_id,
                    "provider": selected_provider.value,
                    "model": selected_model,
                    "is_thinking_session": is_thinking_session,
                    "prompt_length": len(prompt)
                }
            )
            
            # Execute the API call
            response = await self._execute_api_call(api_call, max_retries)
            api_call.response = response
            api_call.status = "completed"
            
            # Handle OpenRouter success tracking
            if selected_provider == ProviderType.OPENROUTER:
                self._handle_openrouter_success()
            
            # Log successful API call
            if selected_provider == ProviderType.OPENROUTER:
                log_openrouter_api_call(
                    agent_id=self.agent.agent_id,
                    api_call_id=call_id,
                    model_used=selected_model,
                    request_type="thinking_session" if is_thinking_session else "standard",
                    response_time_ms=api_call.execution_time * 1000 if api_call.execution_time else 0,
                    success=True
                )
            
            # Log successful activity
            log_agent_activity(
                agent_id=self.agent.agent_id,
                activity_type="api_call_success",
                activity_details={
                    "call_id": call_id,
                    "provider": selected_provider.value,
                    "model": selected_model,
                    "response_length": len(response) if response else 0
                },
                duration_ms=api_call.execution_time * 1000 if api_call.execution_time else 0,
                success=True
            )
            
            # Add thought about successful call
            self.agent.add_thought(
                thought_type="api_response",
                content=f"Received response from {selected_provider.value}: {response[:100]}...",
                confidence=0.9
            )
            
        except Exception as e:
            api_call.error = str(e)
            api_call.status = "failed"
            
            # Handle OpenRouter failure tracking
            if selected_provider == ProviderType.OPENROUTER:
                self._handle_openrouter_failure(e)
            
            # Use comprehensive error handler
            error_handler = get_per_core_error_handler()
            recovery_successful = await error_handler.handle_error(
                error=e,
                agent_id=self.agent.agent_id,
                core_id=self.agent.core_id,
                context={
                    "call_id": call_id,
                    "provider": selected_provider.value,
                    "model": selected_model,
                    "is_thinking_session": is_thinking_session,
                    "retry_count": getattr(api_call, 'retry_count', 0)
                },
                api_client=self
            )
            
            # Log failed API call
            if selected_provider == ProviderType.OPENROUTER:
                log_openrouter_api_call(
                    agent_id=self.agent.agent_id,
                    api_call_id=call_id,
                    model_used=selected_model,
                    request_type="thinking_session" if is_thinking_session else "standard",
                    response_time_ms=api_call.execution_time * 1000 if api_call.execution_time else 0,
                    success=False,
                    error_message=str(e),
                    rate_limited="rate limit" in str(e).lower()
                )
            
            # Log failed activity
            log_agent_activity(
                agent_id=self.agent.agent_id,
                activity_type="api_call_failure",
                activity_details={
                    "call_id": call_id,
                    "provider": selected_provider.value,
                    "model": selected_model,
                    "error_type": type(e).__name__,
                    "recovery_attempted": True,
                    "recovery_successful": recovery_successful
                },
                duration_ms=api_call.execution_time * 1000 if api_call.execution_time else 0,
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Agent {self.agent.agent_id} API call failed: {e}")
            
            # Add thought about failed call
            self.agent.add_thought(
                thought_type="api_error",
                content=f"API call failed: {str(e)} (recovery {'successful' if recovery_successful else 'failed'})",
                confidence=0.9
            )
        
        finally:
            api_call.execution_time = time.time() - api_call.timestamp
            
            # Move to history and remove from active
            self.call_history.append(api_call)
            if call_id in self.active_calls:
                del self.active_calls[call_id]
            
            # Keep only last 50 calls in history
            if len(self.call_history) > 50:
                self.call_history = self.call_history[-50:]
            
            # Update agent status back to active
            self.agent.update_status(AgentStatus.ACTIVE)
        
        return api_call
    
    def _select_provider_and_model(
        self, 
        provider_type: Optional[ProviderType] = None,
        model_name: Optional[str] = None,
        is_thinking_session: bool = False
    ) -> Tuple[Optional[ProviderType], Optional[str]]:
        """Select the best available provider and model for this agent"""
        
        # Get the provider registry lazily
        registry = self._get_provider_registry()

        # Build an ordered list of provider candidates. Start with an explicit
        # request, then follow the agent's preference ordering, and finally add
        # any remaining configured providers.
        candidate_order: list[ProviderType] = []
        
        # If OpenRouter-only mode is enabled, force OpenRouter
        if self.openrouter_only:
            # Check circuit breaker for OpenRouter
            if self._is_openrouter_circuit_open():
                if self.openrouter_config and self.openrouter_config.fallback_enabled:
                    logger.warning(f"OpenRouter circuit breaker open for agent {self.agent.agent_id}, using fallback providers")
                    candidate_order = [ProviderType.OPENAI, ProviderType.GOOGLE, ProviderType.XAI]
                else:
                    logger.error(f"OpenRouter circuit breaker open for agent {self.agent.agent_id} and fallback disabled")
                    return None, None
            else:
                candidate_order = [ProviderType.OPENROUTER]
        else:
            if provider_type:
                candidate_order.append(provider_type)

            preferred = [p for p in self.preferred_providers if p not in candidate_order]
            candidate_order.extend(preferred)

            configured_providers = registry.get_available_providers_with_keys()
            for provider in configured_providers:
                if provider not in candidate_order:
                    candidate_order.append(provider)

        if not candidate_order:
            logger.warning(f"No providers available in registry. Agent {self.agent.agent_id} cannot make API calls.")
            return None, None

        for candidate in candidate_order:
            provider = registry.get_provider(candidate)
            if not provider:
                continue

            available_models = registry.get_available_model_names(candidate)
            if not available_models:
                continue

            # For thinking sessions with OpenRouter config, prefer configured models
            if is_thinking_session and candidate == ProviderType.OPENROUTER and self.openrouter_config:
                preferred_models = self.openrouter_config.preferred_models
                for preferred_model in preferred_models:
                    if preferred_model in available_models:
                        return candidate, preferred_model

            if model_name and model_name in available_models:
                return candidate, model_name

            provider_models = [m for m in available_models if provider.validate_model_name(m)]
            if not provider_models:
                continue

            selected_model = self._select_best_model_for_agent(candidate, provider_models, is_thinking_session)
            logger.debug(
                f"Agent {self.agent.agent_id} selecting model {selected_model} via {candidate.value} provider"
            )
            return candidate, selected_model

        logger.warning(f"Agent {self.agent.agent_id} could not find a valid provider/model combination")
        return None, None
    
    def _select_best_model_for_agent(self, provider_type: ProviderType, available_models: List[str], is_thinking_session: bool = False) -> str:
        """Select the best model for this specific agent role."""
        from utils.agent_core import AgentRole

        # Enhanced model preferences for thinking sessions vs regular calls
        if is_thinking_session:
            # Thinking session models prioritize reasoning capability
            role_thinking_preferences = {
                AgentRole.SECURITY_ANALYST: ["openai/o3", "openai/o3-pro", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
                AgentRole.PERFORMANCE_OPTIMIZER: ["google/gemini-2.5-pro", "openai/o3", "anthropic/claude-3-sonnet", "google/gemini-2.5-flash"],
                AgentRole.ARCHITECTURE_REVIEWER: ["openai/o3-pro", "anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro"],
                AgentRole.CODE_QUALITY_INSPECTOR: ["openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet", "openai/o3-mini"],
                AgentRole.DEBUG_SPECIALIST: ["openai/o3", "openai/o3-pro", "anthropic/claude-3-sonnet", "google/gemini-2.5-pro"],
                AgentRole.PLANNING_COORDINATOR: ["openai/o3-pro", "anthropic/claude-3-opus", "google/gemini-2.5-pro", "openai/o3"],
                AgentRole.CONSENSUS_FACILITATOR: ["anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet"],
                AgentRole.GENERALIST: ["google/gemini-2.5-flash", "openai/o3-mini", "anthropic/claude-3-haiku", "google/gemini-2.5-pro"]
            }
            preferred_models = role_thinking_preferences.get(self.agent.role, ["google/gemini-2.5-flash", "openai/o3-mini"])
        else:
            # Regular model preferences
            role_model_preferences = {
                AgentRole.SECURITY_ANALYST: ["openai/o3", "openai/o3-pro", "anthropic/claude-3-sonnet", "google/gemini-2.5-pro"],
                AgentRole.PERFORMANCE_OPTIMIZER: ["google/gemini-2.5-pro", "openai/o3", "anthropic/claude-3-sonnet", "google/gemini-2.5-flash"],
                AgentRole.ARCHITECTURE_REVIEWER: ["openai/o3-pro", "anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro"],
                AgentRole.CODE_QUALITY_INSPECTOR: ["openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet", "openai/o3-mini"],
                AgentRole.DEBUG_SPECIALIST: ["openai/o3", "openai/o3-pro", "anthropic/claude-3-sonnet", "google/gemini-2.5-pro"],
                AgentRole.PLANNING_COORDINATOR: ["openai/o3-pro", "anthropic/claude-3-opus", "google/gemini-2.5-pro", "openai/o3"],
                AgentRole.CONSENSUS_FACILITATOR: ["anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet"],
                AgentRole.GENERALIST: ["google/gemini-2.5-flash", "openai/o3-mini", "anthropic/claude-3-haiku", "google/gemini-2.5-pro"]
            }
            preferred_models = role_model_preferences.get(self.agent.role, ["google/gemini-2.5-flash", "openai/o3-mini"])
        
        preferred_lower = [(p, p.split("/")[-1]) for p in preferred_models]

        for preferred_full, preferred_suffix in preferred_lower:
            for candidate in available_models:
                parts = {candidate.lower(), candidate.split("/")[-1].lower()}
                if preferred_full.lower() in parts or preferred_suffix.lower() in parts:
                    return candidate

        # Fallback to first available model
        return available_models[0] if available_models else "google/gemini-2.5-flash"
    
    async def _execute_api_call(self, api_call: AgentAPICall, max_retries: int) -> str:
        """Execute the actual API call with retries and enhanced circuit breaker handling"""
        api_call.status = "in_progress"
        
        registry = self._get_provider_registry()
        provider = registry.get_provider(api_call.provider_type)
        if not provider:
            raise Exception(f"Provider {api_call.provider_type.value} not available")
        
        last_exception = None
        is_openrouter = api_call.provider_type == ProviderType.OPENROUTER
        is_half_open = getattr(self, 'openrouter_circuit_half_open', False) if is_openrouter else False
        
        # In half-open state, limit retries to test service recovery
        effective_max_retries = 1 if is_half_open else max_retries
        
        # Enhanced error handling with comprehensive retry logic
        error_handler = get_per_core_error_handler()
        
        for attempt in range(effective_max_retries + 1):
            try:
                api_call.retry_count = attempt
                
                # Log retry attempt if not first attempt
                if attempt > 0:
                    log_agent_activity(
                        agent_id=api_call.agent_id,
                        activity_type="api_call_retry",
                        activity_details={
                            "call_id": api_call.call_id,
                            "attempt": attempt + 1,
                            "max_retries": effective_max_retries + 1,
                            "provider": api_call.provider_type.value,
                            "model": api_call.model_name,
                            "last_error": str(last_exception) if last_exception else None
                        }
                    )
                
                # Prepare parameters for API call
                call_params = {
                    "model_name": api_call.model_name,
                    "prompt": api_call.prompt,
                    "thinking_mode": "standard",  # Default thinking mode
                    "temperature": 0.7,  # Default temperature
                    **api_call.parameters
                }
                
                # Apply OpenRouter-specific parameters if configured
                if is_openrouter and self.openrouter_config:
                    # Use role-specific temperature range
                    temp_min, temp_max = self.openrouter_config.temperature_range
                    call_params["temperature"] = temp_min + (temp_max - temp_min) * 0.5  # Use middle of range
                    
                    # Use configured thinking mode
                    if api_call.is_thinking_session:
                        call_params["thinking_mode"] = self.openrouter_config.thinking_mode_default
                
                # Make the API call through the provider
                response = await provider.generate_content(**call_params)
                
                if response and response.strip():
                    # Success - handle circuit breaker state if OpenRouter
                    if is_openrouter and is_half_open:
                        self._handle_half_open_result(True)
                    
                    # Log successful retry if this was a retry
                    if attempt > 0:
                        log_agent_activity(
                            agent_id=api_call.agent_id,
                            activity_type="api_call_retry_success",
                            activity_details={
                                "call_id": api_call.call_id,
                                "successful_attempt": attempt + 1,
                                "total_attempts": attempt + 1,
                                "provider": api_call.provider_type.value,
                                "model": api_call.model_name
                            },
                            success=True
                        )
                    
                    return response.strip()
                else:
                    raise Exception("Empty response from provider")
                    
            except Exception as e:
                last_exception = e
                
                # Use comprehensive error categorization
                error_category = error_handler.categorize_error(e, {
                    "provider": api_call.provider_type.value,
                    "model": api_call.model_name,
                    "attempt": attempt + 1,
                    "is_thinking_session": api_call.is_thinking_session
                })
                
                # Enhanced error logging
                if is_openrouter:
                    openrouter_error_category = self._categorize_openrouter_error(e)
                    logger.warning(f"Agent {api_call.agent_id} OpenRouter API call attempt {attempt + 1} failed: "
                                 f"{openrouter_error_category} ({error_category.value}) - {e}")
                    
                    # Handle half-open state failure
                    if is_half_open:
                        self._handle_half_open_result(False)
                        # Don't retry in half-open state after failure
                        break
                    
                    # Enhanced backoff strategy based on error type
                    if openrouter_error_category == "rate_limit":
                        # Exponential backoff with jitter for rate limits
                        base_backoff = min(60, 2 ** attempt)
                        jitter = base_backoff * 0.1 * (0.5 + 0.5 * hash(api_call.call_id) % 100 / 100)
                        backoff_time = base_backoff + jitter
                        logger.info(f"Rate limited, backing off for {backoff_time:.2f} seconds")
                        await asyncio.sleep(backoff_time)
                    elif openrouter_error_category == "temporary_server_error":
                        # Shorter backoff for temporary errors with jitter
                        base_backoff = min(10, 2 ** attempt)
                        jitter = base_backoff * 0.2 * (0.5 + 0.5 * hash(api_call.call_id) % 100 / 100)
                        backoff_time = base_backoff + jitter
                        logger.info(f"Temporary server error, backing off for {backoff_time:.2f} seconds")
                        await asyncio.sleep(backoff_time)
                    elif openrouter_error_category in ["authentication", "client_error"]:
                        # Don't retry authentication or client errors
                        logger.error(f"Non-retryable error: {openrouter_error_category}")
                        break
                    elif attempt < effective_max_retries:
                        # Standard exponential backoff for other errors with jitter
                        base_backoff = 2 ** attempt
                        jitter = base_backoff * 0.1 * (0.5 + 0.5 * hash(api_call.call_id) % 100 / 100)
                        backoff_time = base_backoff + jitter
                        logger.info(f"API error, backing off for {backoff_time:.2f} seconds")
                        await asyncio.sleep(backoff_time)
                else:
                    logger.warning(f"Agent {api_call.agent_id} API call attempt {attempt + 1} failed: "
                                 f"{error_category.value} - {e}")
                    
                    # Standard backoff for non-OpenRouter providers
                    if attempt < effective_max_retries:
                        # Don't retry certain error types
                        if error_category in [ErrorCategory.CONFIGURATION_ERROR]:
                            logger.error(f"Non-retryable error category: {error_category.value}")
                            break
                        
                        base_backoff = 2 ** attempt
                        jitter = base_backoff * 0.1 * (0.5 + 0.5 * hash(api_call.call_id) % 100 / 100)
                        backoff_time = base_backoff + jitter
                        logger.info(f"API error, backing off for {backoff_time:.2f} seconds")
                        await asyncio.sleep(backoff_time)
                    
                    if attempt < effective_max_retries:
                        # Standard exponential backoff
                        await asyncio.sleep(2 ** attempt)
        
        # All retries failed - handle circuit breaker state if OpenRouter
        if is_openrouter and is_half_open:
            self._handle_half_open_result(False)
        
        # All retries failed
        raise last_exception or Exception("API call failed after all retries")
    
    async def _wait_for_call(self, call_id: str, timeout: float = 30.0) -> None:
        """Wait for a specific API call to complete"""
        start_time = time.time()
        
        while call_id in self.active_calls and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if call_id in self.active_calls:
            logger.warning(f"Timeout waiting for API call {call_id} to complete")
    
    def get_call_history(self, limit: int = 10) -> List[AgentAPICall]:
        """Get recent API call history for this agent"""
        return self.call_history[-limit:] if self.call_history else []
    
    def get_active_calls(self) -> List[AgentAPICall]:
        """Get currently active API calls for this agent"""
        return list(self.active_calls.values())
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get API call statistics for this agent"""
        total_calls = len(self.call_history)
        successful_calls = len([c for c in self.call_history if c.status == "completed"])
        failed_calls = len([c for c in self.call_history if c.status == "failed"])
        
        if total_calls > 0:
            avg_execution_time = sum(c.execution_time for c in self.call_history) / total_calls
            success_rate = successful_calls / total_calls
        else:
            avg_execution_time = 0.0
            success_rate = 0.0
        
        return {
            "agent_id": self.agent.agent_id,
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "active_calls": len(self.active_calls),
            "preferred_providers": [p.value for p in self.preferred_providers],
            "openrouter_only": self.openrouter_only
        }
    
    def reset_statistics(self) -> None:
        """Reset API call statistics and history for this agent"""
        logger.info(f"Resetting statistics for agent {self.agent.agent_id}")
        
        # Clear call history
        self.call_history.clear()
        
        # Clear active calls (but don't interrupt them)
        # Just clear the tracking - ongoing calls will complete naturally
        self.active_calls.clear()
        
        # Reset OpenRouter-specific statistics if configured
        if self.openrouter_only and self.openrouter_config:
            self.openrouter_failure_count = 0
            self.openrouter_circuit_open = False
            self.openrouter_circuit_open_time = 0.0
            self.openrouter_call_times.clear()
        
        logger.debug(f"Statistics reset completed for agent {self.agent.agent_id}")
    
    def configure_openrouter_only(self, api_key: str, config: Optional[OpenRouterConfig] = None) -> None:
        """
        Configure this agent to use OpenRouter exclusively for all API calls
        
        Args:
            api_key: OpenRouter API key for this agent
            config: Optional OpenRouter configuration, uses defaults if not provided
        """
        self.openrouter_only = True
        
        # Create default config if not provided
        if config is None:
            from utils.agent_core import AgentRole
            
            # Role-specific model preferences for OpenRouter
            role_model_preferences = {
                AgentRole.SECURITY_ANALYST: ["openai/o3", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
                AgentRole.PERFORMANCE_OPTIMIZER: ["google/gemini-2.5-pro", "openai/o3", "anthropic/claude-3-sonnet"],
                AgentRole.ARCHITECTURE_REVIEWER: ["openai/o3-pro", "anthropic/claude-3-opus", "openai/o3"],
                AgentRole.CODE_QUALITY_INSPECTOR: ["openai/o3", "google/gemini-2.5-pro", "anthropic/claude-3-sonnet"],
                AgentRole.DEBUG_SPECIALIST: ["openai/o3", "anthropic/claude-3-sonnet", "google/gemini-2.5-pro"],
                AgentRole.PLANNING_COORDINATOR: ["openai/o3-pro", "anthropic/claude-3-opus", "google/gemini-2.5-pro"],
                AgentRole.CONSENSUS_FACILITATOR: ["anthropic/claude-3-opus", "openai/o3", "google/gemini-2.5-pro"],
                AgentRole.GENERALIST: ["google/gemini-2.5-flash", "openai/o3-mini", "anthropic/claude-3-haiku"]
            }
            
            preferred_models = role_model_preferences.get(self.agent.role, ["google/gemini-2.5-flash", "openai/o3-mini"])
            
            config = OpenRouterConfig(
                api_key=api_key,
                preferred_models=preferred_models,
                rate_limit_per_minute=60,
                max_concurrent_calls=3,
                thinking_mode_default="high"
            )
        
        self.openrouter_config = config
        
        # Update preferred providers to prioritize OpenRouter
        self.preferred_providers = [ProviderType.OPENROUTER]
        
        # Reset circuit breaker state
        self.openrouter_failure_count = 0
        self.openrouter_circuit_open = False
        self.openrouter_circuit_open_time = 0.0
        
        logger.info(f"Agent {self.agent.agent_id} configured for OpenRouter-only operation with {len(config.preferred_models)} preferred models")
    
    async def make_thinking_session_call(
        self, 
        prompt: str, 
        thinking_mode: Optional[str] = None
    ) -> "AgentAPICall":
        """
        Make a thinking session API call optimized for OpenRouter
        
        Args:
            prompt: The thinking prompt
            thinking_mode: Thinking mode intensity ("low", "standard", "high"), uses config default if not provided
            
        Returns:
            AgentAPICall object
        """
        # Use configured thinking mode if not specified
        if thinking_mode is None and self.openrouter_config:
            thinking_mode = self.openrouter_config.thinking_mode_default
        elif thinking_mode is None:
            thinking_mode = "high"
        
        # Optimize parameters for thinking sessions based on OpenRouter config
        temperature = 0.7  # Default
        if self.openrouter_config:
            temp_min, temp_max = self.openrouter_config.temperature_range
            if thinking_mode == "high":
                temperature = temp_max
            elif thinking_mode == "low":
                temperature = temp_min
            else:  # standard
                temperature = (temp_min + temp_max) / 2
        
        thinking_parameters = {
            "thinking_mode": thinking_mode,
            "temperature": temperature,
            "max_tokens": 4000 if thinking_mode == "high" else 2000
        }
        
        # Force OpenRouter if configured
        provider_type = ProviderType.OPENROUTER if self.openrouter_only else None
        
        # Make the API call with thinking session flag
        api_call = await self.make_api_call(
            prompt=prompt,
            provider_type=provider_type,
            parameters=thinking_parameters,
            is_thinking_session=True
        )
        
        return api_call
    
    def get_openrouter_usage_stats(self) -> Dict[str, Any]:
        """
        Get OpenRouter-specific usage statistics for this agent
        
        Returns:
            Dictionary with OpenRouter usage metrics
        """
        openrouter_calls = [
            call for call in self.call_history 
            if call.provider_type == ProviderType.OPENROUTER
        ]
        
        thinking_session_calls = [
            call for call in openrouter_calls
            if call.is_thinking_session
        ]
        
        total_openrouter_calls = len(openrouter_calls)
        successful_openrouter_calls = len([c for c in openrouter_calls if c.status == "completed"])
        total_thinking_sessions = len(thinking_session_calls)
        successful_thinking_sessions = len([c for c in thinking_session_calls if c.status == "completed"])
        
        if total_openrouter_calls > 0:
            openrouter_success_rate = successful_openrouter_calls / total_openrouter_calls
            avg_openrouter_time = sum(c.execution_time for c in openrouter_calls) / total_openrouter_calls
        else:
            openrouter_success_rate = 0.0
            avg_openrouter_time = 0.0
        
        if total_thinking_sessions > 0:
            thinking_success_rate = successful_thinking_sessions / total_thinking_sessions
            avg_thinking_time = sum(c.execution_time for c in thinking_session_calls) / total_thinking_sessions
        else:
            thinking_success_rate = 0.0
            avg_thinking_time = 0.0
        
        return {
            "agent_id": self.agent.agent_id,
            "openrouter_only": self.openrouter_only,
            "total_openrouter_calls": total_openrouter_calls,
            "successful_openrouter_calls": successful_openrouter_calls,
            "openrouter_success_rate": openrouter_success_rate,
            "avg_openrouter_execution_time": avg_openrouter_time,
            "total_thinking_sessions": total_thinking_sessions,
            "successful_thinking_sessions": successful_thinking_sessions,
            "thinking_success_rate": thinking_success_rate,
            "avg_thinking_execution_time": avg_thinking_time,
            "openrouter_configured": self.openrouter_config is not None,
            "circuit_breaker_open": self.openrouter_circuit_open,
            "failure_count": self.openrouter_failure_count,
            "rate_limit_calls_last_minute": len([t for t in self.openrouter_call_times if time.time() - t < 60])
        }
    
    async def _apply_openrouter_rate_limiting(self) -> None:
        """Apply OpenRouter-specific rate limiting for this agent"""
        if not self.openrouter_config:
            return
        
        current_time = time.time()
        
        # Clean up old call times (older than 1 minute)
        self.openrouter_call_times = [t for t in self.openrouter_call_times if current_time - t < 60]
        
        # Check if we're at the rate limit
        if len(self.openrouter_call_times) >= self.openrouter_config.rate_limit_per_minute:
            # Calculate how long to wait
            oldest_call = min(self.openrouter_call_times)
            wait_time = 60 - (current_time - oldest_call)
            
            if wait_time > 0:
                logger.info(f"Agent {self.agent.agent_id} rate limited, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this call time
        self.openrouter_call_times.append(current_time)
    
    def _is_openrouter_circuit_open(self) -> bool:
        """Check if the OpenRouter circuit breaker is open"""
        if not self.openrouter_circuit_open:
            return False
        
        if not self.openrouter_config:
            return False
        
        # Check if circuit breaker timeout has passed
        current_time = time.time()
        if current_time - self.openrouter_circuit_open_time > self.openrouter_config.circuit_breaker_timeout:
            # Reset circuit breaker
            self.openrouter_circuit_open = False
            self.openrouter_failure_count = 0
            logger.info(f"OpenRouter circuit breaker reset for agent {self.agent.agent_id}")
            return False
        
        return True
    
    def _handle_openrouter_success(self) -> None:
        """Handle successful OpenRouter API call"""
        # Reset failure count on success
        if self.openrouter_failure_count > 0:
            logger.debug(f"OpenRouter success for agent {self.agent.agent_id}, resetting failure count")
            self.openrouter_failure_count = 0
    
    def _handle_openrouter_failure(self, error: Exception) -> None:
        """Handle failed OpenRouter API call with enhanced error categorization"""
        if not self.openrouter_config:
            return
        
        # Categorize the error to determine appropriate response
        error_category = self._categorize_openrouter_error(error)
        
        # Only count certain types of errors toward circuit breaker
        if error_category in ["api_error", "timeout", "server_error", "authentication"]:
            self.openrouter_failure_count += 1
            logger.warning(f"OpenRouter failure #{self.openrouter_failure_count} for agent {self.agent.agent_id}: "
                         f"{error_category} - {error}")
            
            # Check if we should open the circuit breaker
            if self.openrouter_failure_count >= self.openrouter_config.circuit_breaker_threshold:
                self._open_circuit_breaker(error_category)
        else:
            # For rate limiting and temporary errors, don't count toward circuit breaker
            logger.info(f"OpenRouter {error_category} for agent {self.agent.agent_id}: {error}")
    
    def _categorize_openrouter_error(self, error: Exception) -> str:
        """
        Categorize OpenRouter API errors for appropriate handling
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error category string
        """
        error_str = str(error).lower()
        
        # Rate limiting errors (don't count toward circuit breaker)
        if any(term in error_str for term in ["rate limit", "too many requests", "429"]):
            return "rate_limit"
        
        # Temporary server errors (don't count toward circuit breaker immediately)
        if any(term in error_str for term in ["502", "503", "504", "temporary", "maintenance"]):
            return "temporary_server_error"
        
        # Authentication/authorization errors (count toward circuit breaker)
        if any(term in error_str for term in ["401", "403", "unauthorized", "forbidden", "api key"]):
            return "authentication"
        
        # Timeout errors (count toward circuit breaker)
        if any(term in error_str for term in ["timeout", "timed out", "connection timeout"]):
            return "timeout"
        
        # Server errors (count toward circuit breaker)
        if any(term in error_str for term in ["500", "internal server error", "server error"]):
            return "server_error"
        
        # Client errors (count toward circuit breaker)
        if any(term in error_str for term in ["400", "bad request", "invalid"]):
            return "client_error"
        
        # Network errors (count toward circuit breaker)
        if any(term in error_str for term in ["connection", "network", "dns", "resolve"]):
            return "network_error"
        
        # Default to API error
        return "api_error"
    
    def cleanup_connections(self) -> None:
        """
        Cleanup API client connections and resources for graceful shutdown
        """
        logger.debug(f"Cleaning up API client connections for agent {self.agent.agent_id}")
        
        try:
            # Cancel any active API calls
            if self.active_calls:
                logger.debug(f"Cancelling {len(self.active_calls)} active API calls for agent {self.agent.agent_id}")
                for call_id, api_call in list(self.active_calls.items()):
                    try:
                        api_call.status = "cancelled"
                        api_call.error = "System shutdown - connection cleanup"
                        api_call.execution_time = time.time() - api_call.timestamp
                    except Exception as e:
                        logger.warning(f"Error updating cancelled API call {call_id}: {e}")
                
                self.active_calls.clear()
            
            # Reset OpenRouter-specific state if configured
            if self.openrouter_config:
                logger.debug(f"Cleaning up OpenRouter state for agent {self.agent.agent_id}")
                
                # Clear rate limiting tracking
                if hasattr(self, 'openrouter_call_times'):
                    self.openrouter_call_times.clear()
                
                # Reset circuit breaker state
                self.openrouter_circuit_open = False
                self.openrouter_failure_count = 0
                self.openrouter_circuit_open_time = 0.0
                
                # Reset half-open state if it exists
                if hasattr(self, 'openrouter_circuit_half_open'):
                    self.openrouter_circuit_half_open = False
                
                logger.debug(f"OpenRouter state cleaned up for agent {self.agent.agent_id}")
            
            # Clear call history but preserve statistics for final reporting
            # Don't clear self.call_history as it may be needed for final metrics
            
            # Reset concurrent call tracking
            self.active_calls.clear()
            
            # Add a final thought about cleanup
            if hasattr(self.agent, 'add_thought'):
                self.agent.add_thought(
                    thought_type="system",
                    content=f"API client connections cleaned up during system shutdown. "
                            f"Total calls made: {len(self.call_history)}, "
                            f"OpenRouter configured: {self.openrouter_config is not None}. "
                            f"Cleanup metadata: cleanup=True, shutdown=True"
                )
            
            logger.info(f"API client cleanup completed for agent {self.agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Error during API client cleanup for agent {self.agent.agent_id}: {e}")
            raise
    
    def _open_circuit_breaker(self, error_category: str) -> None:
        """
        Open the circuit breaker with enhanced logging and notification
        
        Args:
            error_category: The category of error that triggered the circuit breaker
        """
        self.openrouter_circuit_open = True
        self.openrouter_circuit_open_time = time.time()
        self.openrouter_circuit_trigger_reason = error_category
        
        # Calculate backoff time based on error category
        base_timeout = self.openrouter_config.circuit_breaker_timeout
        if error_category == "authentication":
            # Longer timeout for auth errors
            self.openrouter_circuit_timeout = base_timeout * 2
        elif error_category == "rate_limit":
            # Shorter timeout for rate limiting
            self.openrouter_circuit_timeout = base_timeout * 0.5
        else:
            self.openrouter_circuit_timeout = base_timeout
        
        logger.error(f"OpenRouter circuit breaker opened for agent {self.agent.agent_id} "
                    f"after {self.openrouter_failure_count} failures (trigger: {error_category}). "
                    f"Will retry in {self.openrouter_circuit_timeout:.0f} seconds")
        
        # Notify the per-core agent manager about circuit breaker activation
        self._notify_circuit_breaker_opened(error_category)
    
    def _notify_circuit_breaker_opened(self, error_category: str) -> None:
        """
        Notify the system about circuit breaker activation
        
        Args:
            error_category: The category of error that triggered the circuit breaker
        """
        try:
            # This would integrate with the per-core agent manager's graceful degradation
            from utils.agent_communication import get_agent_communication_system
            
            comm_system = get_agent_communication_system()
            if comm_system:
                # Send system message about circuit breaker
                comm_system.send_message(
                    from_agent=self.agent.agent_id,
                    to_agent="system",
                    message_type="circuit_breaker_opened",
                    content={
                        "agent_id": self.agent.agent_id,
                        "error_category": error_category,
                        "failure_count": self.openrouter_failure_count,
                        "timeout_duration": getattr(self, 'openrouter_circuit_timeout', 
                                                  self.openrouter_config.circuit_breaker_timeout)
                    },
                    priority=8
                )
        except Exception as e:
            logger.error(f"Failed to notify system about circuit breaker: {e}")
    
    def _is_openrouter_circuit_open(self) -> bool:
        """Enhanced circuit breaker status check with half-open state"""
        if not self.openrouter_circuit_open:
            return False
        
        if not self.openrouter_config:
            return False
        
        current_time = time.time()
        timeout_duration = getattr(self, 'openrouter_circuit_timeout', 
                                 self.openrouter_config.circuit_breaker_timeout)
        
        # Check if circuit breaker timeout has passed
        if current_time - self.openrouter_circuit_open_time > timeout_duration:
            # Enter half-open state
            self._enter_half_open_state()
            return False
        
        return True
    
    def _enter_half_open_state(self) -> None:
        """
        Enter half-open state to test if the service has recovered
        """
        self.openrouter_circuit_open = False
        self.openrouter_circuit_half_open = True
        self.openrouter_half_open_time = time.time()
        self.openrouter_half_open_attempts = 0
        
        logger.info(f"OpenRouter circuit breaker entering half-open state for agent {self.agent.agent_id}")
    
    def _handle_half_open_result(self, success: bool) -> None:
        """
        Handle the result of an API call in half-open state
        
        Args:
            success: Whether the API call was successful
        """
        if not hasattr(self, 'openrouter_circuit_half_open') or not self.openrouter_circuit_half_open:
            return
        
        self.openrouter_half_open_attempts += 1
        
        if success:
            # Success in half-open state - close the circuit breaker
            self._close_circuit_breaker()
        else:
            # Failure in half-open state - reopen circuit breaker
            if self.openrouter_half_open_attempts >= 3:  # Allow up to 3 attempts in half-open
                self._reopen_circuit_breaker()
    
    def _close_circuit_breaker(self) -> None:
        """
        Close the circuit breaker after successful recovery
        """
        self.openrouter_circuit_open = False
        self.openrouter_circuit_half_open = False
        self.openrouter_failure_count = 0
        
        # Clean up half-open state attributes
        if hasattr(self, 'openrouter_half_open_time'):
            delattr(self, 'openrouter_half_open_time')
        if hasattr(self, 'openrouter_half_open_attempts'):
            delattr(self, 'openrouter_half_open_attempts')
        if hasattr(self, 'openrouter_circuit_timeout'):
            delattr(self, 'openrouter_circuit_timeout')
        if hasattr(self, 'openrouter_circuit_trigger_reason'):
            delattr(self, 'openrouter_circuit_trigger_reason')
        
        logger.info(f"OpenRouter circuit breaker closed for agent {self.agent.agent_id} - service recovered")
        
        # Notify system about recovery
        self._notify_circuit_breaker_closed()
    
    def _reopen_circuit_breaker(self) -> None:
        """
        Reopen the circuit breaker after failed recovery attempts
        """
        self.openrouter_circuit_open = True
        self.openrouter_circuit_half_open = False
        self.openrouter_circuit_open_time = time.time()
        
        # Increase timeout for subsequent attempts
        base_timeout = self.openrouter_config.circuit_breaker_timeout
        self.openrouter_circuit_timeout = min(base_timeout * 2, 1800)  # Max 30 minutes
        
        logger.warning(f"OpenRouter circuit breaker reopened for agent {self.agent.agent_id} "
                      f"after failed recovery attempts. Next retry in {self.openrouter_circuit_timeout:.0f} seconds")
    
    def _notify_circuit_breaker_closed(self) -> None:
        """
        Notify the system about circuit breaker recovery
        """
        try:
            from utils.agent_communication import get_agent_communication_system
            
            comm_system = get_agent_communication_system()
            if comm_system:
                comm_system.send_message(
                    from_agent=self.agent.agent_id,
                    to_agent="system",
                    message_type="circuit_breaker_closed",
                    content={
                        "agent_id": self.agent.agent_id,
                        "recovery_successful": True,
                        "downtime_duration": time.time() - self.openrouter_circuit_open_time
                    },
                    priority=6
                )
        except Exception as e:
            logger.error(f"Failed to notify system about circuit breaker recovery: {e}")
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """
        Get detailed circuit breaker status information
        
        Returns:
            Dictionary with circuit breaker status details
        """
        status = {
            "circuit_open": self.openrouter_circuit_open,
            "failure_count": self.openrouter_failure_count,
            "threshold": self.openrouter_config.circuit_breaker_threshold if self.openrouter_config else 5,
            "half_open": getattr(self, 'openrouter_circuit_half_open', False),
            "last_failure_time": getattr(self, 'openrouter_circuit_open_time', 0),
            "trigger_reason": getattr(self, 'openrouter_circuit_trigger_reason', None),
            "timeout_duration": getattr(self, 'openrouter_circuit_timeout', 
                                      self.openrouter_config.circuit_breaker_timeout if self.openrouter_config else 300),
            "time_until_retry": 0
        }
        
        if status["circuit_open"] and status["last_failure_time"] > 0:
            current_time = time.time()
            elapsed = current_time - status["last_failure_time"]
            status["time_until_retry"] = max(0, status["timeout_duration"] - elapsed)
        
        return status
    
    async def create_and_execute_thinking_session(
        self,
        thinking_prompt: str,
        thinking_mode: Optional[str] = None,
        timeout_seconds: float = 300.0,
        priority: int = 5
    ) -> "AgentThinkingSession":
        """
        Create and execute a thinking session with full coordination support
        
        Args:
            thinking_prompt: The prompt for the thinking session
            thinking_mode: Thinking mode intensity ("low", "standard", "high")
            timeout_seconds: Session timeout in seconds
            priority: Session priority (1-10)
            
        Returns:
            AgentThinkingSession instance
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        # Get the thinking session coordinator
        coordinator = get_thinking_session_coordinator()
        
        # Create the thinking session
        session = coordinator.create_individual_session(
            agent=self.agent,
            thinking_prompt=thinking_prompt,
            model_used="auto",  # Will be determined during execution
            thinking_mode=thinking_mode or (self.openrouter_config.thinking_mode_default if self.openrouter_config else "high"),
            timeout_seconds=timeout_seconds,
            priority=priority
        )
        
        # Start the session
        if not coordinator.start_session(session.session_id):
            logger.error(f"Failed to start thinking session {session.session_id}")
            coordinator.fail_session(session.session_id, "Failed to start session")
            return session
        
        try:
            # Execute the thinking session API call
            api_call = await self.make_thinking_session_call(thinking_prompt, thinking_mode)
            
            # Update session with API call details
            session.api_call_id = api_call.call_id
            session.model_used = api_call.model_name
            session.retry_count = api_call.retry_count
            
            if api_call.status == "completed" and api_call.response:
                # Extract token usage and cost from OpenRouter usage if available
                tokens_used = api_call.openrouter_usage.get("tokens", 0)
                cost_estimate = api_call.openrouter_usage.get("cost", 0.0)
                
                # Complete the session successfully
                coordinator.complete_session(
                    session.session_id,
                    api_call.response,
                    tokens_used,
                    cost_estimate,
                    api_call.openrouter_usage
                )
            else:
                # Session failed
                error_msg = api_call.error or "API call failed without specific error"
                coordinator.fail_session(session.session_id, error_msg)
        
        except Exception as e:
            logger.error(f"Error executing thinking session {session.session_id}: {e}")
            coordinator.fail_session(session.session_id, str(e))
        
        return session
    
    def participate_in_synchronized_session(
        self,
        synchronized_request: "SynchronizedThinkingRequest"
    ) -> Optional["AgentThinkingSession"]:
        """
        Participate in a synchronized thinking session with other agents
        
        Args:
            synchronized_request: The synchronized thinking request
            
        Returns:
            AgentThinkingSession instance if participation was successful, None otherwise
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        # Check if this agent is supposed to participate
        if self.agent.agent_id not in synchronized_request.participating_agent_ids:
            logger.warning(f"Agent {self.agent.agent_id} not in participant list for synchronized session {synchronized_request.request_id}")
            return None
        
        # Get the thinking session coordinator
        coordinator = get_thinking_session_coordinator()
        
        # Create synchronized sessions for all participants
        sessions = coordinator.create_synchronized_session(synchronized_request)
        
        # Get this agent's session
        agent_session = sessions.get(self.agent.agent_id)
        if not agent_session:
            logger.error(f"No session created for agent {self.agent.agent_id} in synchronized request {synchronized_request.request_id}")
            return None
        
        # Update session with agent's core ID
        agent_session.core_id = self.agent.core_id
        
        # Start this agent's session
        if not coordinator.start_session(agent_session.session_id):
            logger.error(f"Failed to start synchronized session {agent_session.session_id} for agent {self.agent.agent_id}")
            return agent_session
        
        # Execute the thinking session asynchronously
        # Note: This should be called in an async context
        asyncio.create_task(self._execute_synchronized_thinking_session(agent_session, coordinator))
        
        return agent_session
    
    async def _execute_synchronized_thinking_session(
        self,
        session: "AgentThinkingSession",
        coordinator: "ThinkingSessionCoordinator"
    ) -> None:
        """
        Execute a synchronized thinking session
        
        Args:
            session: The thinking session to execute
            coordinator: The thinking session coordinator
        """
        try:
            # Execute the thinking session API call
            api_call = await self.make_thinking_session_call(
                session.thinking_prompt,
                session.thinking_mode
            )
            
            # Update session with API call details
            session.api_call_id = api_call.call_id
            session.model_used = api_call.model_name
            session.retry_count = api_call.retry_count
            
            if api_call.status == "completed" and api_call.response:
                # Extract token usage and cost from OpenRouter usage if available
                tokens_used = api_call.openrouter_usage.get("tokens", 0)
                cost_estimate = api_call.openrouter_usage.get("cost", 0.0)
                
                # Complete the session successfully
                coordinator.complete_session(
                    session.session_id,
                    api_call.response,
                    tokens_used,
                    cost_estimate,
                    api_call.openrouter_usage
                )
                
                logger.info(f"Completed synchronized thinking session {session.session_id} for agent {self.agent.agent_id}")
            else:
                # Session failed
                error_msg = api_call.error or "API call failed without specific error"
                coordinator.fail_session(session.session_id, error_msg)
                logger.error(f"Failed synchronized thinking session {session.session_id} for agent {self.agent.agent_id}: {error_msg}")
        
        except Exception as e:
            logger.error(f"Error executing synchronized thinking session {session.session_id} for agent {self.agent.agent_id}: {e}")
            coordinator.fail_session(session.session_id, str(e))
    
    def get_thinking_session_statistics(self) -> Dict[str, Any]:
        """
        Get thinking session statistics for this agent
        
        Returns:
            Dictionary with thinking session metrics
        """
        from utils.agent_thinking_session import get_thinking_session_coordinator
        
        coordinator = get_thinking_session_coordinator()
        
        # Get all sessions for this agent
        agent_sessions = coordinator.get_agent_sessions(self.agent.agent_id, include_completed=True)
        
        # Separate by type
        individual_sessions = [s for s in agent_sessions if s.session_type.value == "individual"]
        synchronized_sessions = [s for s in agent_sessions if s.session_type.value in ["synchronized", "collaborative", "consensus"]]
        
        # Calculate statistics
        total_sessions = len(agent_sessions)
        completed_sessions = [s for s in agent_sessions if s.is_completed()]
        successful_sessions = [s for s in completed_sessions if s.status.value == "completed"]
        
        if total_sessions > 0:
            success_rate = len(successful_sessions) / total_sessions
        else:
            success_rate = 0.0
        
        if completed_sessions:
            avg_duration = sum(s.get_duration() for s in completed_sessions) / len(completed_sessions)
            total_tokens = sum(s.tokens_used for s in completed_sessions)
            total_cost = sum(s.cost_estimate for s in completed_sessions)
        else:
            avg_duration = 0.0
            total_tokens = 0
            total_cost = 0.0
        
        return {
            "agent_id": self.agent.agent_id,
            "total_thinking_sessions": total_sessions,
            "individual_sessions": len(individual_sessions),
            "synchronized_sessions": len(synchronized_sessions),
            "completed_sessions": len(completed_sessions),
            "successful_sessions": len(successful_sessions),
            "success_rate": success_rate,
            "avg_session_duration": avg_duration,
            "total_tokens_used": total_tokens,
            "total_cost_estimate": total_cost,
            "active_sessions": len([s for s in agent_sessions if s.is_active()]),
            "recent_session_ids": [s.session_id for s in agent_sessions[-5:]]  # Last 5 sessions
        }
