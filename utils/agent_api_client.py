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

logger = logging.getLogger(__name__)


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
            "has_error": self.error is not None
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
        self.provider_registry = ModelProviderRegistry()
        
        # Rate limiting per agent
        self.last_call_time = 0.0
        self.min_call_interval = 1.0  # Minimum seconds between calls
        
        # Preferred providers for this agent
        self.preferred_providers = self._determine_preferred_providers()
        
        logger.info(f"Agent API client created for {agent.agent_id} with preferred providers: {[p.value for p in self.preferred_providers]}")
    
    def _determine_preferred_providers(self) -> List[ProviderType]:
        """Determine preferred providers based on agent role"""
        from utils.agent_core import AgentRole
        
        # Different agent roles may prefer different providers
        role_preferences = {
            AgentRole.SECURITY_ANALYST: [ProviderType.OPENAI, ProviderType.GOOGLE, ProviderType.XAI],
            AgentRole.PERFORMANCE_OPTIMIZER: [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.CUSTOM],
            AgentRole.ARCHITECTURE_REVIEWER: [ProviderType.OPENAI, ProviderType.GOOGLE, ProviderType.DIAL],
            AgentRole.CODE_QUALITY_INSPECTOR: [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.XAI],
            AgentRole.DEBUG_SPECIALIST: [ProviderType.OPENAI, ProviderType.XAI, ProviderType.CUSTOM],
            AgentRole.PLANNING_COORDINATOR: [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.DIAL],
            AgentRole.CONSENSUS_FACILITATOR: [ProviderType.OPENAI, ProviderType.GOOGLE, ProviderType.OPENROUTER],
            AgentRole.GENERALIST: [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER]
        }
        
        return role_preferences.get(self.agent.role, [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.OPENROUTER])
    
    async def make_api_call(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        provider_type: Optional[ProviderType] = None,
        parameters: Optional[Dict[str, Any]] = None,
        max_retries: int = 2
    ) -> AgentAPICall:
        """
        Make an API call directly from this agent
        
        Args:
            prompt: The prompt to send to the AI model
            model_name: Specific model to use (optional)
            provider_type: Specific provider to use (optional) 
            parameters: Additional parameters for the API call
            max_retries: Maximum number of retries on failure
            
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
        selected_provider, selected_model = self._select_provider_and_model(provider_type, model_name)
        
        if not selected_provider:
            error_msg = f"No available providers for agent {self.agent.agent_id}"
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
            status="pending"
        )
        
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
            # Execute the API call
            response = await self._execute_api_call(api_call, max_retries)
            api_call.response = response
            api_call.status = "completed"
            
            # Add thought about successful call
            self.agent.add_thought(
                thought_type="api_response",
                content=f"Received response from {selected_provider.value}: {response[:100]}...",
                confidence=0.9
            )
            
        except Exception as e:
            api_call.error = str(e)
            api_call.status = "failed"
            
            logger.error(f"Agent {self.agent.agent_id} API call failed: {e}")
            
            # Add thought about failed call
            self.agent.add_thought(
                thought_type="api_error",
                content=f"API call failed: {str(e)}",
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
        model_name: Optional[str] = None
    ) -> Tuple[Optional[ProviderType], Optional[str]]:
        """Select the best available provider and model for this agent"""
        
        # If provider is specified, try to use it
        if provider_type:
            provider = self.provider_registry.get_provider(provider_type)
            if provider:
                # Get available models for this provider
                available_models = self.provider_registry.get_available_model_names()
                if model_name and model_name in available_models:
                    return provider_type, model_name
                elif available_models:
                    # Use first available model from this provider
                    provider_models = [m for m in available_models if provider.supports_model(m)]
                    if provider_models:
                        return provider_type, provider_models[0]
        
        # Try preferred providers in order
        for pref_provider in self.preferred_providers:
            provider = self.provider_registry.get_provider(pref_provider)
            if provider:
                available_models = self.provider_registry.get_available_model_names()
                provider_models = [m for m in available_models if provider.supports_model(m)]
                
                if model_name and model_name in provider_models:
                    return pref_provider, model_name
                elif provider_models:
                    # Select best model for this agent role
                    selected_model = self._select_best_model_for_agent(provider_models)
                    return pref_provider, selected_model
        
        return None, None
    
    def _select_best_model_for_agent(self, available_models: List[str]) -> str:
        """Select the best model for this specific agent role"""
        from utils.agent_core import AgentRole
        
        # Model preferences based on agent role
        role_model_preferences = {
            AgentRole.SECURITY_ANALYST: ["gpt-4", "gpt-4-turbo", "gemini-pro", "gpt-3.5-turbo"],
            AgentRole.PERFORMANCE_OPTIMIZER: ["gemini-pro", "gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"],
            AgentRole.ARCHITECTURE_REVIEWER: ["gpt-4", "claude-3-opus", "gemini-pro", "gpt-4-turbo"],
            AgentRole.CODE_QUALITY_INSPECTOR: ["gpt-4", "gemini-pro", "claude-3-sonnet", "gpt-3.5-turbo"],
            AgentRole.DEBUG_SPECIALIST: ["gpt-4", "gpt-4-turbo", "claude-3-sonnet", "gemini-pro"],
            AgentRole.PLANNING_COORDINATOR: ["gpt-4", "claude-3-opus", "gemini-pro", "gpt-4-turbo"],
            AgentRole.CONSENSUS_FACILITATOR: ["claude-3-opus", "gpt-4", "gemini-pro", "gpt-4-turbo"],
            AgentRole.GENERALIST: ["gpt-3.5-turbo", "gemini-pro", "gpt-4", "claude-3-sonnet"]
        }
        
        preferred_models = role_model_preferences.get(self.agent.role, ["gpt-3.5-turbo", "gemini-pro"])
        
        # Find first preferred model that's available
        for preferred in preferred_models:
            for available in available_models:
                if preferred.lower() in available.lower():
                    return available
        
        # Fallback to first available model
        return available_models[0] if available_models else "gpt-3.5-turbo"
    
    async def _execute_api_call(self, api_call: AgentAPICall, max_retries: int) -> str:
        """Execute the actual API call with retries"""
        api_call.status = "in_progress"
        
        provider = self.provider_registry.get_provider(api_call.provider_type)
        if not provider:
            raise Exception(f"Provider {api_call.provider_type.value} not available")
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                api_call.retry_count = attempt
                
                # Prepare parameters for API call
                call_params = {
                    "model_name": api_call.model_name,
                    "prompt": api_call.prompt,
                    "thinking_mode": "standard",  # Default thinking mode
                    "temperature": 0.7,  # Default temperature
                    **api_call.parameters
                }
                
                # Make the API call through the provider
                response = await provider.generate_content(**call_params)
                
                if response and response.strip():
                    return response.strip()
                else:
                    raise Exception("Empty response from provider")
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Agent {api_call.agent_id} API call attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
        
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
            "preferred_providers": [p.value for p in self.preferred_providers]
        }