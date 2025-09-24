"""
Orchestration Request Models

This module defines the request models for the Zen Intelligence API,
providing OpenAI-compatible structures with advanced orchestration features.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""
    
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name for the message sender")


class OrchestrationPreferences(BaseModel):
    """User preferences for orchestration behavior."""
    
    preferred_models: Optional[List[str]] = Field(None, description="Preferred models in order of preference")
    orchestration_strategy: Optional[str] = Field(None, description="Preferred strategy: single, parallel, sequential, consensus")
    quality_threshold: Optional[float] = Field(None, description="Minimum quality threshold", ge=0.0, le=1.0)
    max_latency: Optional[float] = Field(None, description="Maximum acceptable latency in seconds")
    max_cost: Optional[float] = Field(None, description="Maximum acceptable cost in USD")
    require_consensus: Optional[bool] = Field(False, description="Whether to require consensus from multiple models")
    enable_learning: Optional[bool] = Field(True, description="Whether to use performance learning")


class OrchestrationRequest(BaseModel):
    """Request for intelligent orchestration."""
    
    # OpenAI-compatible fields
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Optional preferred model (can be overridden by intelligence)")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(1.0, description="Top-p sampling parameter", ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    
    # Zen-specific orchestration fields
    orchestration_preferences: Optional[OrchestrationPreferences] = Field(
        None, description="Advanced orchestration preferences"
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        None, description="Extended conversation history for context"
    )
    domain_hint: Optional[str] = Field(
        None, description="Hint about the query domain for better routing"
    )
    complexity_hint: Optional[str] = Field(
        None, description="Hint about query complexity: simple, moderate, complex, expert"
    )
    urgency_hint: Optional[str] = Field(
        None, description="Hint about urgency: research, implementation, critical_fix"
    )
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    request_id: Optional[str] = Field(None, description="Unique request identifier")


class ZenOrchestrationRequest(BaseModel):
    """Advanced Zen orchestration request with full intelligence features."""
    
    query: str = Field(..., description="The main query/prompt")
    context: Optional[List[str]] = Field(None, description="Additional context information")
    files: Optional[List[str]] = Field(None, description="File references for analysis")
    
    # Orchestration configuration
    strategy: Optional[str] = Field(None, description="Orchestration strategy override")
    models: Optional[List[str]] = Field(None, description="Specific models to use")
    consensus_threshold: Optional[float] = Field(None, description="Consensus threshold override")
    
    # Performance constraints
    max_latency: Optional[float] = Field(None, description="Maximum latency in seconds")
    max_cost: Optional[float] = Field(None, description="Maximum cost in USD")
    quality_target: Optional[float] = Field(None, description="Target quality score")
    
    # Advanced features
    enable_reasoning: Optional[bool] = Field(True, description="Include reasoning in response")
    enable_consensus_details: Optional[bool] = Field(False, description="Include detailed consensus analysis")
    enable_performance_feedback: Optional[bool] = Field(True, description="Record performance metrics")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")


class QueryAnalysisRequest(BaseModel):
    """Request for query analysis and routing recommendations."""
    
    query: str = Field(..., description="Query to analyze")
    context: Optional[List[str]] = Field(None, description="Optional context")
    available_models: Optional[List[str]] = Field(None, description="Available models for routing")
    
    # Analysis options
    include_routing_decision: bool = Field(True, description="Include routing recommendations")
    include_cost_estimate: bool = Field(True, description="Include cost estimates")
    include_performance_prediction: bool = Field(True, description="Include performance predictions")


class FeedbackRequest(BaseModel):
    """Request to provide feedback on orchestration results."""
    
    session_id: str = Field(..., description="Session ID to provide feedback for")
    satisfaction_score: float = Field(..., description="Satisfaction score (0.0 to 1.0)", ge=0.0, le=1.0)
    quality_rating: Optional[float] = Field(None, description="Quality rating (0.0 to 1.0)", ge=0.0, le=1.0)
    speed_rating: Optional[float] = Field(None, description="Speed rating (0.0 to 1.0)", ge=0.0, le=1.0)
    
    # Qualitative feedback
    feedback_text: Optional[str] = Field(None, description="Detailed feedback text")
    liked_aspects: Optional[List[str]] = Field(None, description="Aspects the user liked")
    improvement_suggestions: Optional[List[str]] = Field(None, description="Suggestions for improvement")
    
    # Metadata
    user_id: Optional[str] = Field(None, description="User providing feedback")
    timestamp: Optional[float] = Field(None, description="Feedback timestamp")