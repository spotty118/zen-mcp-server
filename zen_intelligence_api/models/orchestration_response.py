"""
Orchestration Response Models

This module defines the response models for the Zen Intelligence API,
providing OpenAI-compatible structures with advanced orchestration metadata.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.intelligence import RoutingDecision


class ModelResponse(BaseModel):
    """Response from a single model."""
    
    model: str = Field(..., description="Model name")
    content: str = Field(..., description="Response content")
    confidence_score: float = Field(..., description="Model's confidence in the response", ge=0.0, le=1.0)
    
    # Performance metrics
    latency: float = Field(..., description="Response latency in seconds")
    tokens_used: int = Field(..., description="Number of tokens used")
    cost: float = Field(..., description="Cost of the request in USD")
    
    # Status information
    is_primary: bool = Field(default=False, description="Whether this was the primary model")
    error: Optional[str] = Field(None, description="Error message if request failed")
    
    # Quality metrics
    quality_score: Optional[float] = Field(None, description="Assessed quality score", ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description="Reasoning behind the response")


class UsageStats(BaseModel):
    """Usage statistics for the orchestration."""
    
    prompt_tokens: int = Field(..., description="Tokens used in prompts")
    completion_tokens: int = Field(..., description="Tokens used in completions")
    total_tokens: int = Field(..., description="Total tokens used")
    
    # Zen-specific metrics
    models_used: List[str] = Field(..., description="List of models used")
    total_cost: float = Field(..., description="Total cost in USD")
    orchestration_strategy: str = Field(..., description="Strategy used for orchestration")


class OrchestrationMetrics(BaseModel):
    """Detailed orchestration metrics."""
    
    session_id: str = Field(..., description="Unique session identifier")
    total_latency: float = Field(..., description="Total orchestration latency in seconds")
    primary_model_latency: Optional[float] = Field(None, description="Primary model latency")
    supporting_models_latency: Dict[str, float] = Field(default_factory=dict, description="Supporting model latencies")
    
    # Quality metrics
    consensus_score: float = Field(..., description="Consensus score across models", ge=0.0, le=1.0)
    confidence_score: float = Field(..., description="Overall confidence score", ge=0.0, le=1.0)
    quality_score: float = Field(..., description="Overall quality score", ge=0.0, le=1.0)
    
    # Resource usage
    total_tokens: int = Field(..., description="Total tokens used")
    total_cost: float = Field(..., description="Total cost in USD")
    
    # Strategy information
    strategy_used: str = Field(..., description="Orchestration strategy used")
    models_involved: List[str] = Field(..., description="All models involved in orchestration")
    
    # Timing
    start_time: float = Field(..., description="Start timestamp")
    end_time: Optional[float] = Field(None, description="End timestamp")


class OrchestrationResponse(BaseModel):
    """Complete orchestration response with metadata."""
    
    # Main response content
    content: str = Field(..., description="Final synthesized response")
    
    # Individual model responses
    model_responses: List[ModelResponse] = Field(..., description="Responses from individual models")
    
    # Decision information
    routing_decision: Optional[RoutingDecision] = Field(None, description="Routing decision made")
    
    # Quality metrics
    consensus_score: float = Field(..., description="Consensus score across models")
    confidence_score: float = Field(..., description="Overall confidence score")
    quality_score: float = Field(..., description="Overall quality score")
    
    # Session information
    session_id: str = Field(..., description="Session identifier")
    
    # Performance metrics
    metrics: OrchestrationMetrics = Field(..., description="Detailed orchestration metrics")
    
    # Reasoning
    reasoning: str = Field(..., description="Explanation of orchestration decisions")
    
    # Error handling
    error: Optional[str] = Field(None, description="Error message if orchestration failed")
    
    # Usage statistics (OpenAI compatible)
    usage: Optional[UsageStats] = Field(None, description="Token usage statistics")


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    
    id: str = Field(..., description="Unique response identifier")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used (or 'zen-orchestrator' for multi-model)")
    
    # Response content
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    
    # Usage statistics
    usage: UsageStats = Field(..., description="Token usage statistics")
    
    # Zen-specific metadata (optional)
    zen_metadata: Optional[Dict[str, Any]] = Field(None, description="Advanced Zen orchestration metadata")


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    
    id: str = Field(..., description="Response identifier")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model identifier")
    
    # Chunk data
    choices: List[Dict[str, Any]] = Field(..., description="Streaming choices")
    
    # Zen-specific streaming metadata
    zen_chunk_metadata: Optional[Dict[str, Any]] = Field(None, description="Zen streaming metadata")


class PerformanceMetricsResponse(BaseModel):
    """Response with performance analytics."""
    
    # Global statistics
    global_stats: Dict[str, Any] = Field(..., description="Global performance statistics")
    
    # Domain-specific performance
    domain_performance: Dict[str, Any] = Field(..., description="Performance by domain")
    
    # Model rankings
    model_rankings: Dict[str, List[str]] = Field(..., description="Model rankings by domain")
    
    # Performance trends
    trends: Dict[str, List[float]] = Field(..., description="Performance trends over time")
    
    # Best performers
    best_performers: Dict[str, str] = Field(..., description="Best performing models by domain")


class QueryAnalysisResponse(BaseModel):
    """Response from query analysis."""
    
    # Analysis results
    routing_decision: RoutingDecision = Field(..., description="Recommended routing decision")
    
    # Cost and performance estimates
    estimated_cost: float = Field(..., description="Estimated cost in USD")
    estimated_latency: float = Field(..., description="Estimated latency in seconds")
    estimated_quality: float = Field(..., description="Estimated quality score")
    
    # Recommendations
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    
    # Alternative strategies
    alternative_strategies: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Alternative orchestration strategies"
    )


class FeedbackResponse(BaseModel):
    """Response to feedback submission."""
    
    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Status message")
    
    # Updated metrics (if available)
    updated_model_performance: Optional[Dict[str, Any]] = Field(
        None, description="Updated performance metrics for the models"
    )
    
    # Learning insights
    learning_insights: Optional[List[str]] = Field(
        None, description="Insights gained from the feedback"
    )