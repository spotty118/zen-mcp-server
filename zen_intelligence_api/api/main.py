"""
Zen Intelligence API - Main FastAPI Server

This module provides the main FastAPI server that implements an OpenAI-compatible
API with advanced multi-model orchestration capabilities.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from ..core.orchestrator import ZenOrchestrator
from ..core.intelligence import QueryIntelligence
from ..models.orchestration_request import (
    OrchestrationRequest, 
    ZenOrchestrationRequest, 
    QueryAnalysisRequest,
    FeedbackRequest
)
from ..models.orchestration_response import (
    OrchestrationResponse,
    ChatCompletionResponse,
    StreamingChunk,
    PerformanceMetricsResponse,
    QueryAnalysisResponse,
    FeedbackResponse,
    UsageStats
)
from .middleware import add_logging_middleware, add_security_middleware
from .streaming import ZenStreamingHandler


# Initialize FastAPI app
app = FastAPI(
    title="Zen Intelligence API",
    description="Enterprise-Grade Multi-Model AI Orchestration API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
add_logging_middleware(app)
add_security_middleware(app)

# Initialize core components
orchestrator = ZenOrchestrator()
query_intelligence = QueryIntelligence()
streaming_handler = ZenStreamingHandler(orchestrator)

# Security
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Get current user from authorization token.
    In production, this would validate JWT tokens or API keys.
    """
    # Simplified for demo - in production, validate the token
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    # For demo purposes, extract user_id from token (in practice, decode JWT)
    return {"user_id": "demo_user", "token": token}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {
            "orchestrator": "active",
            "query_intelligence": "active",
            "performance_tracker": "active"
        }
    }


# OpenAI-compatible chat completions endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: OrchestrationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a chat completion using intelligent multi-model orchestration.
    
    This endpoint is OpenAI-compatible but provides advanced orchestration
    capabilities through Zen's multi-model intelligence.
    """
    try:
        # Handle streaming requests
        if request.stream:
            return StreamingResponse(
                streaming_handler.stream_chat_completion(request, current_user),
                media_type="text/plain"
            )
        
        # Execute orchestration
        orchestration_response = await orchestrator.intelligent_orchestrate(request)
        
        # Convert to OpenAI-compatible format
        response_id = str(uuid.uuid4())
        created_timestamp = int(time.time())
        
        # Prepare choices
        choices = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": orchestration_response.content
            },
            "finish_reason": "stop"
        }]
        
        # Prepare usage statistics
        usage = UsageStats(
            prompt_tokens=orchestration_response.metrics.total_tokens // 2,  # Estimate
            completion_tokens=orchestration_response.metrics.total_tokens // 2,  # Estimate
            total_tokens=orchestration_response.metrics.total_tokens,
            models_used=orchestration_response.metrics.models_involved,
            total_cost=orchestration_response.metrics.total_cost,
            orchestration_strategy=orchestration_response.metrics.strategy_used
        )
        
        # Prepare Zen metadata (optional)
        zen_metadata = {
            "routing_decision": orchestration_response.routing_decision.dict() if orchestration_response.routing_decision else None,
            "consensus_score": orchestration_response.consensus_score,
            "confidence_score": orchestration_response.confidence_score,
            "quality_score": orchestration_response.quality_score,
            "reasoning": orchestration_response.reasoning,
            "session_id": orchestration_response.session_id,
            "model_responses": [response.dict() for response in orchestration_response.model_responses]
        }
        
        return ChatCompletionResponse(
            id=response_id,
            created=created_timestamp,
            model=orchestration_response.routing_decision.primary_model if orchestration_response.routing_decision else "zen-orchestrator",
            choices=choices,
            usage=usage,
            zen_metadata=zen_metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


# Advanced Zen orchestration endpoint
@app.post("/v1/zen/orchestrate", response_model=OrchestrationResponse)
async def zen_orchestrate(
    request: ZenOrchestrationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Advanced orchestration endpoint with full Zen intelligence features.
    
    This endpoint provides access to all advanced orchestration capabilities
    including custom strategies, detailed consensus analysis, and performance metrics.
    """
    try:
        # Convert ZenOrchestrationRequest to OrchestrationRequest
        orchestration_request = OrchestrationRequest(
            messages=[{"role": "user", "content": request.query}],
            orchestration_preferences={
                "orchestration_strategy": request.strategy,
                "preferred_models": request.models,
                "max_latency": request.max_latency,
                "max_cost": request.max_cost,
                "enable_learning": request.enable_performance_feedback
            },
            conversation_history=[{"role": "user", "content": ctx} for ctx in request.context or []],
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        # Execute orchestration
        response = await orchestrator.intelligent_orchestrate(orchestration_request)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced orchestration failed: {str(e)}")


# Query analysis endpoint
@app.post("/v1/zen/analyze", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QueryAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze a query and provide routing recommendations without execution.
    
    This endpoint provides intelligence analysis and cost/performance estimates
    without actually executing the orchestration.
    """
    try:
        # Analyze query
        routing_decision = query_intelligence.analyze_query(
            query=request.query,
            context=request.context
        )
        
        # Generate alternative strategies
        alternatives = []
        for strategy in ["single", "parallel", "sequential", "consensus"]:
            if strategy != routing_decision.orchestration_strategy:
                alt_routing = routing_decision.copy()
                alt_routing.orchestration_strategy = strategy
                alternatives.append({
                    "strategy": strategy,
                    "estimated_cost": alt_routing.estimated_cost,
                    "estimated_latency": alt_routing.estimated_latency,
                    "reasoning": f"Alternative {strategy} strategy"
                })
        
        # Generate recommendations
        recommendations = [
            f"Primary model: {routing_decision.primary_model}",
            f"Strategy: {routing_decision.orchestration_strategy}",
            f"Expected quality: {routing_decision.expected_quality_score:.2f}",
        ]
        
        if routing_decision.supporting_models:
            recommendations.append(f"Supporting models: {', '.join(routing_decision.supporting_models)}")
        
        return QueryAnalysisResponse(
            routing_decision=routing_decision,
            estimated_cost=routing_decision.estimated_cost,
            estimated_latency=routing_decision.estimated_latency,
            estimated_quality=routing_decision.expected_quality_score,
            recommendations=recommendations,
            alternative_strategies=alternatives
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")


# Performance metrics endpoint
@app.get("/v1/zen/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    current_user: dict = Depends(get_current_user),
    model: Optional[str] = None,
    domain: Optional[str] = None
):
    """
    Get performance metrics and analytics.
    
    Provides detailed performance analytics including model rankings,
    trends, and best performers across different domains.
    """
    try:
        performance_tracker = orchestrator.performance_tracker
        
        # Get global statistics
        global_stats = performance_tracker.get_global_performance_stats()
        
        # Get domain-specific performance
        domain_performance = {}
        model_rankings = {}
        
        for domain_enum in query_intelligence.QueryDomain:
            domain_key = domain_enum.value
            domain_performance[domain_key] = performance_tracker.domain_performances[domain_enum].__dict__
            model_rankings[domain_key] = performance_tracker.get_best_models_for_domain(domain_enum)
        
        # Get trends (if specific model requested)
        trends = {}
        if model:
            for domain_enum in query_intelligence.QueryDomain:
                domain_trends = performance_tracker.get_performance_trends(model, domain_enum)
                if domain_trends:
                    trends[f"{model}_{domain_enum.value}"] = domain_trends
        
        # Get best performers
        best_performers = global_stats.get("best_performers", {})
        
        return PerformanceMetricsResponse(
            global_stats=global_stats,
            domain_performance=domain_performance,
            model_rankings=model_rankings,
            trends=trends,
            best_performers=best_performers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


# Feedback endpoint
@app.post("/v1/zen/learn", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit user feedback for continuous improvement.
    
    This endpoint allows users to provide feedback on orchestration results,
    which is used to improve future routing decisions and model selection.
    """
    try:
        performance_tracker = orchestrator.performance_tracker
        
        # Record feedback
        await performance_tracker.record_user_feedback(
            session_id=request.session_id,
            satisfaction_score=request.satisfaction_score,
            feedback_text=request.feedback_text
        )
        
        # Get learning insights (simplified)
        learning_insights = [
            "Feedback recorded successfully",
            "Model performance metrics updated",
        ]
        
        if request.satisfaction_score < 0.5:
            learning_insights.append("Low satisfaction noted - will adjust routing for similar queries")
        elif request.satisfaction_score > 0.8:
            learning_insights.append("High satisfaction noted - will reinforce successful patterns")
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
            learning_insights=learning_insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


# Model listing endpoint (OpenAI compatible)
@app.get("/v1/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """List available models (OpenAI compatible)."""
    
    models = []
    for model_name in query_intelligence.MODEL_EXPERTISE.keys():
        models.append({
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "zen-intelligence"
        })
    
    # Add Zen orchestrator as a special model
    models.append({
        "id": "zen-orchestrator",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "zen-intelligence"
    })
    
    return {"object": "list", "data": models}


# Advanced configuration endpoint
@app.get("/v1/zen/config")
async def get_zen_config(current_user: dict = Depends(get_current_user)):
    """Get Zen Intelligence API configuration."""
    
    return {
        "version": "1.0.0",
        "features": {
            "intelligent_routing": True,
            "multi_model_orchestration": True,
            "consensus_building": True,
            "performance_learning": True,
            "streaming_support": True,
            "openai_compatibility": True
        },
        "available_strategies": ["single", "parallel", "sequential", "consensus"],
        "supported_domains": [domain.value for domain in query_intelligence.QueryDomain],
        "complexity_levels": ["simple", "moderate", "complex", "expert"],
        "urgency_levels": ["research", "implementation", "critical_fix"],
        "models": {
            model: expertise.dict() 
            for model, expertise in query_intelligence.MODEL_EXPERTISE.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )