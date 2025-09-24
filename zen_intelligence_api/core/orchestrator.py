"""
Advanced Orchestration Engine for Multi-Model Intelligence

This module provides sophisticated multi-model coordination with learning capabilities,
adaptive routing, and intelligent synthesis. It leverages Zen's existing MCP infrastructure
while adding enterprise-grade orchestration capabilities.
"""

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .intelligence import QueryIntelligence, RoutingDecision
from .consensus import ConsensusBuilder
from .performance_tracker import PerformanceTracker
from ..models.orchestration_request import OrchestrationRequest
from ..models.orchestration_response import OrchestrationResponse, ModelResponse


class OrchestrationMetrics(BaseModel):
    """Metrics for an orchestration session."""
    
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    total_latency: Optional[float] = None
    primary_model_latency: Optional[float] = None
    supporting_models_latency: Dict[str, float] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost: float = 0.0
    quality_score: float = 0.0
    confidence_score: float = 0.0
    strategy_used: str = ""
    models_involved: List[str] = Field(default_factory=list)


class ZenOrchestrator:
    """Advanced multi-model coordination with learning capabilities."""
    
    def __init__(self):
        """Initialize the orchestrator with intelligence components."""
        self.query_intelligence = QueryIntelligence()
        self.consensus_builder = ConsensusBuilder()
        self.performance_tracker = PerformanceTracker()
        
        # Session tracking
        self.active_sessions: Dict[str, OrchestrationMetrics] = {}
        
        # Learning from past performances
        self.adaptation_enabled = True
    
    async def intelligent_orchestrate(
        self, 
        request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """
        Advanced multi-model coordination with learning capabilities.
        
        Args:
            request: Orchestration request with query and preferences
            
        Returns:
            OrchestrationResponse with synthesized result and metrics
        """
        session_id = str(uuid.uuid4())
        metrics = OrchestrationMetrics(
            session_id=session_id,
            start_time=time.time()
        )
        
        try:
            # Phase 1: Query Analysis & Strategic Planning
            routing_decision = await self._analyze_and_plan(request)
            metrics.strategy_used = routing_decision.orchestration_strategy
            metrics.models_involved = [routing_decision.primary_model] + routing_decision.supporting_models
            
            # Phase 2: Parallel Specialized Execution
            specialized_responses = await self._execute_parallel_strategy(routing_decision, request)
            
            # Phase 3: Cross-Model Validation & Consensus
            consensus_data = await self._build_consensus(specialized_responses, routing_decision)
            
            # Phase 4: Intelligent Synthesis with Confidence Scoring
            final_response = await self._synthesize_with_intelligence(consensus_data, routing_decision)
            
            # Phase 5: Learning & Performance Feedback
            await self._update_performance_metrics(routing_decision, final_response, metrics)
            
            # Complete metrics
            metrics.end_time = time.time()
            metrics.total_latency = metrics.end_time - metrics.start_time
            
            return OrchestrationResponse(
                content=final_response.content,
                model_responses=specialized_responses,
                routing_decision=routing_decision,
                consensus_score=consensus_data.get("consensus_score", 0.0),
                confidence_score=final_response.confidence_score,
                quality_score=final_response.quality_score,
                session_id=session_id,
                metrics=metrics,
                reasoning=final_response.reasoning
            )
            
        except Exception as e:
            # Record failure metrics
            metrics.end_time = time.time()
            metrics.total_latency = metrics.end_time - metrics.start_time
            
            # Return error response
            return OrchestrationResponse(
                content=f"Orchestration failed: {str(e)}",
                model_responses=[],
                routing_decision=None,
                consensus_score=0.0,
                confidence_score=0.0,
                quality_score=0.0,
                session_id=session_id,
                metrics=metrics,
                reasoning="Error occurred during orchestration",
                error=str(e)
            )
    
    async def _analyze_and_plan(self, request: OrchestrationRequest) -> RoutingDecision:
        """Phase 1: Query Analysis & Strategic Planning."""
        
        # Extract context from conversation history
        context = []
        if request.conversation_history:
            context = [msg.content for msg in request.conversation_history[-5:]]  # Last 5 messages
        
        # Analyze query for optimal routing
        routing_decision = self.query_intelligence.analyze_query(
            query=request.messages[-1].content,  # Last message is the query
            context=context
        )
        
        # Apply user preferences if specified
        if request.preferred_models:
            # Override with user preferences while respecting capabilities
            available_preferred = [m for m in request.preferred_models 
                                 if m in self.query_intelligence.MODEL_EXPERTISE.keys()]
            if available_preferred:
                routing_decision.primary_model = available_preferred[0]
                routing_decision.supporting_models = available_preferred[1:3]  # Up to 2 supporting
        
        # Apply performance learning adjustments
        if self.adaptation_enabled:
            routing_decision = await self._apply_performance_learning(routing_decision, request)
        
        return routing_decision
    
    async def _execute_parallel_strategy(
        self, 
        routing_decision: RoutingDecision, 
        request: OrchestrationRequest
    ) -> List[ModelResponse]:
        """Phase 2: Parallel Specialized Execution."""
        
        responses = []
        
        if routing_decision.orchestration_strategy == "single":
            # Single model execution
            response = await self._execute_single_model(
                routing_decision.primary_model, 
                request, 
                is_primary=True
            )
            responses.append(response)
            
        elif routing_decision.orchestration_strategy == "parallel":
            # Parallel execution of all models
            tasks = []
            
            # Primary model task
            tasks.append(self._execute_single_model(
                routing_decision.primary_model, 
                request, 
                is_primary=True
            ))
            
            # Supporting models tasks
            for model in routing_decision.supporting_models:
                tasks.append(self._execute_single_model(
                    model, 
                    request, 
                    is_primary=False
                ))
            
            # Execute all in parallel
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to ModelResponse objects
            valid_responses = []
            for i, response in enumerate(responses):
                if not isinstance(response, Exception):
                    valid_responses.append(response)
                else:
                    model_name = (routing_decision.primary_model if i == 0 
                                else routing_decision.supporting_models[i-1])
                    # Create error response
                    valid_responses.append(ModelResponse(
                        model=model_name,
                        content=f"Error: {str(response)}",
                        confidence_score=0.0,
                        latency=0.0,
                        tokens_used=0,
                        cost=0.0,
                        error=str(response)
                    ))
            
            responses = valid_responses
            
        elif routing_decision.orchestration_strategy == "sequential":
            # Sequential execution with refinement
            responses = await self._execute_sequential_strategy(routing_decision, request)
            
        elif routing_decision.orchestration_strategy == "consensus":
            # Consensus-based execution
            responses = await self._execute_consensus_strategy(routing_decision, request)
        
        return responses
    
    async def _execute_single_model(
        self, 
        model: str, 
        request: OrchestrationRequest, 
        is_primary: bool = True
    ) -> ModelResponse:
        """Execute a single model call."""
        start_time = time.time()
        
        try:
            # This would integrate with Zen's existing provider system
            # For now, we'll simulate the call
            await asyncio.sleep(0.1)  # Simulate API call
            
            # Simulate response
            content = f"Response from {model}: Analysis of '{request.messages[-1].content[:50]}...'"
            
            end_time = time.time()
            latency = end_time - start_time
            
            return ModelResponse(
                model=model,
                content=content,
                confidence_score=0.85 if is_primary else 0.80,
                latency=latency,
                tokens_used=150,
                cost=0.01,
                is_primary=is_primary
            )
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            return ModelResponse(
                model=model,
                content=f"Error executing {model}",
                confidence_score=0.0,
                latency=latency,
                tokens_used=0,
                cost=0.0,
                is_primary=is_primary,
                error=str(e)
            )
    
    async def _execute_sequential_strategy(
        self, 
        routing_decision: RoutingDecision, 
        request: OrchestrationRequest
    ) -> List[ModelResponse]:
        """Execute sequential refinement strategy."""
        responses = []
        
        # Start with primary model
        primary_response = await self._execute_single_model(
            routing_decision.primary_model, 
            request, 
            is_primary=True
        )
        responses.append(primary_response)
        
        # Refine with supporting models
        for model in routing_decision.supporting_models:
            # Create refinement request based on previous response
            refinement_request = self._create_refinement_request(request, responses[-1])
            
            supporting_response = await self._execute_single_model(
                model, 
                refinement_request, 
                is_primary=False
            )
            responses.append(supporting_response)
        
        return responses
    
    async def _execute_consensus_strategy(
        self, 
        routing_decision: RoutingDecision, 
        request: OrchestrationRequest
    ) -> List[ModelResponse]:
        """Execute consensus-building strategy."""
        
        # First, get parallel responses from all models
        tasks = []
        tasks.append(self._execute_single_model(
            routing_decision.primary_model, 
            request, 
            is_primary=True
        ))
        
        for model in routing_decision.supporting_models:
            tasks.append(self._execute_single_model(
                model, 
                request, 
                is_primary=False
            ))
        
        initial_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid responses
        valid_responses = [r for r in initial_responses if not isinstance(r, Exception)]
        
        # Build consensus through iterative refinement
        if len(valid_responses) > 1:
            consensus_response = await self._build_consensus_response(valid_responses, request)
            valid_responses.append(consensus_response)
        
        return valid_responses
    
    async def _build_consensus(
        self, 
        responses: List[ModelResponse], 
        routing_decision: RoutingDecision
    ) -> Dict[str, Any]:
        """Phase 3: Cross-Model Validation & Consensus."""
        
        if not responses:
            return {"consensus_score": 0.0, "agreements": [], "disagreements": []}
        
        # Use consensus builder to analyze agreements and disagreements
        consensus_data = await self.consensus_builder.analyze_responses(responses)
        
        # Calculate weighted consensus score based on model expertise
        weighted_score = self._calculate_weighted_consensus(
            responses, 
            routing_decision, 
            consensus_data
        )
        
        consensus_data["weighted_consensus_score"] = weighted_score
        return consensus_data
    
    async def _synthesize_with_intelligence(
        self, 
        consensus_data: Dict[str, Any], 
        routing_decision: RoutingDecision
    ) -> ModelResponse:
        """Phase 4: Intelligent Synthesis with Confidence Scoring."""
        
        # Get all model responses
        responses = consensus_data.get("responses", [])
        if not responses:
            return ModelResponse(
                model="synthesizer",
                content="No valid responses to synthesize",
                confidence_score=0.0,
                latency=0.0,
                tokens_used=0,
                cost=0.0
            )
        
        # Find primary response
        primary_response = next((r for r in responses if r.is_primary), responses[0])
        
        # Calculate synthesis confidence based on consensus
        consensus_score = consensus_data.get("weighted_consensus_score", 0.0)
        confidence_score = min(0.95, (primary_response.confidence_score + consensus_score) / 2)
        
        # Create synthesized content
        if len(responses) == 1:
            # Single model response
            synthesized_content = primary_response.content
            reasoning = f"Single model response from {primary_response.model}"
        else:
            # Multi-model synthesis
            synthesized_content = await self._create_synthesized_content(responses, consensus_data)
            reasoning = self._create_synthesis_reasoning(responses, consensus_data, routing_decision)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(responses, consensus_data)
        
        return ModelResponse(
            model="zen-orchestrator",
            content=synthesized_content,
            confidence_score=confidence_score,
            latency=sum(r.latency for r in responses),
            tokens_used=sum(r.tokens_used for r in responses),
            cost=sum(r.cost for r in responses),
            quality_score=quality_score,
            reasoning=reasoning
        )
    
    async def _update_performance_metrics(
        self, 
        routing_decision: RoutingDecision, 
        final_response: ModelResponse,
        metrics: OrchestrationMetrics
    ) -> None:
        """Phase 5: Learning & Performance Feedback."""
        
        # Update metrics
        metrics.total_tokens = final_response.tokens_used
        metrics.total_cost = final_response.cost
        metrics.quality_score = getattr(final_response, 'quality_score', 0.0)
        metrics.confidence_score = final_response.confidence_score
        
        # Store for learning
        self.active_sessions[metrics.session_id] = metrics
        
        # Update performance tracker
        await self.performance_tracker.record_session(
            routing_decision=routing_decision,
            metrics=metrics,
            final_response=final_response
        )
    
    async def _apply_performance_learning(
        self, 
        routing_decision: RoutingDecision, 
        request: OrchestrationRequest
    ) -> RoutingDecision:
        """Apply learning from past performance to routing decision."""
        
        # Get performance history for similar queries
        similar_sessions = await self.performance_tracker.get_similar_sessions(
            domain=routing_decision.detected_domain,
            complexity=routing_decision.detected_complexity
        )
        
        if similar_sessions:
            # Adjust model selection based on historical performance
            best_performing_models = self.performance_tracker.get_best_models_for_domain(
                routing_decision.detected_domain
            )
            
            if best_performing_models:
                # Prefer historically successful models
                if best_performing_models[0] in self.query_intelligence.MODEL_EXPERTISE:
                    routing_decision.primary_model = best_performing_models[0]
        
        return routing_decision
    
    # Helper methods
    
    def _create_refinement_request(
        self, 
        original_request: OrchestrationRequest, 
        previous_response: ModelResponse
    ) -> OrchestrationRequest:
        """Create a refinement request based on previous response."""
        # This would create a modified request for iterative refinement
        return original_request  # Simplified for now
    
    async def _build_consensus_response(
        self, 
        responses: List[ModelResponse], 
        request: OrchestrationRequest
    ) -> ModelResponse:
        """Build a consensus response from multiple model outputs."""
        
        # Simple consensus: take the response with highest confidence
        best_response = max(responses, key=lambda r: r.confidence_score)
        
        return ModelResponse(
            model="consensus",
            content=f"Consensus view: {best_response.content}",
            confidence_score=best_response.confidence_score * 0.9,  # Slight discount for consensus
            latency=0.1,  # Minimal processing time
            tokens_used=10,
            cost=0.001,
            is_primary=False
        )
    
    def _calculate_weighted_consensus(
        self, 
        responses: List[ModelResponse], 
        routing_decision: RoutingDecision,
        consensus_data: Dict[str, Any]
    ) -> float:
        """Calculate weighted consensus score based on model expertise."""
        
        if not responses:
            return 0.0
        
        # Get base consensus score
        base_score = consensus_data.get("consensus_score", 0.0)
        
        # Weight by model expertise for the detected domain
        total_weight = 0.0
        weighted_sum = 0.0
        
        for response in responses:
            if response.model in self.query_intelligence.MODEL_EXPERTISE:
                expertise = self.query_intelligence.MODEL_EXPERTISE[response.model]
                domain_expertise = getattr(expertise, routing_decision.detected_domain.value, 0.7)
                
                weighted_sum += response.confidence_score * domain_expertise
                total_weight += domain_expertise
        
        if total_weight > 0:
            expertise_weighted_score = weighted_sum / total_weight
            return (base_score + expertise_weighted_score) / 2
        
        return base_score
    
    async def _create_synthesized_content(
        self, 
        responses: List[ModelResponse], 
        consensus_data: Dict[str, Any]
    ) -> str:
        """Create synthesized content from multiple responses."""
        
        # Find primary response
        primary_response = next((r for r in responses if r.is_primary), responses[0])
        
        # Get supporting insights
        supporting_insights = []
        for response in responses:
            if not response.is_primary and response.content:
                # Extract key insights (simplified)
                if len(response.content) > 50:
                    insight = response.content[:100] + "..."
                    supporting_insights.append(f"• {response.model}: {insight}")
        
        # Create synthesized response
        synthesized = primary_response.content
        
        if supporting_insights:
            synthesized += "\n\n**Additional Perspectives:**\n" + "\n".join(supporting_insights)
        
        # Add consensus information if available
        if consensus_data.get("agreements"):
            synthesized += f"\n\n**Consensus Points:** {len(consensus_data['agreements'])} agreements found"
        
        return synthesized
    
    def _create_synthesis_reasoning(
        self, 
        responses: List[ModelResponse], 
        consensus_data: Dict[str, Any],
        routing_decision: RoutingDecision
    ) -> str:
        """Create reasoning explanation for the synthesis."""
        
        model_names = [r.model for r in responses]
        consensus_score = consensus_data.get("weighted_consensus_score", 0.0)
        
        return (
            f"Synthesized response from {len(responses)} models ({', '.join(model_names)}) "
            f"using {routing_decision.orchestration_strategy} strategy. "
            f"Consensus score: {consensus_score:.2f}/1.0"
        )
    
    def _calculate_quality_score(
        self, 
        responses: List[ModelResponse], 
        consensus_data: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score for the synthesized response."""
        
        if not responses:
            return 0.0
        
        # Average confidence scores
        avg_confidence = sum(r.confidence_score for r in responses) / len(responses)
        
        # Factor in consensus
        consensus_score = consensus_data.get("weighted_consensus_score", 0.0)
        
        # Combine scores (weighted average)
        quality_score = (avg_confidence * 0.7) + (consensus_score * 0.3)
        
        # Bonus for multiple models agreeing
        if len(responses) > 1 and consensus_score > 0.8:
            quality_score = min(0.95, quality_score * 1.1)
        
        return quality_score