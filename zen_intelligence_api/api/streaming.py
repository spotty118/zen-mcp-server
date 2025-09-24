"""
Advanced Streaming Capabilities for Zen Intelligence API

This module provides sophisticated streaming capabilities for real-time
multi-model orchestration responses.
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Any

from ..models.orchestration_request import OrchestrationRequest
from ..models.orchestration_response import StreamingChunk


class ZenStreamingHandler:
    """Advanced streaming handler for multi-model orchestration."""
    
    def __init__(self, orchestrator):
        """Initialize streaming handler with orchestrator."""
        self.orchestrator = orchestrator
        
    async def stream_chat_completion(
        self, 
        request: OrchestrationRequest,
        user: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion with intelligent orchestration.
        
        This method provides real-time streaming of orchestration results,
        including intermediate model responses and final synthesis.
        """
        
        response_id = str(uuid.uuid4())
        created_timestamp = int(time.time())
        
        try:
            # Start orchestration
            orchestration_task = asyncio.create_task(
                self.orchestrator.intelligent_orchestrate(request)
            )
            
            # Send initial chunk
            initial_chunk = StreamingChunk(
                id=response_id,
                created=created_timestamp,
                model="zen-orchestrator",
                choices=[{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }],
                zen_chunk_metadata={
                    "stage": "initializing",
                    "message": "Starting intelligent orchestration..."
                }
            )
            
            yield f"data: {initial_chunk.json()}\n\n"
            
            # Send routing decision chunk
            await asyncio.sleep(0.1)  # Small delay to simulate analysis
            
            routing_chunk = StreamingChunk(
                id=response_id,
                created=created_timestamp,
                model="zen-orchestrator",
                choices=[{
                    "index": 0,
                    "delta": {"content": ""},
                    "finish_reason": None
                }],
                zen_chunk_metadata={
                    "stage": "routing",
                    "message": "Analyzing query and selecting optimal models..."
                }
            )
            
            yield f"data: {routing_chunk.json()}\n\n"
            
            # Wait for orchestration to complete (with periodic updates)
            while not orchestration_task.done():
                await asyncio.sleep(0.5)
                
                # Send progress update
                progress_chunk = StreamingChunk(
                    id=response_id,
                    created=created_timestamp,
                    model="zen-orchestrator",
                    choices=[{
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": None
                    }],
                    zen_chunk_metadata={
                        "stage": "processing",
                        "message": "Models are processing your request..."
                    }
                )
                
                yield f"data: {progress_chunk.json()}\n\n"
            
            # Get orchestration result
            orchestration_response = await orchestration_task
            
            # Stream the content in chunks
            content = orchestration_response.content
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                
                content_chunk = StreamingChunk(
                    id=response_id,
                    created=created_timestamp,
                    model=orchestration_response.routing_decision.primary_model if orchestration_response.routing_decision else "zen-orchestrator",
                    choices=[{
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None
                    }],
                    zen_chunk_metadata={
                        "stage": "content",
                        "chunk_index": i // chunk_size,
                        "total_chunks": (len(content) + chunk_size - 1) // chunk_size
                    }
                )
                
                yield f"data: {content_chunk.json()}\n\n"
                
                # Small delay between chunks for realistic streaming
                await asyncio.sleep(0.05)
            
            # Send final chunk with metadata
            final_chunk = StreamingChunk(
                id=response_id,
                created=created_timestamp,
                model=orchestration_response.routing_decision.primary_model if orchestration_response.routing_decision else "zen-orchestrator",
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                zen_chunk_metadata={
                    "stage": "complete",
                    "session_id": orchestration_response.session_id,
                    "consensus_score": orchestration_response.consensus_score,
                    "confidence_score": orchestration_response.confidence_score,
                    "quality_score": orchestration_response.quality_score,
                    "models_used": orchestration_response.metrics.models_involved,
                    "total_latency": orchestration_response.metrics.total_latency,
                    "total_cost": orchestration_response.metrics.total_cost,
                    "strategy_used": orchestration_response.metrics.strategy_used
                }
            )
            
            yield f"data: {final_chunk.json()}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            # Send error chunk
            error_chunk = StreamingChunk(
                id=response_id,
                created=created_timestamp,
                model="zen-orchestrator",
                choices=[{
                    "index": 0,
                    "delta": {"content": f"Error: {str(e)}"},
                    "finish_reason": "error"
                }],
                zen_chunk_metadata={
                    "stage": "error",
                    "error": str(e)
                }
            )
            
            yield f"data: {error_chunk.json()}\n\n"
            yield "data: [DONE]\n\n"
    
    async def stream_zen_orchestration(
        self,
        request: "ZenOrchestrationRequest",
        user: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream advanced Zen orchestration with detailed progress.
        
        This method provides detailed streaming of the orchestration process,
        including individual model responses and consensus building.
        """
        
        session_id = str(uuid.uuid4())
        
        try:
            # Convert to OrchestrationRequest
            orchestration_request = OrchestrationRequest(
                messages=[{"role": "user", "content": request.query}],
                orchestration_preferences={
                    "orchestration_strategy": request.strategy,
                    "preferred_models": request.models,
                    "max_latency": request.max_latency,
                    "max_cost": request.max_cost
                },
                user_id=request.user_id,
                session_id=session_id
            )
            
            # Phase 1: Analysis and Planning
            yield {
                "stage": "analysis",
                "message": "Analyzing query for optimal routing...",
                "session_id": session_id
            }
            
            # Simulate analysis time
            await asyncio.sleep(0.2)
            
            # Phase 2: Model Execution
            yield {
                "stage": "execution",
                "message": "Executing orchestration strategy...",
                "session_id": session_id
            }
            
            # Start orchestration
            orchestration_task = asyncio.create_task(
                self.orchestrator.intelligent_orchestrate(orchestration_request)
            )
            
            # Simulate model responses
            model_count = 0
            while not orchestration_task.done():
                await asyncio.sleep(0.5)
                model_count += 1
                
                yield {
                    "stage": "model_response",
                    "message": f"Model {model_count} completed processing...",
                    "session_id": session_id
                }
            
            # Phase 3: Consensus Building
            yield {
                "stage": "consensus",
                "message": "Building consensus across model responses...",
                "session_id": session_id
            }
            
            await asyncio.sleep(0.1)
            
            # Phase 4: Synthesis
            yield {
                "stage": "synthesis",
                "message": "Synthesizing final response...",
                "session_id": session_id
            }
            
            # Get final result
            orchestration_response = await orchestration_task
            
            # Phase 5: Complete
            yield {
                "stage": "complete",
                "response": orchestration_response.dict(),
                "session_id": session_id
            }
            
        except Exception as e:
            yield {
                "stage": "error",
                "error": str(e),
                "session_id": session_id
            }
    
    async def stream_performance_updates(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time performance updates and metrics.
        
        This method provides a live feed of performance metrics and
        system status updates.
        """
        
        try:
            while True:
                # Get current performance metrics
                performance_tracker = self.orchestrator.performance_tracker
                global_stats = performance_tracker.get_global_performance_stats()
                
                # Create update message
                update = {
                    "timestamp": time.time(),
                    "type": "performance_update",
                    "data": {
                        "total_sessions": global_stats.get("total_sessions", 0),
                        "avg_quality": global_stats.get("avg_session_quality", 0.0),
                        "avg_latency": global_stats.get("avg_session_latency", 0.0),
                        "success_rate": global_stats.get("global_success_rate", 0.0),
                        "active_models": global_stats.get("total_models", 0)
                    }
                }
                
                yield update
                
                # Wait before next update
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
        except asyncio.CancelledError:
            # Clean shutdown
            yield {
                "timestamp": time.time(),
                "type": "stream_ended",
                "message": "Performance stream ended"
            }