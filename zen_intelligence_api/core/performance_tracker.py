"""
Performance Tracking for Model Performance Analytics

This module tracks model performance across different domains and use cases
to enable adaptive routing and continuous improvement of orchestration strategies.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .intelligence import QueryDomain, QueryComplexity, RoutingDecision


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model in a specific context."""
    
    model_name: str
    domain: QueryDomain
    complexity: QueryComplexity
    
    # Performance data
    latency_ms: float
    tokens_used: int
    cost_usd: float
    quality_score: float
    confidence_score: float
    user_satisfaction: Optional[float] = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    strategy_used: str = ""
    
    # Success indicators
    completed_successfully: bool = True
    error_message: Optional[str] = None


@dataclass
class DomainPerformance:
    """Aggregated performance data for a domain."""
    
    domain: QueryDomain
    model_performances: Dict[str, List[PerformanceMetrics]] = field(default_factory=dict)
    
    # Aggregated metrics
    avg_quality_by_model: Dict[str, float] = field(default_factory=dict)
    avg_latency_by_model: Dict[str, float] = field(default_factory=dict)
    avg_cost_by_model: Dict[str, float] = field(default_factory=dict)
    success_rate_by_model: Dict[str, float] = field(default_factory=dict)
    
    # Best performers
    best_quality_model: Optional[str] = None
    best_speed_model: Optional[str] = None
    best_cost_model: Optional[str] = None
    best_overall_model: Optional[str] = None


class PerformanceTracker:
    """Tracks and analyzes model performance for adaptive routing."""
    
    def __init__(self, max_metrics_per_model: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            max_metrics_per_model: Maximum number of metrics to keep per model
        """
        self.max_metrics_per_model = max_metrics_per_model
        
        # Performance data storage
        self.domain_performances: Dict[QueryDomain, DomainPerformance] = {}
        self.global_metrics: List[PerformanceMetrics] = []
        
        # Model rankings
        self.model_rankings: Dict[QueryDomain, List[str]] = {}
        self.last_ranking_update = 0.0
        self.ranking_update_interval = 3600.0  # Update every hour
        
        # Performance trends
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize domain performances
        for domain in QueryDomain:
            self.domain_performances[domain] = DomainPerformance(domain=domain)
    
    async def record_session(
        self, 
        routing_decision: RoutingDecision, 
        metrics: 'OrchestrationMetrics',
        final_response: 'ModelResponse'
    ) -> None:
        """
        Record performance metrics from an orchestration session.
        
        Args:
            routing_decision: The routing decision that was made
            metrics: Session metrics
            final_response: Final synthesized response
        """
        
        # Create performance metric for primary model
        primary_metric = PerformanceMetrics(
            model_name=routing_decision.primary_model,
            domain=routing_decision.detected_domain,
            complexity=routing_decision.detected_complexity,
            latency_ms=metrics.primary_model_latency or 0.0,
            tokens_used=metrics.total_tokens,
            cost_usd=metrics.total_cost,
            quality_score=metrics.quality_score,
            confidence_score=metrics.confidence_score,
            session_id=metrics.session_id,
            strategy_used=metrics.strategy_used,
            completed_successfully=not hasattr(final_response, 'error') or not final_response.error
        )
        
        # Record primary model performance
        await self._record_model_performance(primary_metric)
        
        # Record supporting model performances
        for model in routing_decision.supporting_models:
            supporting_latency = metrics.supporting_models_latency.get(model, 0.0)
            
            supporting_metric = PerformanceMetrics(
                model_name=model,
                domain=routing_decision.detected_domain,
                complexity=routing_decision.detected_complexity,
                latency_ms=supporting_latency,
                tokens_used=metrics.total_tokens // len(routing_decision.supporting_models + [routing_decision.primary_model]),
                cost_usd=metrics.total_cost * 0.3,  # Estimate supporting model cost
                quality_score=metrics.quality_score * 0.9,  # Slight discount for supporting
                confidence_score=metrics.confidence_score * 0.9,
                session_id=metrics.session_id,
                strategy_used=metrics.strategy_used,
                completed_successfully=not hasattr(final_response, 'error') or not final_response.error
            )
            
            await self._record_model_performance(supporting_metric)
        
        # Update rankings if needed
        await self._update_rankings_if_needed()
    
    async def _record_model_performance(self, metric: PerformanceMetrics) -> None:
        """Record performance metric for a specific model."""
        
        domain = metric.domain
        model = metric.model_name
        
        # Add to domain performance
        domain_perf = self.domain_performances[domain]
        
        if model not in domain_perf.model_performances:
            domain_perf.model_performances[model] = []
        
        domain_perf.model_performances[model].append(metric)
        
        # Maintain size limit
        if len(domain_perf.model_performances[model]) > self.max_metrics_per_model:
            domain_perf.model_performances[model] = domain_perf.model_performances[model][-self.max_metrics_per_model:]
        
        # Add to global metrics
        self.global_metrics.append(metric)
        if len(self.global_metrics) > self.max_metrics_per_model * 10:
            self.global_metrics = self.global_metrics[-self.max_metrics_per_model * 10:]
        
        # Update trends
        self.performance_trends[f"{model}_{domain.value}_quality"].append(metric.quality_score)
        self.performance_trends[f"{model}_{domain.value}_latency"].append(metric.latency_ms)
        
        # Maintain trend size limit
        for key in self.performance_trends:
            if len(self.performance_trends[key]) > 100:
                self.performance_trends[key] = self.performance_trends[key][-100:]
        
        # Update aggregated metrics for the domain
        await self._update_domain_aggregates(domain)
    
    async def _update_domain_aggregates(self, domain: QueryDomain) -> None:
        """Update aggregated metrics for a domain."""
        
        domain_perf = self.domain_performances[domain]
        
        for model, metrics_list in domain_perf.model_performances.items():
            if not metrics_list:
                continue
            
            # Calculate averages
            domain_perf.avg_quality_by_model[model] = sum(m.quality_score for m in metrics_list) / len(metrics_list)
            domain_perf.avg_latency_by_model[model] = sum(m.latency_ms for m in metrics_list) / len(metrics_list)
            domain_perf.avg_cost_by_model[model] = sum(m.cost_usd for m in metrics_list) / len(metrics_list)
            
            # Calculate success rate
            successful = sum(1 for m in metrics_list if m.completed_successfully)
            domain_perf.success_rate_by_model[model] = successful / len(metrics_list)
        
        # Find best performers
        if domain_perf.avg_quality_by_model:
            domain_perf.best_quality_model = max(
                domain_perf.avg_quality_by_model, 
                key=domain_perf.avg_quality_by_model.get
            )
        
        if domain_perf.avg_latency_by_model:
            domain_perf.best_speed_model = min(
                domain_perf.avg_latency_by_model, 
                key=domain_perf.avg_latency_by_model.get
            )
        
        if domain_perf.avg_cost_by_model:
            domain_perf.best_cost_model = min(
                domain_perf.avg_cost_by_model, 
                key=domain_perf.avg_cost_by_model.get
            )
        
        # Calculate best overall model (weighted score)
        overall_scores = {}
        for model in domain_perf.avg_quality_by_model:
            quality = domain_perf.avg_quality_by_model.get(model, 0.0)
            # Normalize latency (lower is better, so invert)
            latency = domain_perf.avg_latency_by_model.get(model, float('inf'))
            latency_score = 1.0 / (1.0 + latency / 1000.0)  # Convert to 0-1 scale
            # Normalize cost (lower is better, so invert)
            cost = domain_perf.avg_cost_by_model.get(model, float('inf'))
            cost_score = 1.0 / (1.0 + cost * 100)  # Convert to 0-1 scale
            success_rate = domain_perf.success_rate_by_model.get(model, 0.0)
            
            # Weighted overall score
            overall_scores[model] = (
                quality * 0.4 +
                latency_score * 0.2 +
                cost_score * 0.2 +
                success_rate * 0.2
            )
        
        if overall_scores:
            domain_perf.best_overall_model = max(overall_scores, key=overall_scores.get)
    
    async def _update_rankings_if_needed(self) -> None:
        """Update model rankings if enough time has passed."""
        
        current_time = time.time()
        if current_time - self.last_ranking_update < self.ranking_update_interval:
            return
        
        # Update rankings for each domain
        for domain in QueryDomain:
            domain_perf = self.domain_performances[domain]
            
            # Rank models by overall performance
            model_scores = {}
            for model in domain_perf.avg_quality_by_model:
                quality = domain_perf.avg_quality_by_model.get(model, 0.0)
                latency = domain_perf.avg_latency_by_model.get(model, float('inf'))
                cost = domain_perf.avg_cost_by_model.get(model, float('inf'))
                success_rate = domain_perf.success_rate_by_model.get(model, 0.0)
                
                # Calculate composite score
                latency_score = 1.0 / (1.0 + latency / 1000.0)
                cost_score = 1.0 / (1.0 + cost * 100)
                
                model_scores[model] = (
                    quality * 0.4 +
                    latency_score * 0.2 +
                    cost_score * 0.2 +
                    success_rate * 0.2
                )
            
            # Sort by score and store ranking
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            self.model_rankings[domain] = [model for model, _ in ranked_models]
        
        self.last_ranking_update = current_time
    
    async def get_similar_sessions(
        self, 
        domain: QueryDomain, 
        complexity: QueryComplexity,
        limit: int = 10
    ) -> List[PerformanceMetrics]:
        """
        Get performance metrics from similar sessions.
        
        Args:
            domain: Query domain to match
            complexity: Query complexity to match
            limit: Maximum number of sessions to return
            
        Returns:
            List of similar performance metrics
        """
        
        similar_sessions = []
        
        # Search in domain-specific metrics
        domain_perf = self.domain_performances.get(domain)
        if domain_perf:
            for model_metrics in domain_perf.model_performances.values():
                for metric in model_metrics:
                    if metric.complexity == complexity:
                        similar_sessions.append(metric)
        
        # Sort by timestamp (most recent first) and limit
        similar_sessions.sort(key=lambda x: x.timestamp, reverse=True)
        return similar_sessions[:limit]
    
    def get_best_models_for_domain(
        self, 
        domain: QueryDomain, 
        criteria: str = "overall"
    ) -> List[str]:
        """
        Get best performing models for a specific domain.
        
        Args:
            domain: Query domain
            criteria: Performance criteria ("quality", "speed", "cost", "overall")
            
        Returns:
            List of model names ranked by performance
        """
        
        domain_perf = self.domain_performances.get(domain)
        if not domain_perf:
            return []
        
        if criteria == "quality" and domain_perf.best_quality_model:
            ranking = self.model_rankings.get(domain, [])
            # Ensure quality model is first
            result = [domain_perf.best_quality_model]
            result.extend([m for m in ranking if m != domain_perf.best_quality_model])
            return result
        
        elif criteria == "speed" and domain_perf.best_speed_model:
            ranking = self.model_rankings.get(domain, [])
            result = [domain_perf.best_speed_model]
            result.extend([m for m in ranking if m != domain_perf.best_speed_model])
            return result
        
        elif criteria == "cost" and domain_perf.best_cost_model:
            ranking = self.model_rankings.get(domain, [])
            result = [domain_perf.best_cost_model]
            result.extend([m for m in ranking if m != domain_perf.best_cost_model])
            return result
        
        else:  # overall or fallback
            return self.model_rankings.get(domain, [])
    
    def get_model_performance_summary(self, model: str) -> Dict[str, any]:
        """Get performance summary for a specific model."""
        
        summary = {
            "model": model,
            "domains": {},
            "global_stats": {
                "total_sessions": 0,
                "avg_quality": 0.0,
                "avg_latency": 0.0,
                "avg_cost": 0.0,
                "success_rate": 0.0
            }
        }
        
        all_metrics = []
        
        # Collect metrics across all domains
        for domain, domain_perf in self.domain_performances.items():
            if model in domain_perf.model_performances:
                metrics = domain_perf.model_performances[model]
                all_metrics.extend(metrics)
                
                # Domain-specific stats
                summary["domains"][domain.value] = {
                    "sessions": len(metrics),
                    "avg_quality": sum(m.quality_score for m in metrics) / len(metrics),
                    "avg_latency": sum(m.latency_ms for m in metrics) / len(metrics),
                    "avg_cost": sum(m.cost_usd for m in metrics) / len(metrics),
                    "success_rate": sum(1 for m in metrics if m.completed_successfully) / len(metrics)
                }
        
        # Global stats
        if all_metrics:
            summary["global_stats"] = {
                "total_sessions": len(all_metrics),
                "avg_quality": sum(m.quality_score for m in all_metrics) / len(all_metrics),
                "avg_latency": sum(m.latency_ms for m in all_metrics) / len(all_metrics),
                "avg_cost": sum(m.cost_usd for m in all_metrics) / len(all_metrics),
                "success_rate": sum(1 for m in all_metrics if m.completed_successfully) / len(all_metrics)
            }
        
        return summary
    
    def get_performance_trends(self, model: str, domain: QueryDomain) -> Dict[str, List[float]]:
        """Get performance trends for a model in a specific domain."""
        
        trends = {}
        quality_key = f"{model}_{domain.value}_quality"
        latency_key = f"{model}_{domain.value}_latency"
        
        if quality_key in self.performance_trends:
            trends["quality"] = self.performance_trends[quality_key]
        
        if latency_key in self.performance_trends:
            trends["latency"] = self.performance_trends[latency_key]
        
        return trends
    
    def get_global_performance_stats(self) -> Dict[str, any]:
        """Get global performance statistics across all models and domains."""
        
        stats = {
            "total_sessions": len(self.global_metrics),
            "total_models": len(set(m.model_name for m in self.global_metrics)),
            "domains_covered": len([d for d in self.domain_performances.values() if d.model_performances]),
            "avg_session_quality": 0.0,
            "avg_session_latency": 0.0,
            "avg_session_cost": 0.0,
            "global_success_rate": 0.0,
            "best_performers": {}
        }
        
        if self.global_metrics:
            stats["avg_session_quality"] = sum(m.quality_score for m in self.global_metrics) / len(self.global_metrics)
            stats["avg_session_latency"] = sum(m.latency_ms for m in self.global_metrics) / len(self.global_metrics)
            stats["avg_session_cost"] = sum(m.cost_usd for m in self.global_metrics) / len(self.global_metrics)
            stats["global_success_rate"] = sum(1 for m in self.global_metrics if m.completed_successfully) / len(self.global_metrics)
        
        # Best performers by domain
        for domain, domain_perf in self.domain_performances.items():
            if domain_perf.best_overall_model:
                stats["best_performers"][domain.value] = domain_perf.best_overall_model
        
        return stats
    
    async def record_user_feedback(
        self, 
        session_id: str, 
        satisfaction_score: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """
        Record user feedback for a session.
        
        Args:
            session_id: Session ID to update
            satisfaction_score: User satisfaction score (0.0 to 1.0)
            feedback_text: Optional feedback text
        """
        
        # Find metrics for this session
        for metric in self.global_metrics:
            if metric.session_id == session_id:
                metric.user_satisfaction = satisfaction_score
                
                # Update domain aggregates since satisfaction changed
                await self._update_domain_aggregates(metric.domain)