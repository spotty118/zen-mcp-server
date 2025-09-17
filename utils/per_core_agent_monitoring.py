"""
Per-Core Agent Monitoring and Metrics Collection

This module implements comprehensive monitoring and metrics collection for the
per-core agent coordination system, including agent performance tracking,
OpenRouter usage monitoring, system health metrics, and alerting capabilities.

Key Features:
- Real-time agent performance metrics collection
- OpenRouter API usage tracking and cost monitoring
- System health monitoring with configurable thresholds
- Alert generation for agent failures and performance degradation
- Dashboard data structures for monitoring interfaces
- Historical metrics storage and trend analysis
"""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


@dataclass
class Alert:
    """Represents a monitoring alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    agent_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    current_value: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "agent_id": self.agent_id,
            "metric_name": self.metric_name,
            "threshold_value": self.threshold_value,
            "current_value": self.current_value,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at
        }


@dataclass
class MetricDataPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and data storage"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))  # Keep last 1000 points
    labels: Dict[str, str] = field(default_factory=dict)
    
    def add_data_point(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a new data point to the metric"""
        point = MetricDataPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the most recent metric value"""
        if self.data_points:
            return self.data_points[-1].value
        return None
    
    def get_average_over_period(self, seconds: int) -> Optional[float]:
        """Get average value over the last N seconds"""
        cutoff_time = time.time() - seconds
        relevant_points = [
            point.value for point in self.data_points
            if point.timestamp >= cutoff_time
        ]
        
        if relevant_points:
            return sum(relevant_points) / len(relevant_points)
        return None


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for a single agent"""
    agent_id: str
    role: str
    core_id: int
    
    # API call metrics
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    avg_response_time_ms: float = 0.0
    
    # OpenRouter specific metrics
    openrouter_calls: int = 0
    openrouter_success_rate: float = 0.0
    openrouter_cost_usd: float = 0.0
    openrouter_tokens_used: int = 0
    
    # Thinking session metrics
    thinking_sessions_completed: int = 0
    avg_thinking_session_duration_ms: float = 0.0
    thinking_sessions_failed: int = 0
    
    # Health metrics
    last_activity_timestamp: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    is_healthy: bool = True
    consecutive_failures: int = 0
    
    # Performance trends (last 24 hours)
    hourly_api_calls: List[int] = field(default_factory=lambda: [0] * 24)
    hourly_success_rates: List[float] = field(default_factory=lambda: [1.0] * 24)
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_api_calls == 0:
            return 1.0
        return self.successful_api_calls / self.total_api_calls
    
    def get_failure_rate(self) -> float:
        """Calculate overall failure rate"""
        return 1.0 - self.get_success_rate()
    
    def update_hourly_metrics(self) -> None:
        """Update hourly trend metrics"""
        current_hour = int(time.time() // 3600) % 24
        self.hourly_api_calls[current_hour] = self.total_api_calls
        self.hourly_success_rates[current_hour] = self.get_success_rate()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "core_id": self.core_id,
            "total_api_calls": self.total_api_calls,
            "successful_api_calls": self.successful_api_calls,
            "failed_api_calls": self.failed_api_calls,
            "success_rate": self.get_success_rate(),
            "failure_rate": self.get_failure_rate(),
            "avg_response_time_ms": self.avg_response_time_ms,
            "openrouter_calls": self.openrouter_calls,
            "openrouter_success_rate": self.openrouter_success_rate,
            "openrouter_cost_usd": self.openrouter_cost_usd,
            "openrouter_tokens_used": self.openrouter_tokens_used,
            "thinking_sessions_completed": self.thinking_sessions_completed,
            "avg_thinking_session_duration_ms": self.avg_thinking_session_duration_ms,
            "thinking_sessions_failed": self.thinking_sessions_failed,
            "last_activity_timestamp": self.last_activity_timestamp,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "is_healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "hourly_api_calls": self.hourly_api_calls,
            "hourly_success_rates": self.hourly_success_rates
        }


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    total_agents: int = 0
    healthy_agents: int = 0
    unhealthy_agents: int = 0
    offline_agents: int = 0
    
    # System-wide API metrics
    total_system_api_calls: int = 0
    system_success_rate: float = 0.0
    avg_system_response_time_ms: float = 0.0
    
    # OpenRouter system metrics
    total_openrouter_cost_usd: float = 0.0
    total_openrouter_tokens: int = 0
    openrouter_rate_limit_hits: int = 0
    openrouter_circuit_breakers_open: int = 0
    
    # Resource utilization
    total_memory_usage_mb: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    
    # Alert metrics
    active_alerts: int = 0
    critical_alerts: int = 0
    warning_alerts: int = 0
    
    # Uptime and availability
    system_uptime_seconds: float = 0.0
    availability_percentage: float = 100.0
    
    def get_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        if self.total_agents == 0:
            return 0.0
        
        # Base score from healthy agents ratio
        health_ratio = self.healthy_agents / self.total_agents
        
        # Adjust for success rate
        success_factor = self.system_success_rate
        
        # Adjust for critical alerts (penalty)
        alert_penalty = min(0.3, self.critical_alerts * 0.1)
        
        # Calculate final score
        health_score = (health_ratio * 0.6 + success_factor * 0.4) - alert_penalty
        return max(0.0, min(1.0, health_score))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_agents": self.total_agents,
            "healthy_agents": self.healthy_agents,
            "unhealthy_agents": self.unhealthy_agents,
            "offline_agents": self.offline_agents,
            "total_system_api_calls": self.total_system_api_calls,
            "system_success_rate": self.system_success_rate,
            "avg_system_response_time_ms": self.avg_system_response_time_ms,
            "total_openrouter_cost_usd": self.total_openrouter_cost_usd,
            "total_openrouter_tokens": self.total_openrouter_tokens,
            "openrouter_rate_limit_hits": self.openrouter_rate_limit_hits,
            "openrouter_circuit_breakers_open": self.openrouter_circuit_breakers_open,
            "total_memory_usage_mb": self.total_memory_usage_mb,
            "avg_cpu_usage_percent": self.avg_cpu_usage_percent,
            "active_alerts": self.active_alerts,
            "critical_alerts": self.critical_alerts,
            "warning_alerts": self.warning_alerts,
            "system_uptime_seconds": self.system_uptime_seconds,
            "availability_percentage": self.availability_percentage,
            "health_score": self.get_health_score()
        }


@dataclass
class MonitoringThresholds:
    """Configurable thresholds for monitoring and alerting"""
    
    # Agent health thresholds
    agent_success_rate_warning: float = 0.7  # Below 70% success rate
    agent_success_rate_critical: float = 0.5  # Below 50% success rate
    agent_response_time_warning_ms: float = 5000  # Above 5 seconds
    agent_response_time_critical_ms: float = 10000  # Above 10 seconds
    agent_inactivity_warning_seconds: int = 300  # 5 minutes
    agent_inactivity_critical_seconds: int = 600  # 10 minutes
    agent_consecutive_failures_warning: int = 3
    agent_consecutive_failures_critical: int = 5
    
    # System health thresholds
    system_health_score_warning: float = 0.7  # Below 70%
    system_health_score_critical: float = 0.5  # Below 50%
    healthy_agents_ratio_warning: float = 0.8  # Below 80% healthy
    healthy_agents_ratio_critical: float = 0.6  # Below 60% healthy
    
    # OpenRouter thresholds
    openrouter_cost_warning_usd: float = 10.0  # Above $10/hour
    openrouter_cost_critical_usd: float = 25.0  # Above $25/hour
    openrouter_rate_limit_warning_ratio: float = 0.8  # Above 80% of rate limit
    openrouter_circuit_breaker_warning: int = 1  # Any circuit breaker open
    
    # Resource thresholds
    memory_usage_warning_mb: float = 1000  # Above 1GB per agent
    memory_usage_critical_mb: float = 2000  # Above 2GB per agent
    cpu_usage_warning_percent: float = 80  # Above 80% CPU
    cpu_usage_critical_percent: float = 95  # Above 95% CPU


class PerCoreAgentMonitor:
    """
    Comprehensive monitoring and metrics collection system for per-core agents
    """
    
    def __init__(self, thresholds: Optional[MonitoringThresholds] = None):
        """
        Initialize the monitoring system
        
        Args:
            thresholds: Custom monitoring thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or MonitoringThresholds()
        
        # Metrics storage
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.system_metrics = SystemHealthMetrics()
        self.custom_metrics: Dict[str, Metric] = {}
        
        # Alert management
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._metrics_lock = threading.RLock()
        self._alerts_lock = threading.RLock()
        
        # System startup time for uptime calculation
        self._startup_time = time.time()
        
        logger.info("PerCoreAgentMonitor initialized with monitoring thresholds")
    
    def start_monitoring(self, collection_interval_seconds: int = 30) -> None:
        """
        Start the monitoring system
        
        Args:
            collection_interval_seconds: How often to collect metrics
        """
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_active = True
        self._shutdown = False
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(collection_interval_seconds,),
            daemon=True,
            name="PerCoreAgentMonitor"
        )
        self._monitor_thread.start()
        
        logger.info(f"Started per-core agent monitoring with {collection_interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if not self._monitoring_active:
            return
        
        logger.info("Stopping per-core agent monitoring")
        self._shutdown = True
        self._monitoring_active = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        logger.info("Per-core agent monitoring stopped")
    
    def register_agent(self, agent_id: str, role: str, core_id: int) -> None:
        """
        Register an agent for monitoring
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role
            core_id: CPU core ID
        """
        with self._metrics_lock:
            if agent_id not in self.agent_metrics:
                self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                    agent_id=agent_id,
                    role=role,
                    core_id=core_id
                )
                logger.debug(f"Registered agent {agent_id} for monitoring")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from monitoring
        
        Args:
            agent_id: Agent identifier to unregister
        """
        with self._metrics_lock:
            if agent_id in self.agent_metrics:
                del self.agent_metrics[agent_id]
                logger.debug(f"Unregistered agent {agent_id} from monitoring")
        
        # Resolve any active alerts for this agent
        self._resolve_agent_alerts(agent_id)
    
    def update_agent_metrics(self, agent_id: str, metrics_update: Dict[str, Any]) -> None:
        """
        Update metrics for a specific agent
        
        Args:
            agent_id: Agent identifier
            metrics_update: Dictionary of metric updates
        """
        with self._metrics_lock:
            if agent_id not in self.agent_metrics:
                logger.warning(f"Agent {agent_id} not registered for monitoring")
                return
            
            agent_metrics = self.agent_metrics[agent_id]
            
            # Update metrics based on provided data
            for key, value in metrics_update.items():
                if hasattr(agent_metrics, key):
                    setattr(agent_metrics, key, value)
            
            # Update derived metrics
            agent_metrics.last_activity_timestamp = time.time()
            agent_metrics.update_hourly_metrics()
            
            logger.debug(f"Updated metrics for agent {agent_id}")
    
    def record_api_call(self, agent_id: str, success: bool, response_time_ms: float, 
                       openrouter_call: bool = False, cost_usd: float = 0.0, 
                       tokens_used: int = 0) -> None:
        """
        Record an API call for metrics tracking
        
        Args:
            agent_id: Agent that made the call
            success: Whether the call was successful
            response_time_ms: Response time in milliseconds
            openrouter_call: Whether this was an OpenRouter call
            cost_usd: Cost of the call in USD
            tokens_used: Number of tokens used
        """
        with self._metrics_lock:
            if agent_id not in self.agent_metrics:
                return
            
            metrics = self.agent_metrics[agent_id]
            
            # Update general API metrics
            metrics.total_api_calls += 1
            if success:
                metrics.successful_api_calls += 1
                metrics.consecutive_failures = 0
            else:
                metrics.failed_api_calls += 1
                metrics.consecutive_failures += 1
            
            # Update response time (rolling average)
            if metrics.total_api_calls == 1:
                metrics.avg_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                alpha = 0.1
                metrics.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * metrics.avg_response_time_ms
                )
            
            # Update OpenRouter specific metrics
            if openrouter_call:
                metrics.openrouter_calls += 1
                metrics.openrouter_cost_usd += cost_usd
                metrics.openrouter_tokens_used += tokens_used
                
                # Update OpenRouter success rate
                if metrics.openrouter_calls == 1:
                    metrics.openrouter_success_rate = 1.0 if success else 0.0
                else:
                    # Exponential moving average for success rate
                    alpha = 0.1
                    current_success = 1.0 if success else 0.0
                    metrics.openrouter_success_rate = (
                        alpha * current_success + 
                        (1 - alpha) * metrics.openrouter_success_rate
                    )
            
            metrics.last_activity_timestamp = time.time()
    
    def record_thinking_session(self, agent_id: str, success: bool, duration_ms: float) -> None:
        """
        Record a thinking session for metrics tracking
        
        Args:
            agent_id: Agent that performed the thinking session
            success: Whether the session was successful
            duration_ms: Duration in milliseconds
        """
        with self._metrics_lock:
            if agent_id not in self.agent_metrics:
                return
            
            metrics = self.agent_metrics[agent_id]
            
            if success:
                metrics.thinking_sessions_completed += 1
                
                # Update average duration (rolling average)
                if metrics.thinking_sessions_completed == 1:
                    metrics.avg_thinking_session_duration_ms = duration_ms
                else:
                    alpha = 0.1
                    metrics.avg_thinking_session_duration_ms = (
                        alpha * duration_ms + 
                        (1 - alpha) * metrics.avg_thinking_session_duration_ms
                    )
            else:
                metrics.thinking_sessions_failed += 1
            
            metrics.last_activity_timestamp = time.time()
    
    def add_custom_metric(self, name: str, metric_type: MetricType, description: str, 
                         unit: str, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Add a custom metric for tracking
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            unit: Unit of measurement
            labels: Optional labels for the metric
        """
        with self._metrics_lock:
            self.custom_metrics[name] = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                labels=labels or {}
            )
            logger.debug(f"Added custom metric: {name}")
    
    def record_custom_metric(self, name: str, value: float, 
                           labels: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value for a custom metric
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels for this data point
        """
        with self._metrics_lock:
            if name in self.custom_metrics:
                self.custom_metrics[name].add_data_point(value, labels)
    
    def get_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific agent
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent metrics dictionary or None if not found
        """
        with self._metrics_lock:
            if agent_id in self.agent_metrics:
                return self.agent_metrics[agent_id].to_dict()
            return None
    
    def get_all_agent_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all agents
        
        Returns:
            Dictionary mapping agent IDs to their metrics
        """
        with self._metrics_lock:
            return {
                agent_id: metrics.to_dict()
                for agent_id, metrics in self.agent_metrics.items()
            }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get overall system metrics
        
        Returns:
            System metrics dictionary
        """
        with self._metrics_lock:
            # Update system metrics from agent metrics
            self._update_system_metrics()
            return self.system_metrics.to_dict()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for monitoring interfaces
        
        Returns:
            Complete dashboard data structure
        """
        with self._metrics_lock, self._alerts_lock:
            return {
                "timestamp": time.time(),
                "system_metrics": self.get_system_metrics(),
                "agent_metrics": self.get_all_agent_metrics(),
                "active_alerts": {
                    alert_id: alert.to_dict()
                    for alert_id, alert in self.active_alerts.items()
                },
                "alert_summary": {
                    "total_active": len(self.active_alerts),
                    "critical": len([a for a in self.active_alerts.values() 
                                   if a.severity == AlertSeverity.CRITICAL]),
                    "warning": len([a for a in self.active_alerts.values() 
                                  if a.severity == AlertSeverity.WARNING]),
                    "info": len([a for a in self.active_alerts.values() 
                               if a.severity == AlertSeverity.INFO])
                },
                "custom_metrics": {
                    name: {
                        "description": metric.description,
                        "unit": metric.unit,
                        "latest_value": metric.get_latest_value(),
                        "avg_last_hour": metric.get_average_over_period(3600)
                    }
                    for name, metric in self.custom_metrics.items()
                },
                "thresholds": {
                    "agent_success_rate_warning": self.thresholds.agent_success_rate_warning,
                    "agent_success_rate_critical": self.thresholds.agent_success_rate_critical,
                    "system_health_score_warning": self.thresholds.system_health_score_warning,
                    "system_health_score_critical": self.thresholds.system_health_score_critical,
                    "openrouter_cost_warning_usd": self.thresholds.openrouter_cost_warning_usd,
                    "openrouter_cost_critical_usd": self.thresholds.openrouter_cost_critical_usd
                }
            }
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add a callback function to be called when alerts are generated
        
        Args:
            callback: Function to call with Alert object
        """
        self.alert_callbacks.append(callback)
        logger.debug("Added alert callback")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts
        
        Returns:
            List of active alert dictionaries
        """
        with self._alerts_lock:
            return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alert dictionaries
        """
        with self._alerts_lock:
            return [alert.to_dict() for alert in list(self.alert_history)[-limit:]]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Manually resolve an alert
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if alert was resolved, False if not found
        """
        with self._alerts_lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = time.time()
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Manually resolved alert: {alert_id}")
                return True
            return False
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop that runs in background thread"""
        logger.info(f"Starting monitoring loop with {interval_seconds}s interval")
        
        while not self._shutdown:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for alerts
                self._check_and_generate_alerts()
                
                # Sleep for the specified interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
        
        logger.info("Monitoring loop stopped")
    
    def _update_system_metrics(self) -> None:
        """Update system-wide metrics from agent metrics"""
        with self._metrics_lock:
            # Reset system metrics
            self.system_metrics = SystemHealthMetrics()
            
            if not self.agent_metrics:
                return
            
            # Count agents by health status
            healthy_count = 0
            total_api_calls = 0
            total_success_calls = 0
            total_response_time = 0.0
            total_openrouter_cost = 0.0
            total_openrouter_tokens = 0
            total_memory = 0.0
            total_cpu = 0.0
            
            for metrics in self.agent_metrics.values():
                self.system_metrics.total_agents += 1
                
                if metrics.is_healthy:
                    healthy_count += 1
                else:
                    self.system_metrics.unhealthy_agents += 1
                
                # Aggregate API metrics
                total_api_calls += metrics.total_api_calls
                total_success_calls += metrics.successful_api_calls
                total_response_time += metrics.avg_response_time_ms
                
                # Aggregate OpenRouter metrics
                total_openrouter_cost += metrics.openrouter_cost_usd
                total_openrouter_tokens += metrics.openrouter_tokens_used
                
                # Aggregate resource metrics
                total_memory += metrics.memory_usage_mb
                total_cpu += metrics.cpu_usage_percent
            
            # Calculate system-wide metrics
            self.system_metrics.healthy_agents = healthy_count
            self.system_metrics.total_system_api_calls = total_api_calls
            
            if total_api_calls > 0:
                self.system_metrics.system_success_rate = total_success_calls / total_api_calls
            
            if self.system_metrics.total_agents > 0:
                self.system_metrics.avg_system_response_time_ms = (
                    total_response_time / self.system_metrics.total_agents
                )
                self.system_metrics.avg_cpu_usage_percent = (
                    total_cpu / self.system_metrics.total_agents
                )
            
            self.system_metrics.total_openrouter_cost_usd = total_openrouter_cost
            self.system_metrics.total_openrouter_tokens = total_openrouter_tokens
            self.system_metrics.total_memory_usage_mb = total_memory
            
            # Calculate uptime
            self.system_metrics.system_uptime_seconds = time.time() - self._startup_time
            
            # Update alert counts
            with self._alerts_lock:
                self.system_metrics.active_alerts = len(self.active_alerts)
                self.system_metrics.critical_alerts = len([
                    a for a in self.active_alerts.values() 
                    if a.severity == AlertSeverity.CRITICAL
                ])
                self.system_metrics.warning_alerts = len([
                    a for a in self.active_alerts.values() 
                    if a.severity == AlertSeverity.WARNING
                ])
    
    def _check_and_generate_alerts(self) -> None:
        """Check metrics against thresholds and generate alerts"""
        with self._metrics_lock:
            # Check agent-level alerts
            for agent_id, metrics in self.agent_metrics.items():
                self._check_agent_alerts(agent_id, metrics)
            
            # Check system-level alerts
            self._check_system_alerts()
    
    def _check_agent_alerts(self, agent_id: str, metrics: AgentPerformanceMetrics) -> None:
        """Check and generate alerts for a specific agent"""
        current_time = time.time()
        
        # Check success rate
        success_rate = metrics.get_success_rate()
        if success_rate < self.thresholds.agent_success_rate_critical:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} Critical Success Rate",
                description=f"Agent success rate ({success_rate:.1%}) is below critical threshold ({self.thresholds.agent_success_rate_critical:.1%})",
                agent_id=agent_id,
                metric_name="success_rate",
                threshold_value=self.thresholds.agent_success_rate_critical,
                current_value=success_rate
            )
        elif success_rate < self.thresholds.agent_success_rate_warning:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Agent {agent_id} Low Success Rate",
                description=f"Agent success rate ({success_rate:.1%}) is below warning threshold ({self.thresholds.agent_success_rate_warning:.1%})",
                agent_id=agent_id,
                metric_name="success_rate",
                threshold_value=self.thresholds.agent_success_rate_warning,
                current_value=success_rate
            )
        
        # Check response time
        if metrics.avg_response_time_ms > self.thresholds.agent_response_time_critical_ms:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} High Response Time",
                description=f"Agent response time ({metrics.avg_response_time_ms:.0f}ms) exceeds critical threshold ({self.thresholds.agent_response_time_critical_ms:.0f}ms)",
                agent_id=agent_id,
                metric_name="response_time_ms",
                threshold_value=self.thresholds.agent_response_time_critical_ms,
                current_value=metrics.avg_response_time_ms
            )
        elif metrics.avg_response_time_ms > self.thresholds.agent_response_time_warning_ms:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Agent {agent_id} Elevated Response Time",
                description=f"Agent response time ({metrics.avg_response_time_ms:.0f}ms) exceeds warning threshold ({self.thresholds.agent_response_time_warning_ms:.0f}ms)",
                agent_id=agent_id,
                metric_name="response_time_ms",
                threshold_value=self.thresholds.agent_response_time_warning_ms,
                current_value=metrics.avg_response_time_ms
            )
        
        # Check inactivity
        inactivity_seconds = current_time - metrics.last_activity_timestamp
        if inactivity_seconds > self.thresholds.agent_inactivity_critical_seconds:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} Inactive",
                description=f"Agent has been inactive for {inactivity_seconds:.0f} seconds (critical threshold: {self.thresholds.agent_inactivity_critical_seconds}s)",
                agent_id=agent_id,
                metric_name="inactivity_seconds",
                threshold_value=self.thresholds.agent_inactivity_critical_seconds,
                current_value=inactivity_seconds
            )
        elif inactivity_seconds > self.thresholds.agent_inactivity_warning_seconds:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Agent {agent_id} Potentially Inactive",
                description=f"Agent has been inactive for {inactivity_seconds:.0f} seconds (warning threshold: {self.thresholds.agent_inactivity_warning_seconds}s)",
                agent_id=agent_id,
                metric_name="inactivity_seconds",
                threshold_value=self.thresholds.agent_inactivity_warning_seconds,
                current_value=inactivity_seconds
            )
        
        # Check consecutive failures
        if metrics.consecutive_failures >= self.thresholds.agent_consecutive_failures_critical:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} Multiple Consecutive Failures",
                description=f"Agent has {metrics.consecutive_failures} consecutive failures (critical threshold: {self.thresholds.agent_consecutive_failures_critical})",
                agent_id=agent_id,
                metric_name="consecutive_failures",
                threshold_value=self.thresholds.agent_consecutive_failures_critical,
                current_value=metrics.consecutive_failures
            )
        elif metrics.consecutive_failures >= self.thresholds.agent_consecutive_failures_warning:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Agent {agent_id} Consecutive Failures",
                description=f"Agent has {metrics.consecutive_failures} consecutive failures (warning threshold: {self.thresholds.agent_consecutive_failures_warning})",
                agent_id=agent_id,
                metric_name="consecutive_failures",
                threshold_value=self.thresholds.agent_consecutive_failures_warning,
                current_value=metrics.consecutive_failures
            )
        
        # Check memory usage
        if metrics.memory_usage_mb > self.thresholds.memory_usage_critical_mb:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title=f"Agent {agent_id} High Memory Usage",
                description=f"Agent memory usage ({metrics.memory_usage_mb:.0f}MB) exceeds critical threshold ({self.thresholds.memory_usage_critical_mb:.0f}MB)",
                agent_id=agent_id,
                metric_name="memory_usage_mb",
                threshold_value=self.thresholds.memory_usage_critical_mb,
                current_value=metrics.memory_usage_mb
            )
        elif metrics.memory_usage_mb > self.thresholds.memory_usage_warning_mb:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title=f"Agent {agent_id} Elevated Memory Usage",
                description=f"Agent memory usage ({metrics.memory_usage_mb:.0f}MB) exceeds warning threshold ({self.thresholds.memory_usage_warning_mb:.0f}MB)",
                agent_id=agent_id,
                metric_name="memory_usage_mb",
                threshold_value=self.thresholds.memory_usage_warning_mb,
                current_value=metrics.memory_usage_mb
            )
    
    def _check_system_alerts(self) -> None:
        """Check and generate system-level alerts"""
        # Check system health score
        health_score = self.system_metrics.get_health_score()
        if health_score < self.thresholds.system_health_score_critical:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="System Health Critical",
                description=f"System health score ({health_score:.1%}) is below critical threshold ({self.thresholds.system_health_score_critical:.1%})",
                metric_name="system_health_score",
                threshold_value=self.thresholds.system_health_score_critical,
                current_value=health_score
            )
        elif health_score < self.thresholds.system_health_score_warning:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title="System Health Degraded",
                description=f"System health score ({health_score:.1%}) is below warning threshold ({self.thresholds.system_health_score_warning:.1%})",
                metric_name="system_health_score",
                threshold_value=self.thresholds.system_health_score_warning,
                current_value=health_score
            )
        
        # Check healthy agents ratio
        if self.system_metrics.total_agents > 0:
            healthy_ratio = self.system_metrics.healthy_agents / self.system_metrics.total_agents
            if healthy_ratio < self.thresholds.healthy_agents_ratio_critical:
                self._create_alert(
                    severity=AlertSeverity.CRITICAL,
                    title="Critical Agent Availability",
                    description=f"Only {healthy_ratio:.1%} of agents are healthy (critical threshold: {self.thresholds.healthy_agents_ratio_critical:.1%})",
                    metric_name="healthy_agents_ratio",
                    threshold_value=self.thresholds.healthy_agents_ratio_critical,
                    current_value=healthy_ratio
                )
            elif healthy_ratio < self.thresholds.healthy_agents_ratio_warning:
                self._create_alert(
                    severity=AlertSeverity.WARNING,
                    title="Reduced Agent Availability",
                    description=f"Only {healthy_ratio:.1%} of agents are healthy (warning threshold: {self.thresholds.healthy_agents_ratio_warning:.1%})",
                    metric_name="healthy_agents_ratio",
                    threshold_value=self.thresholds.healthy_agents_ratio_warning,
                    current_value=healthy_ratio
                )
        
        # Check OpenRouter cost
        if self.system_metrics.total_openrouter_cost_usd > self.thresholds.openrouter_cost_critical_usd:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="High OpenRouter Costs",
                description=f"OpenRouter costs (${self.system_metrics.total_openrouter_cost_usd:.2f}) exceed critical threshold (${self.thresholds.openrouter_cost_critical_usd:.2f})",
                metric_name="openrouter_cost_usd",
                threshold_value=self.thresholds.openrouter_cost_critical_usd,
                current_value=self.system_metrics.total_openrouter_cost_usd
            )
        elif self.system_metrics.total_openrouter_cost_usd > self.thresholds.openrouter_cost_warning_usd:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title="Elevated OpenRouter Costs",
                description=f"OpenRouter costs (${self.system_metrics.total_openrouter_cost_usd:.2f}) exceed warning threshold (${self.thresholds.openrouter_cost_warning_usd:.2f})",
                metric_name="openrouter_cost_usd",
                threshold_value=self.thresholds.openrouter_cost_warning_usd,
                current_value=self.system_metrics.total_openrouter_cost_usd
            )
    
    def _create_alert(self, severity: AlertSeverity, title: str, description: str,
                     agent_id: Optional[str] = None, metric_name: Optional[str] = None,
                     threshold_value: Optional[float] = None, current_value: Optional[float] = None) -> None:
        """Create a new alert if it doesn't already exist"""
        # Create a unique alert key based on the alert characteristics
        alert_key = f"{agent_id or 'system'}_{metric_name or 'general'}_{severity.value}"
        
        with self._alerts_lock:
            # Check if this alert already exists
            if alert_key in self.active_alerts:
                # Update existing alert with new values
                existing_alert = self.active_alerts[alert_key]
                existing_alert.current_value = current_value
                existing_alert.timestamp = time.time()
                return
            
            # Create new alert
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                severity=severity,
                title=title,
                description=description,
                agent_id=agent_id,
                metric_name=metric_name,
                threshold_value=threshold_value,
                current_value=current_value
            )
            
            self.active_alerts[alert_key] = alert
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error calling alert callback: {e}")
            
            logger.warning(f"Generated {severity.value} alert: {title}")
    
    def _resolve_agent_alerts(self, agent_id: str) -> None:
        """Resolve all active alerts for a specific agent"""
        with self._alerts_lock:
            alerts_to_resolve = [
                (key, alert) for key, alert in self.active_alerts.items()
                if alert.agent_id == agent_id
            ]
            
            for alert_key, alert in alerts_to_resolve:
                alert.resolved = True
                alert.resolved_at = time.time()
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_key]
                
                logger.info(f"Auto-resolved alert for unregistered agent {agent_id}: {alert.title}")


# Global monitor instance
_global_monitor: Optional[PerCoreAgentMonitor] = None


def get_per_core_agent_monitor() -> PerCoreAgentMonitor:
    """Get the global per-core agent monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerCoreAgentMonitor()
    return _global_monitor


def initialize_monitoring(thresholds: Optional[MonitoringThresholds] = None,
                         collection_interval_seconds: int = 30) -> PerCoreAgentMonitor:
    """
    Initialize the global monitoring system
    
    Args:
        thresholds: Custom monitoring thresholds
        collection_interval_seconds: Metrics collection interval
        
    Returns:
        Initialized monitor instance
    """
    global _global_monitor
    _global_monitor = PerCoreAgentMonitor(thresholds)
    _global_monitor.start_monitoring(collection_interval_seconds)
    return _global_monitor


def shutdown_monitoring() -> None:
    """Shutdown the global monitoring system"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
        _global_monitor = None