"""
Monitoring Dashboard Data Structures

This module provides structured data models and utilities for creating
monitoring dashboards and interfaces for the per-core agent system.

Key Features:
- Dashboard widget data structures
- Chart and graph data formatting
- Real-time data streaming support
- Historical data aggregation
- Export capabilities for monitoring data
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Types of charts supported by the dashboard"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    AREA = "area"


class TimeRange(Enum):
    """Time ranges for dashboard data"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


@dataclass
class ChartDataPoint:
    """Single data point for charts"""
    timestamp: float
    value: Union[float, int]
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSeries:
    """Data series for charts"""
    name: str
    data_points: List[ChartDataPoint]
    color: Optional[str] = None
    line_style: str = "solid"  # solid, dashed, dotted
    fill: bool = False


@dataclass
class ChartConfig:
    """Chart configuration"""
    chart_type: ChartType
    title: str
    x_axis_label: str
    y_axis_label: str
    series: List[ChartSeries]
    width: int = 400
    height: int = 300
    show_legend: bool = True
    show_grid: bool = True
    time_range: Optional[TimeRange] = None
    refresh_interval_seconds: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "title": self.title,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "series": [
                {
                    "name": series.name,
                    "data_points": [
                        {
                            "timestamp": point.timestamp,
                            "value": point.value,
                            "label": point.label,
                            "metadata": point.metadata
                        }
                        for point in series.data_points
                    ],
                    "color": series.color,
                    "line_style": series.line_style,
                    "fill": series.fill
                }
                for series in self.series
            ],
            "width": self.width,
            "height": self.height,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "time_range": self.time_range.value if self.time_range else None,
            "refresh_interval_seconds": self.refresh_interval_seconds
        }


@dataclass
class GaugeWidget:
    """Gauge widget for displaying single metrics"""
    title: str
    current_value: float
    min_value: float
    max_value: float
    unit: str
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    color_scheme: str = "green-yellow-red"  # green-yellow-red, blue-white-red, custom
    
    def get_status(self) -> str:
        """Get status based on thresholds"""
        if self.critical_threshold and self.current_value >= self.critical_threshold:
            return "critical"
        elif self.warning_threshold and self.current_value >= self.warning_threshold:
            return "warning"
        else:
            return "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "color_scheme": self.color_scheme,
            "status": self.get_status()
        }


@dataclass
class StatusIndicator:
    """Status indicator widget"""
    title: str
    status: str  # healthy, warning, error, offline
    message: str
    last_updated: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "status": self.status,
            "message": self.message,
            "last_updated": self.last_updated,
            "details": self.details
        }


@dataclass
class TableWidget:
    """Table widget for displaying tabular data"""
    title: str
    headers: List[str]
    rows: List[List[Any]]
    sortable: bool = True
    filterable: bool = True
    max_rows: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows,
            "sortable": self.sortable,
            "filterable": self.filterable,
            "max_rows": self.max_rows
        }


@dataclass
class AlertWidget:
    """Alert widget for displaying active alerts"""
    title: str
    alerts: List[Dict[str, Any]]
    max_alerts: int = 10
    show_resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "alerts": self.alerts[:self.max_alerts],
            "max_alerts": self.max_alerts,
            "show_resolved": self.show_resolved,
            "total_alerts": len(self.alerts)
        }


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    title: str
    widgets: List[Dict[str, Any]]  # Widget configurations with positioning
    refresh_interval_seconds: int = 30
    auto_refresh: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "widgets": self.widgets,
            "refresh_interval_seconds": self.refresh_interval_seconds,
            "auto_refresh": self.auto_refresh,
            "generated_at": time.time()
        }


class MonitoringDashboard:
    """
    Main dashboard class that generates monitoring interfaces
    """
    
    def __init__(self, monitor: 'PerCoreAgentMonitor'):
        """
        Initialize dashboard with monitor instance
        
        Args:
            monitor: PerCoreAgentMonitor instance
        """
        self.monitor = monitor
        self._chart_cache: Dict[str, ChartConfig] = {}
        self._cache_ttl = 30  # Cache TTL in seconds
        self._last_cache_update = 0
    
    def create_system_overview_dashboard(self) -> DashboardLayout:
        """
        Create a system overview dashboard
        
        Returns:
            Dashboard layout for system overview
        """
        system_metrics = self.monitor.get_system_metrics()
        
        widgets = [
            # System health gauge
            {
                "type": "gauge",
                "position": {"x": 0, "y": 0, "width": 2, "height": 2},
                "config": GaugeWidget(
                    title="System Health Score",
                    current_value=system_metrics["health_score"],
                    min_value=0.0,
                    max_value=1.0,
                    unit="",
                    warning_threshold=0.7,
                    critical_threshold=0.5
                ).to_dict()
            },
            
            # Agent status indicators
            {
                "type": "status_grid",
                "position": {"x": 2, "y": 0, "width": 4, "height": 2},
                "config": {
                    "title": "Agent Status",
                    "indicators": [
                        StatusIndicator(
                            title="Total Agents",
                            status="info",
                            message=str(system_metrics["total_agents"])
                        ).to_dict(),
                        StatusIndicator(
                            title="Healthy Agents",
                            status="healthy" if system_metrics["healthy_agents"] == system_metrics["total_agents"] else "warning",
                            message=f"{system_metrics['healthy_agents']}/{system_metrics['total_agents']}"
                        ).to_dict(),
                        StatusIndicator(
                            title="API Success Rate",
                            status="healthy" if system_metrics["system_success_rate"] > 0.9 else "warning",
                            message=f"{system_metrics['system_success_rate']:.1%}"
                        ).to_dict(),
                        StatusIndicator(
                            title="OpenRouter Cost",
                            status="warning" if system_metrics["total_openrouter_cost_usd"] > 10 else "healthy",
                            message=f"${system_metrics['total_openrouter_cost_usd']:.2f}"
                        ).to_dict()
                    ]
                }
            },
            
            # Active alerts
            {
                "type": "alert",
                "position": {"x": 0, "y": 2, "width": 6, "height": 3},
                "config": AlertWidget(
                    title="Active Alerts",
                    alerts=self.monitor.get_active_alerts(),
                    max_alerts=10
                ).to_dict()
            },
            
            # API calls over time chart
            {
                "type": "chart",
                "position": {"x": 0, "y": 5, "width": 6, "height": 3},
                "config": self._create_api_calls_chart().to_dict()
            }
        ]
        
        return DashboardLayout(
            title="System Overview",
            widgets=widgets,
            refresh_interval_seconds=30
        )
    
    def create_agent_performance_dashboard(self) -> DashboardLayout:
        """
        Create an agent performance dashboard
        
        Returns:
            Dashboard layout for agent performance
        """
        agent_metrics = self.monitor.get_all_agent_metrics()
        
        widgets = [
            # Agent performance table
            {
                "type": "table",
                "position": {"x": 0, "y": 0, "width": 6, "height": 4},
                "config": self._create_agent_performance_table(agent_metrics).to_dict()
            },
            
            # Success rate comparison chart
            {
                "type": "chart",
                "position": {"x": 0, "y": 4, "width": 3, "height": 3},
                "config": self._create_success_rate_chart(agent_metrics).to_dict()
            },
            
            # Response time comparison chart
            {
                "type": "chart",
                "position": {"x": 3, "y": 4, "width": 3, "height": 3},
                "config": self._create_response_time_chart(agent_metrics).to_dict()
            },
            
            # Memory usage heatmap
            {
                "type": "chart",
                "position": {"x": 0, "y": 7, "width": 6, "height": 3},
                "config": self._create_memory_usage_heatmap(agent_metrics).to_dict()
            }
        ]
        
        return DashboardLayout(
            title="Agent Performance",
            widgets=widgets,
            refresh_interval_seconds=30
        )
    
    def create_openrouter_dashboard(self) -> DashboardLayout:
        """
        Create an OpenRouter usage dashboard
        
        Returns:
            Dashboard layout for OpenRouter monitoring
        """
        agent_metrics = self.monitor.get_all_agent_metrics()
        system_metrics = self.monitor.get_system_metrics()
        
        widgets = [
            # OpenRouter cost gauge
            {
                "type": "gauge",
                "position": {"x": 0, "y": 0, "width": 2, "height": 2},
                "config": GaugeWidget(
                    title="OpenRouter Cost",
                    current_value=system_metrics["total_openrouter_cost_usd"],
                    min_value=0.0,
                    max_value=50.0,
                    unit="USD",
                    warning_threshold=10.0,
                    critical_threshold=25.0
                ).to_dict()
            },
            
            # Token usage gauge
            {
                "type": "gauge",
                "position": {"x": 2, "y": 0, "width": 2, "height": 2},
                "config": GaugeWidget(
                    title="Tokens Used",
                    current_value=system_metrics["total_openrouter_tokens"],
                    min_value=0,
                    max_value=1000000,
                    unit="tokens",
                    warning_threshold=500000,
                    critical_threshold=800000
                ).to_dict()
            },
            
            # OpenRouter usage by agent
            {
                "type": "chart",
                "position": {"x": 4, "y": 0, "width": 2, "height": 2},
                "config": self._create_openrouter_usage_by_agent_chart(agent_metrics).to_dict()
            },
            
            # Cost over time
            {
                "type": "chart",
                "position": {"x": 0, "y": 2, "width": 6, "height": 3},
                "config": self._create_cost_over_time_chart().to_dict()
            },
            
            # OpenRouter performance table
            {
                "type": "table",
                "position": {"x": 0, "y": 5, "width": 6, "height": 3},
                "config": self._create_openrouter_performance_table(agent_metrics).to_dict()
            }
        ]
        
        return DashboardLayout(
            title="OpenRouter Usage",
            widgets=widgets,
            refresh_interval_seconds=30
        )
    
    def create_alerts_dashboard(self) -> DashboardLayout:
        """
        Create an alerts dashboard
        
        Returns:
            Dashboard layout for alerts monitoring
        """
        active_alerts = self.monitor.get_active_alerts()
        alert_history = self.monitor.get_alert_history(50)
        
        widgets = [
            # Alert summary
            {
                "type": "status_grid",
                "position": {"x": 0, "y": 0, "width": 6, "height": 2},
                "config": {
                    "title": "Alert Summary",
                    "indicators": [
                        StatusIndicator(
                            title="Active Alerts",
                            status="error" if len(active_alerts) > 0 else "healthy",
                            message=str(len(active_alerts))
                        ).to_dict(),
                        StatusIndicator(
                            title="Critical Alerts",
                            status="critical" if any(a["severity"] == "critical" for a in active_alerts) else "healthy",
                            message=str(len([a for a in active_alerts if a["severity"] == "critical"]))
                        ).to_dict(),
                        StatusIndicator(
                            title="Warning Alerts",
                            status="warning" if any(a["severity"] == "warning" for a in active_alerts) else "healthy",
                            message=str(len([a for a in active_alerts if a["severity"] == "warning"]))
                        ).to_dict(),
                        StatusIndicator(
                            title="Alerts Today",
                            status="info",
                            message=str(len([a for a in alert_history if a["timestamp"] > time.time() - 86400]))
                        ).to_dict()
                    ]
                }
            },
            
            # Active alerts table
            {
                "type": "alert",
                "position": {"x": 0, "y": 2, "width": 6, "height": 4},
                "config": AlertWidget(
                    title="Active Alerts",
                    alerts=active_alerts,
                    max_alerts=20
                ).to_dict()
            },
            
            # Alert history chart
            {
                "type": "chart",
                "position": {"x": 0, "y": 6, "width": 6, "height": 3},
                "config": self._create_alert_history_chart(alert_history).to_dict()
            }
        ]
        
        return DashboardLayout(
            title="Alerts Dashboard",
            widgets=widgets,
            refresh_interval_seconds=15  # More frequent updates for alerts
        )
    
    def get_real_time_data(self, widget_type: str, widget_id: str) -> Dict[str, Any]:
        """
        Get real-time data for a specific widget
        
        Args:
            widget_type: Type of widget
            widget_id: Widget identifier
            
        Returns:
            Real-time data for the widget
        """
        current_time = time.time()
        
        if widget_type == "system_health":
            system_metrics = self.monitor.get_system_metrics()
            return {
                "timestamp": current_time,
                "health_score": system_metrics["health_score"],
                "healthy_agents": system_metrics["healthy_agents"],
                "total_agents": system_metrics["total_agents"],
                "success_rate": system_metrics["system_success_rate"]
            }
        
        elif widget_type == "alerts":
            return {
                "timestamp": current_time,
                "active_alerts": self.monitor.get_active_alerts(),
                "alert_count": len(self.monitor.get_active_alerts())
            }
        
        elif widget_type == "agent_metrics":
            return {
                "timestamp": current_time,
                "agent_metrics": self.monitor.get_all_agent_metrics()
            }
        
        elif widget_type == "openrouter_usage":
            system_metrics = self.monitor.get_system_metrics()
            return {
                "timestamp": current_time,
                "total_cost": system_metrics["total_openrouter_cost_usd"],
                "total_tokens": system_metrics["total_openrouter_tokens"]
            }
        
        else:
            return {"timestamp": current_time, "error": f"Unknown widget type: {widget_type}"}
    
    def export_dashboard_data(self, dashboard_type: str, format: str = "json") -> Dict[str, Any]:
        """
        Export dashboard data for external use
        
        Args:
            dashboard_type: Type of dashboard to export
            format: Export format (json, csv, etc.)
            
        Returns:
            Exported dashboard data
        """
        if dashboard_type == "system_overview":
            dashboard = self.create_system_overview_dashboard()
        elif dashboard_type == "agent_performance":
            dashboard = self.create_agent_performance_dashboard()
        elif dashboard_type == "openrouter":
            dashboard = self.create_openrouter_dashboard()
        elif dashboard_type == "alerts":
            dashboard = self.create_alerts_dashboard()
        else:
            raise ValueError(f"Unknown dashboard type: {dashboard_type}")
        
        export_data = {
            "dashboard": dashboard.to_dict(),
            "raw_data": {
                "system_metrics": self.monitor.get_system_metrics(),
                "agent_metrics": self.monitor.get_all_agent_metrics(),
                "active_alerts": self.monitor.get_active_alerts(),
                "alert_history": self.monitor.get_alert_history(100)
            },
            "export_timestamp": time.time(),
            "format": format
        }
        
        return export_data
    
    def _create_api_calls_chart(self) -> ChartConfig:
        """Create API calls over time chart"""
        # This would typically pull from historical data
        # For now, create a sample chart structure
        current_time = time.time()
        
        # Generate sample data points for the last hour
        data_points = []
        for i in range(60):  # Last 60 minutes
            timestamp = current_time - (59 - i) * 60
            # This would come from actual metrics
            value = 10 + (i % 10)  # Sample data
            data_points.append(ChartDataPoint(timestamp=timestamp, value=value))
        
        series = ChartSeries(
            name="API Calls per Minute",
            data_points=data_points,
            color="#2196F3"
        )
        
        return ChartConfig(
            chart_type=ChartType.LINE,
            title="API Calls Over Time",
            x_axis_label="Time",
            y_axis_label="Calls per Minute",
            series=[series],
            time_range=TimeRange.LAST_HOUR
        )
    
    def _create_agent_performance_table(self, agent_metrics: Dict[str, Dict[str, Any]]) -> TableWidget:
        """Create agent performance table"""
        headers = [
            "Agent ID", "Role", "Core", "Success Rate", "API Calls", 
            "Avg Response Time", "Memory Usage", "Status"
        ]
        
        rows = []
        for agent_id, metrics in agent_metrics.items():
            rows.append([
                agent_id,
                metrics["role"],
                metrics["core_id"],
                f"{metrics['success_rate']:.1%}",
                metrics["total_api_calls"],
                f"{metrics['avg_response_time_ms']:.0f}ms",
                f"{metrics['memory_usage_mb']:.1f}MB",
                "Healthy" if metrics["is_healthy"] else "Unhealthy"
            ])
        
        return TableWidget(
            title="Agent Performance",
            headers=headers,
            rows=rows
        )
    
    def _create_success_rate_chart(self, agent_metrics: Dict[str, Dict[str, Any]]) -> ChartConfig:
        """Create success rate comparison chart"""
        data_points = []
        for agent_id, metrics in agent_metrics.items():
            data_points.append(ChartDataPoint(
                timestamp=time.time(),
                value=metrics["success_rate"],
                label=f"{metrics['role']} (Core {metrics['core_id']})"
            ))
        
        series = ChartSeries(
            name="Success Rate",
            data_points=data_points,
            color="#4CAF50"
        )
        
        return ChartConfig(
            chart_type=ChartType.BAR,
            title="Agent Success Rates",
            x_axis_label="Agent",
            y_axis_label="Success Rate",
            series=[series]
        )
    
    def _create_response_time_chart(self, agent_metrics: Dict[str, Dict[str, Any]]) -> ChartConfig:
        """Create response time comparison chart"""
        data_points = []
        for agent_id, metrics in agent_metrics.items():
            data_points.append(ChartDataPoint(
                timestamp=time.time(),
                value=metrics["avg_response_time_ms"],
                label=f"{metrics['role']} (Core {metrics['core_id']})"
            ))
        
        series = ChartSeries(
            name="Response Time",
            data_points=data_points,
            color="#FF9800"
        )
        
        return ChartConfig(
            chart_type=ChartType.BAR,
            title="Agent Response Times",
            x_axis_label="Agent",
            y_axis_label="Response Time (ms)",
            series=[series]
        )
    
    def _create_memory_usage_heatmap(self, agent_metrics: Dict[str, Dict[str, Any]]) -> ChartConfig:
        """Create memory usage heatmap"""
        data_points = []
        for agent_id, metrics in agent_metrics.items():
            data_points.append(ChartDataPoint(
                timestamp=time.time(),
                value=metrics["memory_usage_mb"],
                label=f"Core {metrics['core_id']}",
                metadata={"agent_id": agent_id, "role": metrics["role"]}
            ))
        
        series = ChartSeries(
            name="Memory Usage",
            data_points=data_points,
            color="#9C27B0"
        )
        
        return ChartConfig(
            chart_type=ChartType.HEATMAP,
            title="Memory Usage by Core",
            x_axis_label="CPU Core",
            y_axis_label="Memory Usage (MB)",
            series=[series]
        )
    
    def _create_openrouter_usage_by_agent_chart(self, agent_metrics: Dict[str, Dict[str, Any]]) -> ChartConfig:
        """Create OpenRouter usage by agent pie chart"""
        data_points = []
        for agent_id, metrics in agent_metrics.items():
            if metrics["openrouter_cost_usd"] > 0:
                data_points.append(ChartDataPoint(
                    timestamp=time.time(),
                    value=metrics["openrouter_cost_usd"],
                    label=f"{metrics['role']}"
                ))
        
        series = ChartSeries(
            name="OpenRouter Cost",
            data_points=data_points
        )
        
        return ChartConfig(
            chart_type=ChartType.PIE,
            title="OpenRouter Usage by Agent",
            x_axis_label="",
            y_axis_label="Cost (USD)",
            series=[series]
        )
    
    def _create_cost_over_time_chart(self) -> ChartConfig:
        """Create cost over time chart"""
        # This would typically pull from historical cost data
        current_time = time.time()
        
        data_points = []
        cumulative_cost = 0.0
        for i in range(24):  # Last 24 hours
            timestamp = current_time - (23 - i) * 3600
            hourly_cost = 0.5 + (i % 5) * 0.1  # Sample data
            cumulative_cost += hourly_cost
            data_points.append(ChartDataPoint(timestamp=timestamp, value=cumulative_cost))
        
        series = ChartSeries(
            name="Cumulative Cost",
            data_points=data_points,
            color="#F44336"
        )
        
        return ChartConfig(
            chart_type=ChartType.AREA,
            title="OpenRouter Cost Over Time",
            x_axis_label="Time",
            y_axis_label="Cumulative Cost (USD)",
            series=[series],
            time_range=TimeRange.LAST_24_HOURS
        )
    
    def _create_openrouter_performance_table(self, agent_metrics: Dict[str, Dict[str, Any]]) -> TableWidget:
        """Create OpenRouter performance table"""
        headers = [
            "Agent ID", "Role", "OpenRouter Calls", "Success Rate", 
            "Cost (USD)", "Tokens Used", "Avg Cost per Call"
        ]
        
        rows = []
        for agent_id, metrics in agent_metrics.items():
            if metrics["openrouter_calls"] > 0:
                avg_cost_per_call = metrics["openrouter_cost_usd"] / metrics["openrouter_calls"]
                rows.append([
                    agent_id,
                    metrics["role"],
                    metrics["openrouter_calls"],
                    f"{metrics['openrouter_success_rate']:.1%}",
                    f"${metrics['openrouter_cost_usd']:.3f}",
                    metrics["openrouter_tokens_used"],
                    f"${avg_cost_per_call:.4f}"
                ])
        
        return TableWidget(
            title="OpenRouter Performance by Agent",
            headers=headers,
            rows=rows
        )
    
    def _create_alert_history_chart(self, alert_history: List[Dict[str, Any]]) -> ChartConfig:
        """Create alert history chart"""
        # Group alerts by hour for the last 24 hours
        current_time = time.time()
        hourly_counts = {}
        
        for alert in alert_history:
            if alert["timestamp"] > current_time - 86400:  # Last 24 hours
                hour = int(alert["timestamp"] // 3600) * 3600
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        data_points = []
        for i in range(24):
            hour_timestamp = current_time - (23 - i) * 3600
            hour_key = int(hour_timestamp // 3600) * 3600
            count = hourly_counts.get(hour_key, 0)
            data_points.append(ChartDataPoint(timestamp=hour_timestamp, value=count))
        
        series = ChartSeries(
            name="Alerts per Hour",
            data_points=data_points,
            color="#FF5722"
        )
        
        return ChartConfig(
            chart_type=ChartType.BAR,
            title="Alert History (Last 24 Hours)",
            x_axis_label="Time",
            y_axis_label="Number of Alerts",
            series=[series],
            time_range=TimeRange.LAST_24_HOURS
        )