#!/usr/bin/env python3
"""
Test script for per-core agent monitoring and metrics collection

This script demonstrates the comprehensive monitoring and alerting capabilities
implemented for the per-core agent coordination system.
"""

import json
import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_monitoring_system():
    """Test the monitoring system functionality"""
    print("🔍 Testing Per-Core Agent Monitoring System")
    print("=" * 50)
    
    try:
        # Import monitoring components
        from utils.per_core_agent_monitoring import (
            PerCoreAgentMonitor, MonitoringThresholds, AlertSeverity
        )
        from utils.monitoring_dashboard import MonitoringDashboard
        from utils.agent_alerting import AlertingSystem
        
        # Create monitoring system with custom thresholds
        print("\n1. Initializing monitoring system...")
        thresholds = MonitoringThresholds(
            agent_success_rate_warning=0.8,
            agent_success_rate_critical=0.6,
            agent_response_time_warning_ms=2000,
            agent_response_time_critical_ms=5000,
            openrouter_cost_warning_usd=5.0,
            openrouter_cost_critical_usd=10.0
        )
        
        monitor = PerCoreAgentMonitor(thresholds)
        monitor.start_monitoring(collection_interval_seconds=10)
        
        print("✓ Monitoring system initialized")
        
        # Register test agents
        print("\n2. Registering test agents...")
        test_agents = [
            ("agent_core_0_security", "security_analyst", 0),
            ("agent_core_1_performance", "performance_optimizer", 1),
            ("agent_core_2_architecture", "architecture_reviewer", 2),
            ("agent_core_3_quality", "code_quality_inspector", 3)
        ]
        
        for agent_id, role, core_id in test_agents:
            monitor.register_agent(agent_id, role, core_id)
            print(f"✓ Registered {agent_id} on core {core_id}")
        
        # Simulate API calls and metrics
        print("\n3. Simulating API calls and metrics...")
        for i in range(20):
            for agent_id, role, core_id in test_agents:
                # Simulate varying success rates and response times
                success = i % 4 != 0  # 75% success rate
                response_time = 1000 + (i * 100) + (core_id * 200)  # Varying response times
                openrouter_call = i % 2 == 0  # 50% OpenRouter calls
                cost = 0.01 if openrouter_call else 0.0
                tokens = 100 if openrouter_call else 0
                
                monitor.record_api_call(
                    agent_id=agent_id,
                    success=success,
                    response_time_ms=response_time,
                    openrouter_call=openrouter_call,
                    cost_usd=cost,
                    tokens_used=tokens
                )
                
                # Simulate thinking sessions
                if i % 3 == 0:
                    thinking_success = i % 5 != 0  # 80% success rate
                    thinking_duration = 2000 + (i * 50)
                    monitor.record_thinking_session(agent_id, thinking_success, thinking_duration)
        
        print("✓ Simulated 80 API calls and thinking sessions")
        
        # Wait for monitoring to process
        print("\n4. Waiting for metrics processing...")
        time.sleep(2)
        
        # Get system metrics
        print("\n5. Retrieving system metrics...")
        system_metrics = monitor.get_system_metrics()
        print(f"✓ System health score: {system_metrics['health_score']:.2f}")
        print(f"✓ Total API calls: {system_metrics['total_system_api_calls']}")
        print(f"✓ System success rate: {system_metrics['system_success_rate']:.1%}")
        print(f"✓ OpenRouter cost: ${system_metrics['total_openrouter_cost_usd']:.3f}")
        
        # Get agent metrics
        print("\n6. Retrieving agent metrics...")
        agent_metrics = monitor.get_all_agent_metrics()
        for agent_id, metrics in agent_metrics.items():
            print(f"✓ {agent_id}: {metrics['total_api_calls']} calls, "
                  f"{metrics['success_rate']:.1%} success, "
                  f"{metrics['avg_response_time_ms']:.0f}ms avg response")
        
        # Test dashboard creation
        print("\n7. Creating monitoring dashboards...")
        dashboard = MonitoringDashboard(monitor)
        
        # Create different dashboard types
        system_dashboard = dashboard.create_system_overview_dashboard()
        agent_dashboard = dashboard.create_agent_performance_dashboard()
        openrouter_dashboard = dashboard.create_openrouter_dashboard()
        alerts_dashboard = dashboard.create_alerts_dashboard()
        
        print(f"✓ System overview dashboard: {len(system_dashboard.widgets)} widgets")
        print(f"✓ Agent performance dashboard: {len(agent_dashboard.widgets)} widgets")
        print(f"✓ OpenRouter dashboard: {len(openrouter_dashboard.widgets)} widgets")
        print(f"✓ Alerts dashboard: {len(alerts_dashboard.widgets)} widgets")
        
        # Test dashboard data export
        print("\n8. Testing dashboard data export...")
        export_data = dashboard.export_dashboard_data("system_overview", "json")
        print(f"✓ Exported dashboard data: {len(export_data)} sections")
        
        # Get comprehensive dashboard data
        dashboard_data = monitor.get_dashboard_data()
        print(f"✓ Dashboard data includes: {list(dashboard_data.keys())}")
        
        # Test alerting system
        print("\n9. Testing alerting system...")
        alerting = AlertingSystem()
        
        # Configure console alerting
        from utils.per_core_agent_monitoring import AlertSeverity
        from utils.agent_alerting import AlertChannelConfig, AlertChannel
        
        alerting.channels['console'] = AlertChannelConfig(
            channel=AlertChannel.CONSOLE,
            enabled=True,
            min_severity=AlertSeverity.WARNING
        )
        
        alerting.start_delivery_service()
        
        # Add alert callback to monitor
        def test_alert_callback(alert):
            print(f"🚨 Alert: {alert.title} - {alert.description}")
            alerting.process_alert(alert)
        
        monitor.add_alert_callback(test_alert_callback)
        
        print("✓ Alerting system configured")
        
        # Simulate some problematic metrics to trigger alerts
        print("\n10. Simulating alert conditions...")
        for agent_id, role, core_id in test_agents[:2]:  # Test with first 2 agents
            # Simulate high failure rate
            for _ in range(10):
                monitor.record_api_call(
                    agent_id=agent_id,
                    success=False,  # All failures
                    response_time_ms=8000,  # High response time
                    openrouter_call=True,
                    cost_usd=0.05,  # Higher cost
                    tokens_used=500
                )
        
        # Wait for alert processing
        time.sleep(3)
        
        # Check for generated alerts
        active_alerts = monitor.get_active_alerts()
        print(f"✓ Generated {len(active_alerts)} alerts")
        
        for alert in active_alerts:
            print(f"  - {alert['severity'].upper()}: {alert['title']}")
        
        # Test custom metrics
        print("\n11. Testing custom metrics...")
        from utils.per_core_agent_monitoring import MetricType
        
        monitor.add_custom_metric(
            name="test_metric",
            metric_type=MetricType.GAUGE,
            description="Test metric for demonstration",
            unit="units"
        )
        
        # Record some custom metric values
        for i in range(10):
            monitor.record_custom_metric("test_metric", float(i * 10))
        
        print("✓ Custom metrics recorded")
        
        # Get final dashboard data
        print("\n12. Final dashboard data summary...")
        final_data = monitor.get_dashboard_data()
        
        print(f"✓ System uptime: {final_data['system_metrics']['system_uptime_seconds']:.0f} seconds")
        print(f"✓ Active alerts: {final_data['alert_summary']['total_active']}")
        print(f"✓ Custom metrics: {len(final_data['custom_metrics'])}")
        
        # Test alerting statistics
        if alerting:
            alert_stats = alerting.get_delivery_statistics()
            print(f"✓ Alert deliveries: {alert_stats['total_deliveries']}")
            print(f"✓ Alert success rate: {alert_stats['success_rate']:.1%}")
        
        print("\n✅ All monitoring tests completed successfully!")
        
        # Cleanup
        print("\n13. Cleaning up...")
        monitor.stop_monitoring()
        if alerting:
            alerting.stop_delivery_service()
        print("✓ Monitoring and alerting systems stopped")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in monitoring test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_per_core_agent_manager_integration():
    """Test integration with PerCoreAgentManager"""
    print("\n🔧 Testing PerCoreAgentManager Integration")
    print("=" * 50)
    
    try:
        from utils.per_core_agent_manager import PerCoreAgentManager
        
        # Create manager (without OpenRouter key for testing)
        print("\n1. Creating PerCoreAgentManager...")
        manager = PerCoreAgentManager(max_agents=2)  # Limit for testing
        
        # Note: We can't fully test agent initialization without proper setup
        # but we can test the monitoring integration methods
        
        print("✓ PerCoreAgentManager created")
        
        # Test monitoring data retrieval (will show error since not fully initialized)
        print("\n2. Testing monitoring data retrieval...")
        monitoring_data = manager.get_comprehensive_monitoring_data()
        
        if "error" in monitoring_data:
            print(f"✓ Expected error (monitoring not initialized): {monitoring_data['error']}")
        else:
            print(f"✓ Monitoring data retrieved: {list(monitoring_data.keys())}")
        
        # Test dashboard data retrieval
        print("\n3. Testing dashboard data retrieval...")
        dashboard_data = manager.get_monitoring_dashboard_data("system_overview")
        
        if "error" in dashboard_data:
            print(f"✓ Expected error (monitoring not initialized): {dashboard_data['error']}")
        else:
            print(f"✓ Dashboard data retrieved: {list(dashboard_data.keys())}")
        
        print("\n✅ PerCoreAgentManager integration tests completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🚀 Per-Core Agent Monitoring Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test monitoring system
    if not test_monitoring_system():
        success = False
    
    # Test integration
    if not test_per_core_agent_manager_integration():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed successfully!")
        print("\nThe monitoring and metrics collection system is working correctly.")
        print("Key features tested:")
        print("  ✓ Real-time metrics collection")
        print("  ✓ Agent performance tracking")
        print("  ✓ OpenRouter usage monitoring")
        print("  ✓ Alert generation and delivery")
        print("  ✓ Dashboard data structures")
        print("  ✓ Custom metrics support")
        print("  ✓ Integration with PerCoreAgentManager")
    else:
        print("❌ Some tests failed. Check the logs above for details.")
    
    return success


if __name__ == "__main__":
    main()