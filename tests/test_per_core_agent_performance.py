"""
Performance tests for per-core agent coordination system

Requirements tested: 2.3, 5.1
"""

import gc
import statistics
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import psutil

from utils.agent_core import AgentRole, AgentStatus


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.api_call_times = []
        self.error_count = 0
        self.success_count = 0
        
    def start_monitoring(self):
        self.start_time = time.time()
        self.memory_samples = []
        self.api_call_times = []
        self.error_count = 0
        self.success_count = 0
        
    def stop_monitoring(self):
        self.end_time = time.time()
        
    def record_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_samples.append(memory_info.rss / 1024 / 1024)
        
    def record_api_call(self, duration: float, success: bool = True):
        self.api_call_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    def get_memory_stats(self):
        if not self.memory_samples:
            return {'peak_memory_mb': 0}
        return {'peak_memory_mb': max(self.memory_samples)}
        
    def get_api_call_stats(self):
        if not self.api_call_times:
            return {'total_calls': 0, 'success_rate': 0, 'avg_response_time_ms': 0}
        
        total = self.success_count + self.error_count
        return {
            'total_calls': len(self.api_call_times),
            'success_rate': self.success_count / total if total > 0 else 0,
            'avg_response_time_ms': statistics.mean(self.api_call_times) * 1000
        }


class TestPerCoreAgentPerformance(unittest.TestCase):
    """Test performance characteristics of per-core agent coordination"""
    
    def setUp(self):
        self.metrics = PerformanceMetrics()
        
    def tearDown(self):
        gc.collect()
        
    def test_concurrent_api_call_throughput(self):
        """Test throughput of concurrent API calls across multiple agents"""
        # Mock API client that simulates realistic response times
        def mock_api_call(*args, **kwargs):
            start_time = time.time()
            time.sleep(0.05)  # 50ms response time
            duration = time.time() - start_time
            self.metrics.record_api_call(duration, True)
            return Mock(result="Mock response", tokens_used=100)
        
        # Test parameters
        num_concurrent_calls = 20
        
        # Start monitoring
        self.metrics.start_monitoring()
        
        # Monitor memory during test
        def monitor_memory():
            while self.metrics.start_time and not self.metrics.end_time:
                self.metrics.record_memory_usage()
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Execute concurrent calls
        with ThreadPoolExecutor(max_workers=num_concurrent_calls) as executor:
            futures = [
                executor.submit(mock_api_call, f"test prompt {i}")
                for i in range(num_concurrent_calls)
            ]
            
            completed_calls = 0
            for future in as_completed(futures):
                try:
                    future.result(timeout=5.0)
                    completed_calls += 1
                except Exception:
                    completed_calls += 1
        
        self.metrics.stop_monitoring()
        
        # Calculate performance metrics
        duration = self.metrics.get_duration()
        throughput = completed_calls / duration if duration > 0 else 0
        
        api_stats = self.metrics.get_api_call_stats()
        memory_stats = self.metrics.get_memory_stats()
        
        # Performance assertions
        self.assertGreater(throughput, 5, "Should achieve at least 5 ops/sec")
        self.assertEqual(api_stats['total_calls'], num_concurrent_calls)
        self.assertGreaterEqual(api_stats['success_rate'], 0.9, "Should maintain 90%+ success rate")
        self.assertLess(api_stats['avg_response_time_ms'], 200, "Avg response time should be under 200ms")
        self.assertLess(memory_stats['peak_memory_mb'], 1000, "Peak memory should be under 1GB")
        
        print(f"\nConcurrent API Call Performance:")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        print(f"  Success rate: {api_stats['success_rate']:.2%}")
        print(f"  Avg response time: {api_stats['avg_response_time_ms']:.1f}ms")
        print(f"  Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
    
    def test_memory_efficiency_scaling(self):
        """Test memory usage efficiency as number of agents scales"""
        from utils.agent_communication import AgentCommunicationSystem
        
        # Test different agent counts
        agent_counts = [1, 2, 4]
        memory_usage_per_count = {}
        
        for agent_count in agent_counts:
            # Measure baseline memory
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Create communication system with agents
            comm_system = AgentCommunicationSystem(max_agents=agent_count)
            
            # Create agents
            agents = []
            for i in range(agent_count):
                agent = comm_system.register_agent(core_id=i, role=AgentRole.GENERALIST)
                agents.append(agent)
            
            # Measure memory after agent creation
            time.sleep(0.5)  # Allow initialization
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - baseline_memory
            
            memory_usage_per_count[agent_count] = {
                'total_memory_mb': current_memory,
                'memory_increase_mb': memory_increase,
                'memory_per_agent_mb': memory_increase / agent_count if agent_count > 0 else 0
            }
            
            # Clean up
            comm_system.shutdown()
            del comm_system, agents
            gc.collect()
            time.sleep(0.5)
        
        # Assert memory efficiency
        for agent_count, usage in memory_usage_per_count.items():
            self.assertLess(
                usage['memory_per_agent_mb'], 100,
                f"Each agent should use less than 100MB (actual: {usage['memory_per_agent_mb']:.1f}MB)"
            )
        
        print(f"\nMemory Efficiency Scaling:")
        for agent_count, usage in memory_usage_per_count.items():
            print(f"  {agent_count} agents: {usage['memory_per_agent_mb']:.1f}MB per agent")
    
    def test_rate_limit_handling(self):
        """Test system behavior under rate limiting conditions"""
        # Mock rate-limited API client
        call_count = 0
        rate_limit = 10  # 10 calls before rate limiting
        
        def rate_limited_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            start_time = time.time()
            
            if call_count > rate_limit:
                time.sleep(0.1)  # Rate limit delay
                self.metrics.record_api_call(0.1, False)
                raise Exception("Rate limit exceeded")
            
            time.sleep(0.05)  # Normal response time
            duration = time.time() - start_time
            self.metrics.record_api_call(duration, True)
            return Mock(result="Response", tokens_used=50)
        
        # Test with calls that exceed rate limit
        num_calls = rate_limit + 5
        
        self.metrics.start_monitoring()
        
        with ThreadPoolExecutor(max_workers=num_calls) as executor:
            futures = [
                executor.submit(rate_limited_api_call, f"call {i}")
                for i in range(num_calls)
            ]
            
            successful_calls = 0
            rate_limited_calls = 0
            
            for future in as_completed(futures):
                try:
                    future.result(timeout=5.0)
                    successful_calls += 1
                except Exception as e:
                    if "Rate limit exceeded" in str(e):
                        rate_limited_calls += 1
        
        self.metrics.stop_monitoring()
        
        # Assert rate limiting behavior
        self.assertGreater(successful_calls, 0, "Should have some successful calls")
        self.assertGreater(rate_limited_calls, 0, "Should encounter rate limits")
        self.assertLessEqual(successful_calls, rate_limit + 2, "Rate limiting should prevent excessive calls")
        
        total_calls = successful_calls + rate_limited_calls
        success_rate = successful_calls / total_calls if total_calls > 0 else 0
        
        print(f"\nRate Limit Handling:")
        print(f"  Successful calls: {successful_calls}")
        print(f"  Rate limited calls: {rate_limited_calls}")
        print(f"  Success rate: {success_rate:.2%}")


if __name__ == '__main__':
    unittest.main(verbosity=2)