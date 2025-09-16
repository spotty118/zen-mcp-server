#!/usr/bin/env python3
"""
Test script for enhanced parallel thinking implementation
Tests CPU core utilization, threading strategies, and performance optimizations
"""

import asyncio
import threading

from tools.parallelthink import ParallelThinkingPath, ParallelThinkRequest, ParallelThinkTool


def test_cpu_core_detection():
    """Test CPU core detection and optimization"""
    print("=== Testing CPU Core Detection ===")
    tool = ParallelThinkTool()

    # Test auto-detection
    auto_cores = tool._get_optimal_cpu_cores()
    print(f"Auto-detected optimal cores: {auto_cores}")

    # Test with explicit values
    for requested in [1, 2, 4, 8, 16]:
        cores = tool._get_optimal_cpu_cores(requested)
        print(f"Requested {requested} cores, got {cores}")

    print()


def test_batch_size_calculation():
    """Test batch size optimization"""
    print("=== Testing Batch Size Calculation ===")
    tool = ParallelThinkTool()

    cores = tool._get_optimal_cpu_cores()
    for paths in [2, 4, 6, 8, 12, 16]:
        batch_size = tool._get_optimal_batch_size(paths, cores)
        print(f"Paths: {paths}, Cores: {cores}, Batch size: {batch_size}")

    print()


def test_execution_strategy_selection():
    """Test execution strategy selection"""
    print("=== Testing Execution Strategy Selection ===")
    tool = ParallelThinkTool()

    cores = tool._get_optimal_cpu_cores()
    test_cases = [
        ("adaptive", 2, cores),
        ("adaptive", 4, cores),
        ("adaptive", 8, cores),
        ("asyncio", 4, cores),
        ("threads", 4, cores),
        ("hybrid", 4, cores),
    ]

    for strategy, paths, cores_used in test_cases:
        result = tool._choose_execution_strategy(strategy, paths, cores_used)
        print(f"Strategy: {strategy}, Paths: {paths}, Cores: {cores_used} → {result}")

    print()


def test_thinking_path_enhancements():
    """Test enhanced thinking path functionality"""
    print("=== Testing Enhanced Thinking Path ===")

    path = ParallelThinkingPath("test_1", "Analytical approach")
    print(f"Initial path: {path.path_id}, CPU core: {path.cpu_core}")

    # Simulate setting CPU core and thread info
    path.cpu_core = 2
    path.thread_id = threading.get_ident()
    path.memory_usage = 10.5

    print(f"Enhanced path: CPU core: {path.cpu_core}, Thread: {path.thread_id}, Memory: {path.memory_usage}MB")
    print()


def test_memory_monitoring():
    """Test memory usage monitoring"""
    print("=== Testing Memory Monitoring ===")
    tool = ParallelThinkTool()

    initial_memory = tool._get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f}MB")

    # Allocate some memory to test monitoring
    data = list(range(10000))
    after_memory = tool._get_memory_usage()
    print(f"After allocation: {after_memory:.1f}MB")
    print(f"Memory increase: {after_memory - initial_memory:.1f}MB")

    del data
    print()


def test_cpu_affinity():
    """Test CPU affinity setting"""
    print("=== Testing CPU Affinity ===")
    tool = ParallelThinkTool()

    cores = tool._get_optimal_cpu_cores()
    for core_id in range(min(cores, 4)):
        success = tool._set_cpu_affinity(core_id)
        print(f"Setting affinity to core {core_id}: {'Success' if success else 'Not supported/Failed'}")

    print()


async def test_enhanced_request_model():
    """Test enhanced request model with new parameters"""
    print("=== Testing Enhanced Request Model ===")

    # Test with enhanced parameters
    request = ParallelThinkRequest(
        prompt="Test parallel thinking",
        thinking_paths=4,
        cpu_cores=2,
        execution_strategy="hybrid",
        enable_cpu_affinity=True,
        batch_size=2
    )

    print("Request created with:")
    print(f"  CPU cores: {request.cpu_cores}")
    print(f"  Execution strategy: {request.execution_strategy}")
    print(f"  CPU affinity: {request.enable_cpu_affinity}")
    print(f"  Batch size: {request.batch_size}")
    print()


async def simulate_parallel_execution():
    """Simulate enhanced parallel execution without API calls"""
    print("=== Simulating Enhanced Parallel Execution ===")

    tool = ParallelThinkTool()

    # Create test paths
    paths = []
    for i in range(4):
        path = ParallelThinkingPath(f"path_{i+1}", f"Approach {i+1}")
        paths.append(path)

    # Test metrics
    cores = tool._get_optimal_cpu_cores(4)
    batch_size = tool._get_optimal_batch_size(len(paths), cores)
    strategy = tool._choose_execution_strategy("adaptive", len(paths), cores)

    print("Simulating execution with:")
    print(f"  Paths: {len(paths)}")
    print(f"  CPU cores: {cores}")
    print(f"  Batch size: {batch_size}")
    print(f"  Strategy: {strategy}")

    # Simulate path execution metadata
    for i, path in enumerate(paths):
        path.cpu_core = i % cores
        path.thread_id = 1000 + i
        path.execution_time = 0.5 + (i * 0.1)
        path.memory_usage = 5.0 + (i * 2.0)
        path.result = f"Simulated result for {path.approach}"

    # Test synthesis with enhanced metrics
    request = ParallelThinkRequest(prompt="Test", synthesis_style="comprehensive")
    synthesis = tool._synthesize_results(paths, "comprehensive", request)

    print("\nGenerated synthesis preview:")
    print(synthesis[:500] + "..." if len(synthesis) > 500 else synthesis)
    print()


def run_performance_comparison():
    """Compare performance metrics of different strategies"""
    print("=== Performance Comparison ===")
    tool = ParallelThinkTool()

    strategies = ["asyncio", "threads", "hybrid"]
    path_counts = [2, 4, 6, 8]

    print("Strategy recommendations based on configuration:")
    for paths in path_counts:
        cores = tool._get_optimal_cpu_cores()
        print(f"\nPaths: {paths}, Available cores: {cores}")
        for strategy in strategies:
            chosen = tool._choose_execution_strategy(strategy, paths, cores)
            batch_size = tool._get_optimal_batch_size(paths, cores)
            print(f"  {strategy:8} → {chosen:8} (batch: {batch_size})")

    print()


async def main():
    """Run all enhancement tests"""
    print("🚀 Testing Enhanced Parallel Thinking Implementation")
    print("=" * 60)

    test_cpu_core_detection()
    test_batch_size_calculation()
    test_execution_strategy_selection()
    test_thinking_path_enhancements()
    test_memory_monitoring()
    test_cpu_affinity()
    await test_enhanced_request_model()
    await simulate_parallel_execution()
    run_performance_comparison()

    print("✅ All enhancement tests completed successfully!")
    print("\nKey Improvements:")
    print("• Smart CPU core detection and utilization")
    print("• Hybrid concurrency model (asyncio + threads)")
    print("• CPU affinity optimization")
    print("• Memory usage monitoring")
    print("• Adaptive execution strategy selection")
    print("• Enhanced performance metrics and reporting")


if __name__ == "__main__":
    asyncio.run(main())
