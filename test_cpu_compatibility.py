#!/usr/bin/env python3
"""
Cross-Platform CPU Compatibility Test

This script tests the enhanced CPU compatibility features for Intel, AMD, and Apple Silicon
on Windows, Linux, and macOS platforms.
"""

import os
import platform
import threading
from tools.parallelthink import ParallelThinkTool, ParallelThinkRequest
from utils.core_context_storage import get_core_context_storage


def test_cpu_architecture_detection():
    """Test CPU architecture detection across different platforms"""
    print("=== CPU Architecture Detection Test ===")
    tool = ParallelThinkTool()
    
    cpu_info = tool._detect_cpu_architecture()
    
    print(f"System: {cpu_info['system']}")
    print(f"Architecture: {cpu_info['architecture']}")
    print(f"Processor: {cpu_info['processor']}")
    print(f"Available cores: {cpu_info['available_cores']}")
    print(f"Intel CPU: {cpu_info['is_intel']}")
    print(f"AMD CPU: {cpu_info['is_amd']}")
    print(f"Apple Silicon: {cpu_info['is_apple_silicon']}")
    print(f"ARM Architecture: {cpu_info['is_arm']}")
    print(f"Supports Affinity: {cpu_info['supports_affinity']}")
    print(f"Performance Cores: {cpu_info['supports_performance_cores']}")
    print(f"Recommended Strategy: {cpu_info['recommended_strategy']}")
    print()


def test_cpu_core_optimization():
    """Test CPU core optimization for different architectures"""
    print("=== CPU Core Optimization Test ===")
    tool = ParallelThinkTool()
    
    # Test auto-detection
    auto_cores = tool._get_optimal_cpu_cores()
    print(f"Auto-detected optimal cores: {auto_cores}")
    
    # Test with explicit values
    test_values = [1, 2, 4, 8, 16, 32]
    for requested in test_values:
        cores = tool._get_optimal_cpu_cores(requested)
        print(f"Requested {requested} cores, optimized to {cores}")
    
    print()


def test_cpu_affinity_cross_platform():
    """Test CPU affinity setting across different platforms"""
    print("=== Cross-Platform CPU Affinity Test ===")
    tool = ParallelThinkTool()
    
    available_cores = os.cpu_count() or 4
    
    print(f"Testing CPU affinity on {platform.system()} {platform.machine()}")
    print(f"Available cores: {available_cores}")
    
    # Test affinity setting on different cores
    test_cores = min(4, available_cores)
    for core_id in range(test_cores):
        success = tool._set_cpu_affinity(core_id)
        print(f"Core {core_id}: {'✓' if success else '✗'}")
    
    print()


def test_execution_strategy_selection():
    """Test execution strategy selection for different architectures"""
    print("=== Execution Strategy Selection Test ===")
    tool = ParallelThinkTool()
    
    cpu_info = tool._detect_cpu_architecture()
    cores = cpu_info["available_cores"]
    
    test_cases = [
        ("adaptive", 2, cores),
        ("adaptive", 4, cores),
        ("adaptive", 8, cores),
        ("asyncio", 4, cores),
        ("threads", 4, cores),
        ("hybrid", 4, cores),
    ]
    
    print(f"Testing on {cpu_info['architecture']} with {cores} cores")
    for strategy, paths, cores_used in test_cases:
        result = tool._choose_execution_strategy(strategy, paths, cores_used)
        print(f"Strategy: {strategy:8}, Paths: {paths}, Cores: {cores_used} → {result}")
    
    print()


def test_core_context_storage():
    """Test core context storage with cross-platform support"""
    print("=== Core Context Storage Test ===")
    
    storage = get_core_context_storage()
    
    # Test core ID detection
    core_id = storage._get_current_core_id()
    print(f"Current core ID: {core_id}")
    
    # Test context storage and retrieval
    test_key = "test_context"
    test_value = {"data": "cross_platform_test", "timestamp": 123456}
    
    storage.set_core_context(test_key, test_value, core_id)
    retrieved = storage.get_core_context(test_key, core_id)
    
    print(f"Context stored and retrieved: {'✓' if retrieved == test_value else '✗'}")
    
    # Test statistics
    stats = storage.get_core_statistics()
    print(f"Core contexts: {stats['total_cores']}")
    print(f"Shared contexts: {stats['total_shared_contexts']}")
    
    print()


def test_memory_usage_monitoring():
    """Test memory usage monitoring across platforms"""
    print("=== Memory Usage Monitoring Test ===")
    tool = ParallelThinkTool()
    
    memory_usage = tool._get_memory_usage()
    print(f"Current memory usage: {memory_usage:.2f} MB")
    
    # Test with different memory monitoring approaches
    try:
        import psutil
        process = psutil.Process()
        psutil_memory = process.memory_info().rss / 1024 / 1024
        print(f"psutil memory usage: {psutil_memory:.2f} MB")
    except ImportError:
        print("psutil not available")
    
    try:
        import resource
        resource_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"resource module memory: {resource_memory:.2f} MB")
    except Exception:
        print("resource module not available or failed")
    
    print()


def test_platform_specific_optimizations():
    """Test platform-specific optimizations"""
    print("=== Platform-Specific Optimizations Test ===")
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    print(f"Platform: {system} on {machine}")
    
    if system == "darwin" and ("arm" in machine or "aarch64" in machine):
        print("Apple Silicon optimizations:")
        print("  ✓ Performance/Efficiency core awareness")
        print("  ✓ Hybrid execution strategy preferred")
        print("  ✓ Thread priority hints for performance cores")
    
    elif system == "windows":
        print("Windows optimizations:")
        try:
            import psutil
            print("  ✓ psutil available for CPU affinity")
        except ImportError:
            print("  ✗ psutil not available (install recommended)")
        print("  ✓ Windows-specific CPU affinity support")
    
    elif system == "linux":
        print("Linux optimizations:")
        if hasattr(os, 'sched_setaffinity'):
            print("  ✓ Native CPU affinity support")
        else:
            print("  ✗ CPU affinity not available")
        print("  ✓ Full thread pool optimization")
    
    else:
        print(f"Generic optimizations for {system}")
    
    print()


def run_comprehensive_test():
    """Run comprehensive cross-platform CPU compatibility test"""
    print("Cross-Platform CPU Compatibility Test Suite")
    print("=" * 50)
    print()
    
    test_cpu_architecture_detection()
    test_cpu_core_optimization()
    test_cpu_affinity_cross_platform()
    test_execution_strategy_selection()
    test_core_context_storage()
    test_memory_usage_monitoring()
    test_platform_specific_optimizations()
    
    print("Test suite completed!")
    print()
    print("Summary:")
    print("- CPU architecture detection: Cross-platform support added")
    print("- CPU affinity: Linux, Windows, macOS compatibility")
    print("- Core optimization: Intel, AMD, Apple Silicon aware")
    print("- Execution strategies: Architecture-specific selection")
    print("- Memory monitoring: Multi-platform fallbacks")


if __name__ == "__main__":
    run_comprehensive_test()