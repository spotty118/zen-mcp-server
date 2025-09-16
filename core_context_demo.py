#!/usr/bin/env python3
"""
Core Context Sharing Demo
Demonstrates the enhanced parallel thinking with per-core context isolation and inter-core sharing
"""

import asyncio

from tools.parallelthink import ParallelThinkRequest, ParallelThinkTool
from utils.core_context_storage import get_core_context_storage


async def demo_core_context_basics():
    """Demo basic core context functionality"""
    print("=== Core Context Storage Demo ===")
    storage = get_core_context_storage()

    # Demonstrate per-core isolation
    print("\n1. Core Isolation:")
    storage.set_core_context("analysis_type", "performance", core_id=0)
    storage.set_core_context("analysis_type", "security", core_id=1)

    print(f"   Core 0 analysis: {storage.get_core_context('analysis_type', core_id=0)}")
    print(f"   Core 1 analysis: {storage.get_core_context('analysis_type', core_id=1)}")

    # Demonstrate context sharing
    print("\n2. Context Sharing:")
    storage.set_core_context("key_insight", "Memory optimization critical", core_id=0, share_with_others=True)

    # Access from different core
    shared_insight = storage.get_core_context("key_insight", core_id=1, check_shared=True)
    print(f"   Shared insight accessed from core 1: {shared_insight}")

    # Demonstrate explicit sharing
    print("\n3. Explicit Core-to-Core Sharing:")
    storage.set_core_context("best_practice", "Use connection pooling", core_id=0)
    success = storage.share_context_between_cores("best_practice", 0, {1, 2})
    print(f"   Sharing success: {success}")

    # Show statistics
    print("\n4. Core Statistics:")
    stats = storage.get_core_statistics()
    print(f"   Total cores active: {stats['total_cores']}")
    print(f"   Total shared contexts: {stats['total_shared_contexts']}")
    for core_id, core_stats in stats["cores"].items():
        print(f"   Core {core_id}: {core_stats['context_count']} contexts, {core_stats['shared_keys']} shared")


async def demo_enhanced_parallel_thinking():
    """Demo enhanced parallel thinking with core context"""
    print("\n\n=== Enhanced Parallel Thinking Demo ===")

    tool = ParallelThinkTool()

    # Demo GPU detection
    print("\n1. System Capabilities:")
    gpu_info = tool._detect_gpu_availability()
    print(f"   GPU Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"   GPU Type: {gpu_info['type']}")
        print("   Note: GPU detected but using CPU-first approach to avoid overkill")

    # Demo core allocation
    print("\n2. CPU Core Optimization:")
    optimal_cores = tool._get_optimal_cpu_cores()
    print(f"   Optimal cores detected: {optimal_cores}")

    for paths in [2, 4, 6, 8]:
        batch_size = tool._get_optimal_batch_size(paths, optimal_cores)
        strategy = tool._choose_execution_strategy("adaptive", paths, optimal_cores)
        print(f"   {paths} paths → strategy: {strategy}, batch: {batch_size}")


async def demo_request_with_core_context():
    """Demo creating requests with core context parameters"""
    print("\n\n=== Core Context Request Demo ===")

    # Create request with core context enabled
    request = ParallelThinkRequest(
        prompt="How can we optimize database performance for a high-traffic application?",
        thinking_paths=4,
        approach_diversity=True,
        enable_core_context=True,
        share_insights_between_cores=True,
        context_sharing_threshold=0.8,
        cpu_cores=4,
        execution_strategy="adaptive",
        synthesis_style="comprehensive"
    )

    print("Request created with core context features:")
    print(f"   Core context enabled: {request.enable_core_context}")
    print(f"   Insights sharing: {request.share_insights_between_cores}")
    print(f"   Sharing threshold: {request.context_sharing_threshold}")
    print(f"   CPU cores: {request.cpu_cores}")
    print(f"   Execution strategy: {request.execution_strategy}")


async def demo_path_enhancements():
    """Demo enhanced thinking path tracking"""
    print("\n\n=== Enhanced Path Tracking Demo ===")

    from tools.parallelthink import ParallelThinkingPath

    # Create sample paths with enhanced tracking
    paths = []
    approaches = [
        "Performance-focused analysis",
        "Security-centered review",
        "Scalability assessment",
        "Cost optimization study"
    ]

    for i, approach in enumerate(approaches):
        path = ParallelThinkingPath(f"path_{i}", approach)
        path.cpu_core = i % 4
        path.core_context_used = True
        path.shared_context_keys = [f"insight_{approach.split()[0].lower()}"]
        path.execution_time = 1.2 + i * 0.3
        path.memory_usage = 12.5 + i * 2.1
        paths.append(path)

    print("Enhanced path tracking:")
    for path in paths:
        print(f"   {path.approach}:")
        print(f"     CPU Core: {path.cpu_core}")
        print(f"     Context Used: {path.core_context_used}")
        print(f"     Shared Keys: {path.shared_context_keys}")
        print(f"     Execution: {path.execution_time:.1f}s, Memory: {path.memory_usage:.1f}MB")


async def demo_inter_core_collaboration():
    """Demo inter-core collaboration scenario"""
    print("\n\n=== Inter-Core Collaboration Demo ===")

    storage = get_core_context_storage()

    # Simulate cores working on different aspects
    print("\n1. Cores analyzing different aspects:")

    # Core 0: Performance analysis
    storage.set_core_context("analysis_focus", "performance", core_id=0)
    storage.set_core_context("findings", ["Database queries are slow", "Memory usage high"], core_id=0, share_with_others=True)
    print("   Core 0: Performance analysis complete, findings shared")

    # Core 1: Security analysis
    storage.set_core_context("analysis_focus", "security", core_id=1)
    storage.set_core_context("findings", ["SQL injection risk", "Weak authentication"], core_id=1, share_with_others=True)
    print("   Core 1: Security analysis complete, findings shared")

    # Core 2: Scalability analysis accessing shared insights
    performance_findings = storage.get_core_context("findings", core_id=2, check_shared=True)
    storage.set_core_context("analysis_focus", "scalability", core_id=2)
    storage.set_core_context("enhanced_analysis", f"Building on insights: {performance_findings}", core_id=2)
    print("   Core 2: Scalability analysis enhanced with shared performance insights")

    # Show collaboration statistics
    print("\n2. Collaboration Statistics:")
    stats = storage.get_core_statistics()
    for core_id, core_info in stats["cores"].items():
        focus = storage.get_core_context("analysis_focus", core_id=core_id, check_shared=False)
        print(f"   Core {core_id} ({focus}): {core_info['context_count']} contexts, {core_info['shared_keys']} shared")


async def demo_memory_efficiency():
    """Demo memory efficiency with core-specific storage"""
    print("\n\n=== Memory Efficiency Demo ===")

    storage = get_core_context_storage()

    print("\n1. Memory usage by core:")

    # Simulate different memory usage patterns
    for core_id in range(4):
        # Add varying amounts of data per core
        data_size = (core_id + 1) * 100
        test_data = list(range(data_size))
        storage.set_core_context("large_dataset", test_data, core_id=core_id)

        # Update memory tracking (simplified)
        if core_id in storage._core_contexts:
            storage._core_contexts[core_id].memory_usage = len(test_data) * 0.01  # Rough estimate

    # Show memory distribution
    stats = storage.get_core_statistics()
    total_memory = sum(core_info.get('memory_usage', 0) for core_info in stats['cores'].values())

    print(f"   Total memory across cores: {total_memory:.2f}MB")
    for core_id, core_info in stats['cores'].items():
        memory = core_info.get('memory_usage', 0)
        percentage = (memory / total_memory * 100) if total_memory > 0 else 0
        print(f"   Core {core_id}: {memory:.2f}MB ({percentage:.1f}%)")


async def main():
    """Run the complete core context demo"""
    print("🧠 Core-Specific Context Sharing Enhancement Demo")
    print("="*60)
    print("Demonstrating per-core context isolation with intelligent sharing")
    print("Avoiding GPU overkill while maximizing CPU efficiency")

    await demo_core_context_basics()
    await demo_enhanced_parallel_thinking()
    await demo_request_with_core_context()
    await demo_path_enhancements()
    await demo_inter_core_collaboration()
    await demo_memory_efficiency()

    print("\n\n✅ Core Context Enhancement Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Per-core context isolation for optimal cache locality")
    print("• Intelligent inter-core context sharing")
    print("• Thread-safe operations with minimal lock contention")
    print("• Memory-efficient context synchronization")
    print("• GPU detection with CPU-first approach (avoiding overkill)")
    print("• Enhanced performance metrics and monitoring")
    print("• Backward compatibility with existing functionality")


if __name__ == "__main__":
    asyncio.run(main())
