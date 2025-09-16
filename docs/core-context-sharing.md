# Core-Specific Context Sharing Enhancement

## Overview

This enhancement introduces per-CPU-core context isolation with intelligent inter-core sharing capabilities to the parallel thinking system. Each CPU core maintains its own context storage space while allowing selective sharing of relevant information between cores for enhanced collaborative processing.

## Problem Statement

The original question was: "Is it possible that each core carries its own context and shares between other cores/paths? We can add light GPU support but that may be an overkill."

This enhancement addresses exactly that need by implementing:
- **Per-core context isolation** for optimal cache locality and performance
- **Intelligent inter-core sharing** for collaborative insights
- **Light GPU detection** (without overkill) for future optimization potential

## Key Features

### 1. Per-Core Context Isolation

Each CPU core gets its own isolated context storage:

```python
# Each core maintains separate context
storage.set_core_context("analysis_type", "performance", core_id=0)
storage.set_core_context("analysis_type", "security", core_id=1)

# Cores access their own context independently
core_0_analysis = storage.get_core_context("analysis_type", core_id=0)  # "performance"
core_1_analysis = storage.get_core_context("analysis_type", core_id=1)  # "security"
```

### 2. Intelligent Inter-Core Sharing

Cores can share relevant insights with other cores:

```python
# Share insights between cores
storage.set_core_context(
    "key_insight", 
    "Memory optimization critical", 
    core_id=0, 
    share_with_others=True
)

# Other cores can access shared insights
shared_insight = storage.get_core_context("key_insight", core_id=1, check_shared=True)
```

### 3. Explicit Core-to-Core Sharing

Direct sharing between specific cores:

```python
# Explicit sharing from core 0 to cores 1 and 2
storage.share_context_between_cores("best_practice", source_core=0, target_cores={1, 2})
```

### 4. Enhanced Parallel Thinking Integration

The parallel thinking tool automatically utilizes core contexts:

```python
request = ParallelThinkRequest(
    prompt="Optimize database performance",
    thinking_paths=4,
    enable_core_context=True,              # NEW: Enable per-core context
    share_insights_between_cores=True,     # NEW: Enable insight sharing
    context_sharing_threshold=0.7,         # NEW: Sharing confidence threshold
    cpu_cores=4,
    execution_strategy="adaptive"
)
```

### 5. Light GPU Detection (Avoiding Overkill)

Detects GPU availability without using it by default:

```python
gpu_info = tool._detect_gpu_availability()
# Returns: {"available": True, "type": "nvidia", "memory": "8GB", "compute_capability": None}
# Note: Detection only - actual GPU usage avoided to prevent overkill
```

## Architecture

### Core Context Storage (`utils/core_context_storage.py`)

- **Thread-safe operations** with minimal lock contention
- **Singleton pattern** for consistent state
- **Automatic cleanup** of expired contexts
- **Memory tracking** per core
- **Statistics collection** for monitoring

### Enhanced Parallel Thinking (`tools/parallelthink.py`)

- **Automatic insight extraction** from thinking results
- **Context sharing** based on approach similarity
- **Enhanced synthesis** with core context statistics
- **Backward compatibility** with existing functionality

## Usage Examples

### Basic Core Context Operations

```python
from utils.core_context_storage import get_core_context_storage

storage = get_core_context_storage()

# Set context for specific core
storage.set_core_context("strategy", "optimize_queries", core_id=0)

# Share insights between cores
storage.set_core_context("insight", "Use connection pooling", core_id=0, share_with_others=True)

# Retrieve context (checks local first, then shared)
value = storage.get_core_context("insight", core_id=1, check_shared=True)
```

### Enhanced Parallel Thinking

```python
from tools.parallelthink import ParallelThinkTool, ParallelThinkRequest

tool = ParallelThinkTool()

# Create request with core context features
request = ParallelThinkRequest(
    prompt="How to scale our microservices architecture?",
    thinking_paths=4,
    approach_diversity=True,
    enable_core_context=True,           # Enable core-specific context
    share_insights_between_cores=True,  # Enable insight sharing
    context_sharing_threshold=0.8,      # High confidence threshold
    execution_strategy="adaptive"
)

# Execute with core context support
results = await tool.execute(request.model_dump())
```

### Core Statistics and Monitoring

```python
# Get detailed core usage statistics
stats = storage.get_core_statistics()

print(f"Active cores: {stats['total_cores']}")
print(f"Shared contexts: {stats['total_shared_contexts']}")

for core_id, core_info in stats['cores'].items():
    print(f"Core {core_id}: {core_info['context_count']} contexts, "
          f"{core_info['shared_keys']} shared, "
          f"{core_info['memory_usage']:.1f}MB")
```

## Performance Benefits

### Memory Efficiency

- **Per-core isolation** reduces memory contention
- **Selective sharing** minimizes unnecessary data replication
- **Automatic cleanup** prevents memory leaks
- **Memory tracking** enables optimization decisions

### CPU Cache Optimization

- **Core affinity** ensures consistent core assignment
- **Local context access** improves cache hit rates
- **Reduced context switching** overhead
- **Optimal batch sizing** based on core count

### Collaborative Intelligence

- **Insight sharing** between different analytical approaches
- **Cross-core learning** from successful strategies
- **Adaptive sharing** based on confidence thresholds
- **Enhanced synthesis** with collaborative information

## Configuration Options

### Core Context Parameters

```python
# In ParallelThinkRequest
enable_core_context: bool = True              # Enable per-core context isolation
share_insights_between_cores: bool = True     # Allow cross-core insight sharing  
context_sharing_threshold: float = 0.7        # Confidence threshold for sharing (0.0-1.0)
```

### System-Specific Tuning

```python
# High-core systems (8+ cores)
ParallelThinkRequest(
    cpu_cores=8,
    execution_strategy="threads", 
    enable_cpu_affinity=True,
    enable_core_context=True,
    share_insights_between_cores=True
)

# Memory-constrained systems  
ParallelThinkRequest(
    thinking_paths=4,
    batch_size=1,
    execution_strategy="hybrid",
    enable_core_context=True,
    share_insights_between_cores=False  # Reduce overhead
)
```

## Implementation Details

### Thread Safety

- **Read-Write locks** for core contexts
- **Separate locks** for shared contexts
- **Atomic operations** for statistics
- **Deadlock prevention** with lock ordering

### Memory Management

- **TTL-based expiration** for contexts
- **Background cleanup** thread
- **Memory usage tracking** per core
- **Configurable cleanup intervals**

### GPU Detection

- **NVIDIA GPU** detection via nvidia-smi
- **Integrated GPU** detection via /sys/class/drm
- **Graceful fallback** when detection fails
- **Information only** - no actual GPU usage to avoid overkill

## Backward Compatibility

All enhancements are fully backward compatible:

- **Existing APIs** work unchanged
- **Default parameters** preserve original behavior
- **Graceful degradation** when core features are unavailable
- **Optional feature activation** via request parameters

## Testing

Comprehensive test suite covers:

- **Core isolation** and sharing functionality
- **Thread safety** under concurrent access
- **Memory management** and cleanup
- **Integration** with parallel thinking tool
- **Performance** benchmarks and statistics

Run tests with:
```bash
python -m pytest tests/test_core_context.py -v
```

## Demo

Experience the enhancement with the demo script:
```bash
python core_context_demo.py
```

## Future Enhancements

The core context framework enables future optimizations:

- **Machine learning** insight classification
- **Advanced sharing** algorithms based on approach similarity
- **GPU acceleration** when beneficial (avoiding current overkill)
- **Distributed computing** support for multi-machine parallelism
- **Performance profiling** integration

## Conclusion

This enhancement successfully addresses the original question by providing per-core context isolation with intelligent sharing capabilities. The implementation avoids GPU overkill while providing a solid foundation for future GPU integration when appropriate. The result is improved performance, better memory efficiency, and enhanced collaborative intelligence between parallel thinking paths.