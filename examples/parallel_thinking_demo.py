#!/usr/bin/env python3
"""
Demonstration script for the Parallel Thinking Tool

This script shows examples of how to use the parallel thinking functionality
for different types of problems and scenarios.
"""

import asyncio
from unittest.mock import Mock, patch

from tools.parallelthink import ParallelThinkTool


async def demo_basic_parallel_thinking():
    """Demonstrate basic parallel thinking with approach diversity"""
    print("🔄 Demo 1: Basic Parallel Thinking")
    print("="*50)

    tool = ParallelThinkTool()

    # Mock the provider for demo purposes
    mock_registry = Mock()
    mock_provider = Mock()

    # Mock different responses for different approaches
    responses = [
        "ANALYTICAL APPROACH: Breaking down the problem systematically, I identify three key components...",
        "CREATIVE APPROACH: Let me think outside the box - what if we approach this completely differently...",
        "RISK-FOCUSED APPROACH: The main risks I see are potential scalability issues and security concerns..."
    ]

    mock_provider.generate_content.side_effect = [
        Mock(content=resp) for resp in responses
    ]
    mock_provider.get_default_model.return_value = "demo-model"
    mock_registry.get_default_provider.return_value = mock_provider

    with patch('tools.parallelthink.ModelProviderRegistry', return_value=mock_registry):
        result = await tool.execute({
            "prompt": "What's the best approach to building a new web application with high scalability requirements?",
            "thinking_paths": 3,
            "approach_diversity": True,
            "synthesis_style": "comprehensive"
        })

    print("📊 Results:")
    response = result[0]
    print(f"Total paths: {response['execution_summary']['total_paths']}")
    print(f"Successful paths: {response['execution_summary']['successful_paths']}")
    print(f"Approaches: {response['execution_summary']['approaches_used']}")
    print()


async def demo_hypothesis_testing():
    """Demonstrate hypothesis testing mode"""
    print("🔬 Demo 2: Hypothesis Testing")
    print("="*50)

    tool = ParallelThinkTool()

    # Mock the provider
    mock_registry = Mock()
    mock_provider = Mock()

    # Mock hypothesis-based responses
    hypothesis_responses = [
        "Testing efficiency hypothesis: The performance issues seem to stem from database queries...",
        "Testing complexity hypothesis: The problem appears to be architectural complexity rather than...",
        "Testing scalability hypothesis: The bottleneck is likely in the horizontal scaling approach..."
    ]

    mock_provider.generate_content.side_effect = [
        Mock(content=resp) for resp in hypothesis_responses
    ]
    mock_provider.get_default_model.return_value = "demo-model"
    mock_registry.get_default_provider.return_value = mock_provider

    with patch('tools.parallelthink.ModelProviderRegistry', return_value=mock_registry):
        result = await tool.execute({
            "prompt": "Why is our application experiencing performance degradation?",
            "thinking_paths": 3,
            "hypothesis_testing": True,
            "synthesis_style": "consensus"
        })

    print("📊 Results:")
    response = result[0]
    print(f"Total paths: {response['execution_summary']['total_paths']}")
    print(f"Synthesis style: {response['execution_summary']['synthesis_style']}")
    print()


async def demo_model_diversity():
    """Demonstrate multi-model thinking"""
    print("🤖 Demo 3: Model Diversity")
    print("="*50)

    tool = ParallelThinkTool()

    # Mock multiple models
    mock_registry = Mock()
    mock_provider = Mock()

    # Mock different model responses
    model_responses = [
        "GPT-4 perspective: From a deep learning optimization standpoint, the key consideration is...",
        "Gemini perspective: Looking at this from a multi-modal analysis angle, I notice patterns that..."
    ]

    mock_provider.generate_content.side_effect = [
        Mock(content=resp) for resp in model_responses
    ]
    mock_provider.get_default_model.return_value = "default-model"
    mock_registry.get_default_provider.return_value = mock_provider
    mock_registry.get_provider_for_model.return_value = mock_provider

    # Mock available models
    with patch.object(tool, '_get_available_models', return_value=["gpt-4", "gemini-pro"]):
        with patch('tools.parallelthink.ModelProviderRegistry', return_value=mock_registry):
            result = await tool.execute({
                "prompt": "How can we improve the machine learning pipeline efficiency?",
                "thinking_paths": 2,
                "model_diversity": True,
                "synthesis_style": "diverse"
            })

    print("📊 Results:")
    response = result[0]
    print(f"Models used: {response['execution_summary']['models_used']}")
    print(f"Individual paths included: {'individual_paths' in response}")
    print()


async def demo_focused_analysis():
    """Demonstrate focused analysis with specific areas"""
    print("🎯 Demo 4: Focused Analysis")
    print("="*50)

    tool = ParallelThinkTool()

    # Mock the provider
    mock_registry = Mock()
    mock_provider = Mock()

    # Mock focused responses
    focused_responses = [
        "SECURITY-FOCUSED ANALYSIS: The main security considerations include authentication, authorization...",
        "PERFORMANCE-FOCUSED ANALYSIS: From a performance optimization perspective, the critical factors...",
        "MAINTAINABILITY-FOCUSED ANALYSIS: Long-term maintainability requires attention to code structure..."
    ]

    mock_provider.generate_content.side_effect = [
        Mock(content=resp) for resp in focused_responses
    ]
    mock_provider.get_default_model.return_value = "demo-model"
    mock_registry.get_default_provider.return_value = mock_provider

    with patch('tools.parallelthink.ModelProviderRegistry', return_value=mock_registry):
        result = await tool.execute({
            "prompt": "Evaluate this API design for production readiness",
            "thinking_paths": 3,
            "focus_areas": ["security", "performance", "maintainability"],
            "synthesis_style": "comprehensive",
            "include_individual_paths": True
        })

    print("📊 Results:")
    response = result[0]
    print("Focus areas applied: security, performance, maintainability")
    print(f"Individual path results: {len(response.get('individual_paths', []))}")
    print()


def print_tool_info():
    """Print information about the parallel thinking tool"""
    print("🧠 Parallel Thinking Tool")
    print("="*50)

    tool = ParallelThinkTool()
    print(f"Tool name: {tool.get_name()}")
    print(f"Description: {tool.get_description()}")
    print(f"Model category: {tool.get_model_category()}")
    print(f"Default temperature: {tool.get_default_temperature()}")
    print(f"Default thinking mode: {tool.get_default_thinking_mode()}")
    print()


async def main():
    """Run all demonstrations"""
    print("🎯 Parallel Thinking Tool Demonstration")
    print("="*80)
    print()

    # Show tool information
    print_tool_info()

    # Run demos
    await demo_basic_parallel_thinking()
    await demo_hypothesis_testing()
    await demo_model_diversity()
    await demo_focused_analysis()

    print("✅ All demonstrations completed!")
    print()
    print("🔗 Key Benefits of Parallel Thinking:")
    print("   • Multiple perspectives on complex problems")
    print("   • Concurrent execution for faster results")
    print("   • Diverse approaches for comprehensive analysis")
    print("   • Intelligent synthesis of different viewpoints")
    print("   • Support for hypothesis testing and validation")
    print()
    print("📚 For more information, see docs/parallel_thinking.md")


if __name__ == "__main__":
    asyncio.run(main())
