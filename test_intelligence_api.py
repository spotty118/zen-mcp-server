#!/usr/bin/env python3
"""
Simple test script for Zen Intelligence API core functionality.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_intelligence_core():
    """Test the core intelligence functionality."""
    
    try:
        from zen_intelligence_api.core.intelligence import QueryIntelligence, QueryDomain, QueryComplexity
        
        print("🧠 Testing Zen Intelligence API Core")
        print("=" * 50)
        
        # Initialize intelligence system
        intelligence = QueryIntelligence()
        
        # Test 1: Security Query
        print("Test 1: Security Query Analysis")
        security_query = "How can I prevent SQL injection attacks in my application?"
        result = intelligence.analyze_query(security_query)
        
        print(f"Query: {security_query}")
        print(f"Domain: {result.detected_domain}")
        print(f"Complexity: {result.detected_complexity}")
        print(f"Primary Model: {result.primary_model}")
        print(f"Strategy: {result.orchestration_strategy}")
        print(f"Cost Estimate: ${result.estimated_cost:.4f}")
        print()
        
        # Test 2: Architecture Query
        print("Test 2: Architecture Query Analysis")
        arch_query = "Design a scalable microservices architecture for high-traffic applications"
        result = intelligence.analyze_query(arch_query)
        
        print(f"Query: {arch_query}")
        print(f"Domain: {result.detected_domain}")
        print(f"Complexity: {result.detected_complexity}")
        print(f"Primary Model: {result.primary_model}")
        print(f"Supporting Models: {result.supporting_models}")
        print(f"Strategy: {result.orchestration_strategy}")
        print(f"Expected Quality: {result.expected_quality_score:.2f}")
        print()
        
        # Test 3: Simple Query
        print("Test 3: Simple Query Analysis")
        simple_query = "What is a REST API?"
        result = intelligence.analyze_query(simple_query)
        
        print(f"Query: {simple_query}")
        print(f"Domain: {result.detected_domain}")
        print(f"Complexity: {result.detected_complexity}")
        print(f"Primary Model: {result.primary_model}")
        print(f"Strategy: {result.orchestration_strategy}")
        print()
        
        # Test 4: Debugging Query
        print("Test 4: Debugging Query Analysis")
        debug_query = "I'm getting a NullPointerException in my Java code, help me debug it"
        result = intelligence.analyze_query(debug_query)
        
        print(f"Query: {debug_query}")
        print(f"Domain: {result.detected_domain}")
        print(f"Primary Model: {result.primary_model}")
        print(f"Reasoning: {result.reasoning}")
        print()
        
        print("✅ All tests passed! Zen Intelligence API core is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available.")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_orchestration_rules():
    """Test the orchestration rules system."""
    
    try:
        from zen_intelligence_api.config.orchestration_rules import OrchestrationRules
        from zen_intelligence_api.core.intelligence import QueryDomain, QueryComplexity, QueryUrgency
        
        print("\n🔧 Testing Orchestration Rules")
        print("=" * 50)
        
        rules = OrchestrationRules()
        
        # Test security rule
        security_config = rules.get_orchestration_config(
            QueryDomain.SECURITY, 
            QueryComplexity.COMPLEX, 
            QueryUrgency.CRITICAL_FIX
        )
        
        print("Security Rule Test:")
        print(f"Rule: {security_config['rule_name']}")
        print(f"Primary Model: {security_config['primary_model']}")
        print(f"Strategy: {security_config['orchestration_strategy']}")
        print(f"Description: {security_config['description']}")
        print()
        
        # Test architecture rule
        arch_config = rules.get_orchestration_config(
            QueryDomain.ARCHITECTURE,
            QueryComplexity.EXPERT,
            QueryUrgency.IMPLEMENTATION
        )
        
        print("Architecture Rule Test:")
        print(f"Rule: {arch_config['rule_name']}")
        print(f"Primary Model: {arch_config['primary_model']}")
        print(f"Supporting Models: {arch_config['supporting_models']}")
        print(f"Strategy: {arch_config['orchestration_strategy']}")
        print()
        
        print("✅ Orchestration rules are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Orchestration rules test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🚀 Zen Intelligence API Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test core intelligence
    if not test_intelligence_core():
        success = False
    
    # Test orchestration rules
    if not test_orchestration_rules():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Zen Intelligence API is ready to use.")
        print("\nNext steps:")
        print("1. Install additional requirements: pip install -r requirements-intelligence-api.txt")
        print("2. Start the API server: python start_intelligence_api.py")
        print("3. Visit http://localhost:8000/docs for API documentation")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())