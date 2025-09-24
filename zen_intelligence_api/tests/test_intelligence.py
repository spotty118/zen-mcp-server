"""
Tests for Query Intelligence Analysis

This module tests the intelligent query analysis and routing capabilities.
"""

import pytest
from zen_intelligence_api.core.intelligence import (
    QueryIntelligence, 
    QueryDomain, 
    QueryComplexity, 
    QueryUrgency
)


class TestQueryIntelligence:
    """Test the QueryIntelligence class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.intelligence = QueryIntelligence()
    
    def test_detect_security_domain(self):
        """Test detection of security-related queries."""
        query = "How can I prevent SQL injection attacks in my application?"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_domain == QueryDomain.SECURITY
        assert routing_decision.primary_model in ["claude-sonnet", "claude-opus"]
    
    def test_detect_architecture_domain(self):
        """Test detection of architecture-related queries."""
        query = "What's the best microservices architecture pattern for a scalable system?"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_domain == QueryDomain.ARCHITECTURE
        assert routing_decision.primary_model in ["gpt-4", "gpt-5"]
    
    def test_detect_debugging_domain(self):
        """Test detection of debugging-related queries."""
        query = "I'm getting a NullPointerException in my Java code, help me debug it"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_domain == QueryDomain.DEBUGGING
        assert routing_decision.primary_model == "gemini-pro"
    
    def test_detect_simple_complexity(self):
        """Test detection of simple query complexity."""
        query = "What is a REST API?"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_complexity == QueryComplexity.SIMPLE
    
    def test_detect_expert_complexity(self):
        """Test detection of expert-level complexity."""
        query = "Design a distributed consensus algorithm for a blockchain network with Byzantine fault tolerance"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_complexity == QueryComplexity.EXPERT
    
    def test_detect_critical_urgency(self):
        """Test detection of critical urgency."""
        query = "URGENT: Production server is down with OutOfMemoryError, need immediate fix!"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_urgency == QueryUrgency.CRITICAL_FIX
    
    def test_detect_research_urgency(self):
        """Test detection of research urgency."""
        query = "I want to research different approaches to machine learning for my thesis"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_urgency == QueryUrgency.RESEARCH
    
    def test_model_selection_for_security(self):
        """Test optimal model selection for security queries."""
        query = "Analyze this code for security vulnerabilities"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        # Should prefer Claude for security analysis
        assert routing_decision.primary_model.startswith("claude")
        assert routing_decision.orchestration_strategy in ["consensus", "parallel"]
    
    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        query = "Simple question about Python syntax"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.estimated_cost > 0
        assert routing_decision.estimated_tokens > 0
        assert routing_decision.estimated_latency > 0
    
    def test_context_affects_complexity(self):
        """Test that context affects complexity detection."""
        query = "Fix this bug"
        context = [
            "I've been analyzing this distributed system for weeks",
            "The issue involves race conditions in concurrent processes",
            "Multiple microservices are affected"
        ]
        
        routing_decision = self.intelligence.analyze_query(query, context)
        
        # With complex context, should detect higher complexity
        assert routing_decision.detected_complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
    
    def test_routing_decision_completeness(self):
        """Test that routing decision contains all required fields."""
        query = "Help me optimize database queries"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        # Verify all required fields are present
        assert routing_decision.primary_model is not None
        assert routing_decision.orchestration_strategy is not None
        assert routing_decision.confidence_threshold > 0
        assert routing_decision.expected_quality_score > 0
        assert routing_decision.reasoning is not None
        assert routing_decision.detected_domain is not None
        assert routing_decision.detected_complexity is not None
        assert routing_decision.detected_urgency is not None
    
    def test_general_domain_fallback(self):
        """Test fallback to general domain for unclear queries."""
        query = "Hello, how are you today?"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        assert routing_decision.detected_domain == QueryDomain.GENERAL
    
    def test_supporting_models_for_complex_queries(self):
        """Test that complex queries get supporting models."""
        query = "Design and implement a comprehensive security audit system for enterprise applications"
        
        routing_decision = self.intelligence.analyze_query(query)
        
        # Complex queries should have supporting models
        assert len(routing_decision.supporting_models) > 0
        assert routing_decision.orchestration_strategy != "single"


if __name__ == "__main__":
    pytest.main([__file__])