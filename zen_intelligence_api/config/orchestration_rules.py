"""
Smart Orchestration Rules Configuration

This module defines intelligent orchestration rules that determine
how queries are routed and processed across multiple AI models.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from ..core.intelligence import QueryDomain, QueryComplexity, QueryUrgency


@dataclass
class OrchestrationRule:
    """Rule for orchestration behavior based on query characteristics."""
    
    # Conditions
    domains: List[QueryDomain]
    complexities: List[QueryComplexity]  
    urgencies: List[QueryUrgency]
    
    # Actions
    preferred_models: List[str]
    fallback_models: List[str]
    orchestration_strategy: str
    
    # Constraints
    max_models: int = 3
    consensus_threshold: float = 0.8
    quality_threshold: float = 0.85
    
    # Metadata
    rule_name: str = ""
    description: str = ""
    priority: int = 0  # Higher priority rules take precedence


class OrchestrationRules:
    """Smart orchestration rules engine."""
    
    def __init__(self):
        """Initialize with default orchestration rules."""
        self.rules: List[OrchestrationRule] = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default orchestration rules."""
        
        # Security Analysis Rule - High Priority
        self.rules.append(OrchestrationRule(
            rule_name="security_analysis",
            description="Security analysis requires Claude's expertise with GPT validation",
            domains=[QueryDomain.SECURITY],
            complexities=[QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            urgencies=[QueryUrgency.IMPLEMENTATION, QueryUrgency.CRITICAL_FIX],
            preferred_models=["claude-sonnet", "claude-opus"],
            fallback_models=["gpt-4", "gpt-5"],
            orchestration_strategy="consensus",
            max_models=3,
            consensus_threshold=0.85,
            quality_threshold=0.9,
            priority=100
        ))
        
        # Code Architecture Rule - High Priority  
        self.rules.append(OrchestrationRule(
            rule_name="architecture_design",
            description="Architecture decisions benefit from GPT's design expertise with Claude review",
            domains=[QueryDomain.ARCHITECTURE],
            complexities=[QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            urgencies=[QueryUrgency.RESEARCH, QueryUrgency.IMPLEMENTATION],
            preferred_models=["gpt-4", "gpt-5"],
            fallback_models=["claude-opus", "claude-sonnet"],
            orchestration_strategy="sequential",
            max_models=2,
            consensus_threshold=0.8,
            quality_threshold=0.9,
            priority=90
        ))
        
        # Debugging Rule - Medium Priority
        self.rules.append(OrchestrationRule(
            rule_name="debugging_assistance",
            description="Debugging benefits from Gemini's analysis with GPT optimization",
            domains=[QueryDomain.DEBUGGING],
            complexities=[QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
            urgencies=[QueryUrgency.IMPLEMENTATION, QueryUrgency.CRITICAL_FIX],
            preferred_models=["gemini-pro"],
            fallback_models=["gpt-4", "claude-sonnet"],
            orchestration_strategy="parallel",
            max_models=2,
            consensus_threshold=0.75,
            quality_threshold=0.8,
            priority=70
        ))
        
        # Performance Optimization Rule - Medium Priority
        self.rules.append(OrchestrationRule(
            rule_name="performance_optimization",
            description="Performance optimization requires GPT with Gemini validation",
            domains=[QueryDomain.PERFORMANCE],
            complexities=[QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            urgencies=[QueryUrgency.IMPLEMENTATION, QueryUrgency.CRITICAL_FIX],
            preferred_models=["gpt-4", "gpt-5"],
            fallback_models=["gemini-pro", "claude-sonnet"],
            orchestration_strategy="parallel",
            max_models=2,
            consensus_threshold=0.8,
            quality_threshold=0.85,
            priority=75
        ))
        
        # Code Review Rule - Medium Priority
        self.rules.append(OrchestrationRule(
            rule_name="comprehensive_code_review",
            description="Code reviews benefit from multiple perspectives for thorough analysis",
            domains=[QueryDomain.CODE_REVIEW],
            complexities=[QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            urgencies=[QueryUrgency.RESEARCH, QueryUrgency.IMPLEMENTATION],
            preferred_models=["claude-sonnet", "gpt-4"],
            fallback_models=["gemini-pro", "gpt-5"],
            orchestration_strategy="consensus",
            max_models=3,
            consensus_threshold=0.8,
            quality_threshold=0.85,
            priority=80
        ))
        
        # Fast Response Rule - High Priority for Urgent Simple Tasks
        self.rules.append(OrchestrationRule(
            rule_name="urgent_simple_tasks",
            description="Simple urgent tasks prioritize speed over consensus",
            domains=[QueryDomain.GENERAL, QueryDomain.DEBUGGING, QueryDomain.PERFORMANCE],
            complexities=[QueryComplexity.SIMPLE, QueryComplexity.MODERATE],
            urgencies=[QueryUrgency.CRITICAL_FIX],
            preferred_models=["gemini-flash", "gpt-4"],
            fallback_models=["gemini-pro"],
            orchestration_strategy="single",
            max_models=1,
            consensus_threshold=0,
            quality_threshold=0.7,
            priority=95
        ))
        
        # Research and Planning Rule - Medium Priority
        self.rules.append(OrchestrationRule(
            rule_name="research_and_planning",
            description="Research tasks benefit from Claude's analytical depth with GPT synthesis",
            domains=[QueryDomain.RESEARCH, QueryDomain.PLANNING],
            complexities=[QueryComplexity.COMPLEX, QueryComplexity.EXPERT],
            urgencies=[QueryUrgency.RESEARCH],
            preferred_models=["claude-opus", "gpt-5"],
            fallback_models=["claude-sonnet", "gpt-4"],
            orchestration_strategy="sequential",
            max_models=2,
            consensus_threshold=0.8,
            quality_threshold=0.9,
            priority=65
        ))
        
        # Testing Strategy Rule - Low Priority
        self.rules.append(OrchestrationRule(
            rule_name="testing_strategy",
            description="Testing strategies benefit from Gemini's systematic approach",
            domains=[QueryDomain.TESTING],
            complexities=[QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
            urgencies=[QueryUrgency.IMPLEMENTATION],
            preferred_models=["gemini-pro"],
            fallback_models=["gpt-4", "claude-sonnet"],
            orchestration_strategy="single",
            max_models=1,
            consensus_threshold=0,
            quality_threshold=0.8,
            priority=50
        ))
        
        # Documentation Rule - Low Priority
        self.rules.append(OrchestrationRule(
            rule_name="documentation_writing",
            description="Documentation benefits from Claude's writing clarity",
            domains=[QueryDomain.DOCUMENTATION],
            complexities=[QueryComplexity.SIMPLE, QueryComplexity.MODERATE, QueryComplexity.COMPLEX],
            urgencies=[QueryUrgency.IMPLEMENTATION],
            preferred_models=["claude-sonnet", "claude-opus"],
            fallback_models=["gpt-4"],
            orchestration_strategy="single",
            max_models=1,
            consensus_threshold=0,
            quality_threshold=0.8,
            priority=40
        ))
        
        # General Fallback Rule - Lowest Priority
        self.rules.append(OrchestrationRule(
            rule_name="general_fallback",
            description="Default rule for queries that don't match specific patterns",
            domains=list(QueryDomain),  # All domains
            complexities=list(QueryComplexity),  # All complexities
            urgencies=list(QueryUrgency),  # All urgencies
            preferred_models=["gpt-4", "claude-sonnet"],
            fallback_models=["gemini-pro", "gpt-5"],
            orchestration_strategy="single",
            max_models=1,
            consensus_threshold=0,
            quality_threshold=0.75,
            priority=1
        ))
        
        # Sort rules by priority (highest first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def find_matching_rule(
        self, 
        domain: QueryDomain, 
        complexity: QueryComplexity, 
        urgency: QueryUrgency
    ) -> Optional[OrchestrationRule]:
        """
        Find the best matching rule for given query characteristics.
        
        Args:
            domain: Query domain
            complexity: Query complexity  
            urgency: Query urgency
            
        Returns:
            Best matching orchestration rule or None
        """
        
        for rule in self.rules:
            # Check if query matches rule conditions
            if (domain in rule.domains and 
                complexity in rule.complexities and 
                urgency in rule.urgencies):
                return rule
        
        return None
    
    def get_orchestration_config(
        self, 
        domain: QueryDomain, 
        complexity: QueryComplexity, 
        urgency: QueryUrgency,
        available_models: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Get orchestration configuration based on query characteristics.
        
        Args:
            domain: Query domain
            complexity: Query complexity
            urgency: Query urgency  
            available_models: List of available models (for filtering)
            
        Returns:
            Orchestration configuration dictionary
        """
        
        # Find matching rule
        rule = self.find_matching_rule(domain, complexity, urgency)
        if not rule:
            # This shouldn't happen due to general fallback rule
            rule = self.rules[-1]  # Use last rule as ultimate fallback
        
        # Filter models by availability if specified
        preferred_models = rule.preferred_models
        fallback_models = rule.fallback_models
        
        if available_models:
            preferred_models = [m for m in preferred_models if m in available_models]
            fallback_models = [m for m in fallback_models if m in available_models]
        
        # Combine preferred and fallback models (remove duplicates)
        all_models = preferred_models + [m for m in fallback_models if m not in preferred_models]
        
        # Limit to max_models
        selected_models = all_models[:rule.max_models]
        
        return {
            "rule_name": rule.rule_name,
            "description": rule.description,
            "primary_model": selected_models[0] if selected_models else "gpt-4",
            "supporting_models": selected_models[1:] if len(selected_models) > 1 else [],
            "orchestration_strategy": rule.orchestration_strategy,
            "consensus_threshold": rule.consensus_threshold,
            "quality_threshold": rule.quality_threshold,
            "max_models": rule.max_models,
            "reasoning": f"Applied rule '{rule.rule_name}' for {domain.value} domain with {complexity.value} complexity"
        }
    
    def add_custom_rule(self, rule: OrchestrationRule):
        """Add a custom orchestration rule."""
        self.rules.append(rule)
        # Re-sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.rule_name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def get_all_rules(self) -> List[OrchestrationRule]:
        """Get all orchestration rules."""
        return self.rules.copy()
    
    def get_rules_for_domain(self, domain: QueryDomain) -> List[OrchestrationRule]:
        """Get all rules applicable to a specific domain."""
        return [rule for rule in self.rules if domain in rule.domains]