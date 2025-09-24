"""
Query Intelligence Analysis for Advanced Model Routing

This module provides sophisticated query analysis to determine optimal model
routing strategies based on task complexity, domain expertise, and performance history.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class QueryComplexity(Enum):
    """Query complexity levels for model routing decisions."""
    
    SIMPLE = "simple"           # Basic questions, quick responses
    MODERATE = "moderate"       # Standard analysis, medium reasoning
    COMPLEX = "complex"         # Deep analysis, multi-step reasoning
    EXPERT = "expert"           # Requires specialized domain expertise


class QueryDomain(Enum):
    """Domain categories for specialized model routing."""
    
    ARCHITECTURE = "architecture"        # System design, architecture decisions
    DEBUGGING = "debugging"              # Bug analysis, troubleshooting
    SECURITY = "security"                # Security analysis, vulnerability assessment
    PERFORMANCE = "performance"          # Optimization, performance analysis
    CODE_REVIEW = "code_review"          # Code quality, best practices
    TESTING = "testing"                  # Test strategy, test generation
    DOCUMENTATION = "documentation"      # Technical writing, docs
    REFACTORING = "refactoring"          # Code structure improvements
    GENERAL = "general"                  # General purpose questions
    RESEARCH = "research"                # Exploratory analysis
    PLANNING = "planning"                # Project planning, strategy


class QueryUrgency(Enum):
    """Urgency levels affecting routing strategy."""
    
    RESEARCH = "research"                # Exploratory, no time pressure
    IMPLEMENTATION = "implementation"    # Standard development work
    CRITICAL_FIX = "critical_fix"        # Urgent bug fixes, outages


class ModelExpertise(BaseModel):
    """Model expertise ratings by domain."""
    
    architecture: float = Field(default=0.7, ge=0.0, le=1.0)
    debugging: float = Field(default=0.7, ge=0.0, le=1.0)
    security: float = Field(default=0.7, ge=0.0, le=1.0)
    performance: float = Field(default=0.7, ge=0.0, le=1.0)
    code_review: float = Field(default=0.7, ge=0.0, le=1.0)
    testing: float = Field(default=0.7, ge=0.0, le=1.0)
    documentation: float = Field(default=0.7, ge=0.0, le=1.0)
    refactoring: float = Field(default=0.7, ge=0.0, le=1.0)
    general: float = Field(default=0.8, ge=0.0, le=1.0)
    research: float = Field(default=0.8, ge=0.0, le=1.0)
    planning: float = Field(default=0.7, ge=0.0, le=1.0)


class RoutingDecision(BaseModel):
    """Routing decision with model assignments and strategy."""
    
    primary_model: str = Field(..., description="Primary model for the task")
    supporting_models: List[str] = Field(default_factory=list, description="Supporting models for validation/enhancement")
    orchestration_strategy: str = Field(..., description="Strategy: single, parallel, sequential, consensus")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    expected_quality_score: float = Field(default=0.85, ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation of routing decision")
    
    # Analysis results
    detected_complexity: QueryComplexity
    detected_domain: QueryDomain
    detected_urgency: QueryUrgency
    
    # Resource optimization
    estimated_tokens: int = Field(default=1000, gt=0)
    estimated_cost: float = Field(default=0.01, ge=0.0)
    estimated_latency: float = Field(default=5.0, ge=0.0, description="Estimated response time in seconds")


class QueryIntelligence:
    """Analyzes queries for optimal model routing and orchestration."""
    
    # Model expertise configurations based on known strengths
    MODEL_EXPERTISE = {
        "gpt-4": ModelExpertise(
            architecture=0.95, performance=0.9, code_review=0.9, general=0.9
        ),
        "gpt-5": ModelExpertise(
            architecture=0.95, performance=0.9, code_review=0.9, general=0.95, research=0.9
        ),
        "claude-sonnet": ModelExpertise(
            security=0.95, code_review=0.95, documentation=0.9, research=0.9
        ),
        "claude-opus": ModelExpertise(
            security=0.95, research=0.95, planning=0.9, general=0.9
        ),
        "gemini-pro": ModelExpertise(
            debugging=0.9, testing=0.85, refactoring=0.8, general=0.85
        ),
        "gemini-flash": ModelExpertise(
            general=0.8, debugging=0.8, performance=0.75
        ),
    }
    
    # Domain detection patterns
    DOMAIN_PATTERNS = {
        QueryDomain.ARCHITECTURE: [
            r'\b(architecture|design|system\s+design|microservices|scalability|patterns)\b',
            r'\b(distributed|infrastructure|deployment|scaling)\b',
        ],
        QueryDomain.DEBUGGING: [
            r'\b(debug|bug|error|exception|traceback|stack trace|crash)\b',
            r'\b(fix|troubleshoot|investigate|diagnose)\b',
        ],
        QueryDomain.SECURITY: [
            r'\b(security|vulnerability|exploit|injection|xss|csrf|auth)\b',
            r'\b(secure|encrypt|decrypt|hash|token|permission)\b',
        ],
        QueryDomain.PERFORMANCE: [
            r'\b(performance|optimize|slow|fast|memory|cpu|latency)\b',
            r'\b(benchmark|profile|efficiency|bottleneck|cache)\b',
        ],
        QueryDomain.CODE_REVIEW: [
            r'\b(review|code\s+review|quality|best\s+practices|refactor)\b',
            r'\b(clean\s+code|maintainable|readable|standards)\b',
        ],
        QueryDomain.TESTING: [
            r'\b(test|testing|unit\s+test|integration|coverage)\b',
            r'\b(mock|stub|assert|pytest|unittest)\b',
        ],
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        QueryComplexity.SIMPLE: [
            r'\b(what\s+is|how\s+to|explain|simple|basic)\b',
            r'^(yes|no|true|false)\b',
        ],
        QueryComplexity.EXPERT: [
            r'\b(complex|advanced|enterprise|scalable|distributed)\b',
            r'\b(architecture|framework|infrastructure|optimization)\b',
        ],
    }
    
    def analyze_query(self, query: str, context: Optional[List[str]] = None) -> RoutingDecision:
        """
        Analyze query for optimal model routing.
        
        Args:
            query: The input query/prompt
            context: Optional conversation context
            
        Returns:
            RoutingDecision with model assignments and strategy
        """
        # Normalize query for analysis
        normalized_query = query.lower()
        
        # Detect domain
        domain = self._detect_domain(normalized_query)
        
        # Detect complexity
        complexity = self._detect_complexity(normalized_query, context)
        
        # Detect urgency
        urgency = self._detect_urgency(normalized_query)
        
        # Select optimal models
        primary_model, supporting_models = self._select_models(domain, complexity, urgency)
        
        # Determine orchestration strategy
        strategy = self._determine_strategy(complexity, domain, len(supporting_models))
        
        # Estimate resource requirements
        tokens = self._estimate_tokens(query, context)
        cost = self._estimate_cost(tokens, primary_model, supporting_models)
        latency = self._estimate_latency(strategy, complexity)
        
        return RoutingDecision(
            primary_model=primary_model,
            supporting_models=supporting_models,
            orchestration_strategy=strategy,
            confidence_threshold=self._get_confidence_threshold(complexity),
            expected_quality_score=self._get_expected_quality(domain, complexity),
            reasoning=self._generate_reasoning(domain, complexity, urgency, strategy),
            detected_complexity=complexity,
            detected_domain=domain,
            detected_urgency=urgency,
            estimated_tokens=tokens,
            estimated_cost=cost,
            estimated_latency=latency
        )
    
    def _detect_domain(self, query: str) -> QueryDomain:
        """Detect query domain based on content patterns."""
        domain_scores = {}
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            domain_scores[domain] = score
        
        # Return domain with highest score, default to GENERAL
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return QueryDomain.GENERAL
    
    def _detect_complexity(self, query: str, context: Optional[List[str]] = None) -> QueryComplexity:
        """Detect query complexity level."""
        # Check for explicit complexity indicators
        for complexity, patterns in self.COMPLEXITY_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return complexity
        
        # Heuristic-based complexity detection
        word_count = len(query.split())
        context_length = len(context) if context else 0
        
        if word_count < 10 and context_length == 0:
            return QueryComplexity.SIMPLE
        elif word_count > 100 or context_length > 5:
            return QueryComplexity.COMPLEX
        elif "analyze" in query.lower() or "review" in query.lower():
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.MODERATE
    
    def _detect_urgency(self, query: str) -> QueryUrgency:
        """Detect urgency level from query."""
        urgent_patterns = [
            r'\b(urgent|critical|emergency|asap|immediately)\b',
            r'\b(production|outage|down|broken|fix)\b',
        ]
        
        research_patterns = [
            r'\b(research|explore|investigate|study|learn)\b',
            r'\b(what\s+are|how\s+does|understand|compare)\b',
        ]
        
        for pattern in urgent_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryUrgency.CRITICAL_FIX
        
        for pattern in research_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return QueryUrgency.RESEARCH
        
        return QueryUrgency.IMPLEMENTATION
    
    def _select_models(self, domain: QueryDomain, complexity: QueryComplexity, 
                      urgency: QueryUrgency) -> Tuple[str, List[str]]:
        """Select optimal primary and supporting models."""
        
        # Get available models (simplified - in real implementation, query from providers)
        available_models = list(self.MODEL_EXPERTISE.keys())
        
        # Score models for this domain
        model_scores = {}
        for model in available_models:
            expertise = self.MODEL_EXPERTISE[model]
            domain_score = getattr(expertise, domain.value, 0.7)
            
            # Adjust for complexity and urgency
            if complexity == QueryComplexity.EXPERT:
                domain_score *= 1.1  # Prefer higher expertise for expert queries
            elif complexity == QueryComplexity.SIMPLE and urgency == QueryUrgency.CRITICAL_FIX:
                # For simple urgent fixes, prefer fast models
                if "flash" in model.lower():
                    domain_score *= 1.2
            
            model_scores[model] = domain_score
        
        # Select primary model (highest score)
        primary_model = max(model_scores, key=model_scores.get)
        
        # Select supporting models for complex queries
        supporting_models = []
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            # Sort by score and take top 2-3 excluding primary
            sorted_models = sorted(
                [(m, s) for m, s in model_scores.items() if m != primary_model],
                key=lambda x: x[1], reverse=True
            )
            supporting_models = [m for m, _ in sorted_models[:2]]
        
        return primary_model, supporting_models
    
    def _determine_strategy(self, complexity: QueryComplexity, domain: QueryDomain, 
                          supporting_count: int) -> str:
        """Determine orchestration strategy."""
        if supporting_count == 0:
            return "single"
        
        if complexity == QueryComplexity.EXPERT or domain == QueryDomain.SECURITY:
            return "consensus"  # Need agreement from multiple models
        elif domain in [QueryDomain.CODE_REVIEW, QueryDomain.ARCHITECTURE]:
            return "sequential"  # Iterative refinement
        else:
            return "parallel"   # Fast parallel processing
    
    def _estimate_tokens(self, query: str, context: Optional[List[str]] = None) -> int:
        """Estimate token requirements."""
        # Rough estimation: 1 token ≈ 0.75 words
        query_tokens = len(query.split()) * 1.33
        context_tokens = 0
        if context:
            context_text = " ".join(context)
            context_tokens = len(context_text.split()) * 1.33
        
        # Add overhead for system prompts and formatting
        overhead = 500
        
        return int(query_tokens + context_tokens + overhead)
    
    def _estimate_cost(self, tokens: int, primary: str, supporting: List[str]) -> float:
        """Estimate API cost."""
        # Simplified cost model (per 1k tokens)
        cost_per_1k = {
            "gpt-4": 0.03,
            "gpt-5": 0.05,
            "claude-sonnet": 0.003,
            "claude-opus": 0.015,
            "gemini-pro": 0.001,
            "gemini-flash": 0.0001,
        }
        
        primary_cost = (tokens / 1000) * cost_per_1k.get(primary, 0.01)
        supporting_cost = sum(
            (tokens / 1000) * cost_per_1k.get(model, 0.01) 
            for model in supporting
        )
        
        return primary_cost + supporting_cost
    
    def _estimate_latency(self, strategy: str, complexity: QueryComplexity) -> float:
        """Estimate response latency."""
        base_latency = {
            "single": 3.0,
            "parallel": 5.0,
            "sequential": 8.0,
            "consensus": 12.0,
        }
        
        complexity_multiplier = {
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MODERATE: 1.0,
            QueryComplexity.COMPLEX: 1.5,
            QueryComplexity.EXPERT: 2.0,
        }
        
        return base_latency[strategy] * complexity_multiplier[complexity]
    
    def _get_confidence_threshold(self, complexity: QueryComplexity) -> float:
        """Get confidence threshold based on complexity."""
        return {
            QueryComplexity.SIMPLE: 0.7,
            QueryComplexity.MODERATE: 0.8,
            QueryComplexity.COMPLEX: 0.85,
            QueryComplexity.EXPERT: 0.9,
        }[complexity]
    
    def _get_expected_quality(self, domain: QueryDomain, complexity: QueryComplexity) -> float:
        """Get expected quality score."""
        base_quality = 0.85
        
        # Adjust for domain difficulty
        domain_difficulty = {
            QueryDomain.SECURITY: 0.9,
            QueryDomain.ARCHITECTURE: 0.9,
            QueryDomain.PERFORMANCE: 0.85,
            QueryDomain.GENERAL: 0.8,
        }
        
        return min(0.95, base_quality + domain_difficulty.get(domain, 0.0) * 0.1)
    
    def _generate_reasoning(self, domain: QueryDomain, complexity: QueryComplexity,
                          urgency: QueryUrgency, strategy: str) -> str:
        """Generate explanation for routing decision."""
        return (
            f"Detected {domain.value} query with {complexity.value} complexity and {urgency.value} urgency. "
            f"Using {strategy} orchestration strategy for optimal balance of quality and performance."
        )