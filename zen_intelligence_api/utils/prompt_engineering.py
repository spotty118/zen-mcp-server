"""
Advanced Prompt Engineering for Multi-Model Intelligence

This module provides context-aware system prompts and advanced prompt
crafting techniques optimized for different models and use cases.
"""

from typing import Dict, List, Optional, Any
from ..core.intelligence import QueryDomain, QueryComplexity, QueryUrgency


class ZenPromptEngineer:
    """Advanced prompt engineering for multi-model orchestration."""
    
    # Model-specific instruction templates
    MODEL_INSTRUCTIONS = {
        "gpt-4": {
            "strengths": "architectural design, optimization, code structure",
            "focus": "systematic analysis, best practices, scalable solutions",
            "style": "detailed technical explanations with concrete examples"
        },
        "gpt-5": {
            "strengths": "advanced reasoning, complex problem solving, innovation",
            "focus": "deep analysis, creative solutions, comprehensive understanding",
            "style": "thorough reasoning with step-by-step explanations"
        },
        "claude-sonnet": {
            "strengths": "security analysis, code review, synthesis, documentation",
            "focus": "thoroughness, accuracy, risk assessment, clarity",
            "style": "clear, well-structured analysis with actionable insights"
        },
        "claude-opus": {
            "strengths": "research, strategic planning, complex analysis",
            "focus": "comprehensive understanding, long-term implications, nuanced thinking",
            "style": "deep, contextual analysis with strategic insights"
        },
        "gemini-pro": {
            "strengths": "debugging, edge cases, systematic testing, alternatives",
            "focus": "problem identification, root cause analysis, validation",
            "style": "methodical investigation with multiple perspectives"
        },
        "gemini-flash": {
            "strengths": "quick responses, summarization, classification",
            "focus": "speed, efficiency, direct answers",
            "style": "concise, practical responses with key insights"
        }
    }
    
    # Domain-specific context templates
    DOMAIN_CONTEXTS = {
        QueryDomain.ARCHITECTURE: {
            "context": "system design and architectural decisions",
            "key_aspects": ["scalability", "maintainability", "performance", "security", "patterns"],
            "considerations": ["future growth", "technical debt", "team capabilities", "constraints"]
        },
        QueryDomain.SECURITY: {
            "context": "security analysis and vulnerability assessment", 
            "key_aspects": ["vulnerabilities", "attack vectors", "compliance", "best practices"],
            "considerations": ["threat modeling", "risk assessment", "mitigation strategies"]
        },
        QueryDomain.DEBUGGING: {
            "context": "bug analysis and troubleshooting",
            "key_aspects": ["root cause", "symptoms", "reproduction", "fix strategies"],
            "considerations": ["impact assessment", "testing", "prevention"]
        },
        QueryDomain.PERFORMANCE: {
            "context": "performance optimization and analysis",
            "key_aspects": ["bottlenecks", "metrics", "profiling", "optimization"],
            "considerations": ["trade-offs", "resource usage", "monitoring"]
        },
        QueryDomain.CODE_REVIEW: {
            "context": "code quality assessment and improvement",
            "key_aspects": ["readability", "maintainability", "correctness", "standards"],
            "considerations": ["team guidelines", "best practices", "refactoring opportunities"]
        }
    }
    
    def generate_intelligent_prompt(
        self, 
        model: str, 
        role: str, 
        context: Dict[str, Any], 
        query_analysis: Dict[str, Any],
        orchestration_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate optimized prompts based on model capabilities and context.
        
        Args:
            model: Target model name
            role: Role in orchestration (primary, supporting, validator, synthesizer)
            context: Request context and conversation history
            query_analysis: Analysis of the query (domain, complexity, etc.)
            orchestration_metadata: Additional orchestration context
            
        Returns:
            Optimized system prompt for the model
        """
        
        # Get model-specific instructions
        model_info = self.MODEL_INSTRUCTIONS.get(model, self.MODEL_INSTRUCTIONS["gpt-4"])
        
        # Get domain context
        domain = query_analysis.get("domain", QueryDomain.GENERAL)
        domain_context = self.DOMAIN_CONTEXTS.get(domain, {})
        
        # Get complexity and urgency
        complexity = query_analysis.get("complexity", QueryComplexity.MODERATE)
        urgency = query_analysis.get("urgency", QueryUrgency.IMPLEMENTATION)
        
        # Build the prompt
        prompt_parts = []
        
        # Role and expertise
        prompt_parts.append(self._generate_role_section(model, role, model_info))
        
        # Context awareness
        if orchestration_metadata:
            prompt_parts.append(self._generate_collaboration_section(orchestration_metadata))
        
        # Domain expertise
        if domain_context:
            prompt_parts.append(self._generate_domain_section(domain, domain_context))
        
        # Quality expectations
        prompt_parts.append(self._generate_quality_section(complexity, urgency, query_analysis))
        
        # Previous context
        if context.get("previous_responses"):
            prompt_parts.append(self._generate_context_section(context))
        
        # Instructions
        prompt_parts.append(self._generate_instructions_section(model_info, role))
        
        return "\n\n".join(prompt_parts)
    
    def _generate_role_section(self, model: str, role: str, model_info: Dict[str, Any]) -> str:
        """Generate role and expertise section."""
        
        role_descriptions = {
            "primary": f"You are the PRIMARY MODEL in a collaborative AI team",
            "supporting": f"You are a SUPPORTING MODEL providing additional expertise",
            "validator": f"You are a VALIDATOR MODEL reviewing and validating responses",
            "synthesizer": f"You are a SYNTHESIZER MODEL combining multiple perspectives"
        }
        
        base_role = role_descriptions.get(role, role_descriptions["primary"])
        
        return f"""
{base_role} using {model}.

Your specialized expertise includes: {model_info['strengths']}
Your focus areas: {model_info['focus']}
Your communication style: {model_info['style']}
        """.strip()
    
    def _generate_collaboration_section(self, orchestration_metadata: Dict[str, Any]) -> str:
        """Generate collaboration context section."""
        
        strategy = orchestration_metadata.get("strategy", "single")
        other_models = orchestration_metadata.get("other_models", [])
        
        if strategy == "single":
            return "You are working independently on this analysis."
        
        elif strategy == "parallel":
            return f"""
COLLABORATIVE CONTEXT:
You are working in PARALLEL with other AI models: {', '.join(other_models)}
Each model will provide independent analysis. Focus on your expertise areas while being comprehensive.
Your response will be synthesized with others to provide the best possible answer.
            """.strip()
        
        elif strategy == "sequential":
            return f"""
COLLABORATIVE CONTEXT:
You are part of a SEQUENTIAL workflow with models: {', '.join(other_models)}
Build upon insights from previous models while adding your unique expertise.
Reference and extend previous analysis where appropriate.
            """.strip()
        
        elif strategy == "consensus":
            return f"""
COLLABORATIVE CONTEXT:
You are participating in CONSENSUS BUILDING with models: {', '.join(other_models)}
Your analysis will be compared with others to identify agreements and resolve disagreements.
Be clear about your confidence levels and reasoning behind your conclusions.
            """.strip()
        
        return ""
    
    def _generate_domain_section(self, domain: QueryDomain, domain_context: Dict[str, Any]) -> str:
        """Generate domain-specific context section."""
        
        return f"""
DOMAIN EXPERTISE: {domain.value.replace('_', ' ').title()}
Context: {domain_context.get('context', '')}

Key aspects to consider:
{chr(10).join(f"• {aspect}" for aspect in domain_context.get('key_aspects', []))}

Important considerations:
{chr(10).join(f"• {consideration}" for consideration in domain_context.get('considerations', []))}
        """.strip()
    
    def _generate_quality_section(
        self, 
        complexity: QueryComplexity, 
        urgency: QueryUrgency, 
        query_analysis: Dict[str, Any]
    ) -> str:
        """Generate quality expectations section."""
        
        complexity_guidance = {
            QueryComplexity.SIMPLE: "Provide clear, direct answers with essential details.",
            QueryComplexity.MODERATE: "Provide comprehensive analysis with good depth and examples.",
            QueryComplexity.COMPLEX: "Provide thorough, multi-faceted analysis with detailed reasoning.",
            QueryComplexity.EXPERT: "Provide expert-level analysis considering advanced concepts and edge cases."
        }
        
        urgency_guidance = {
            QueryUrgency.RESEARCH: "Take time for thorough exploration and comprehensive analysis.",
            QueryUrgency.IMPLEMENTATION: "Balance thoroughness with practical, actionable guidance.", 
            QueryUrgency.CRITICAL_FIX: "Prioritize immediate, practical solutions while ensuring correctness."
        }
        
        confidence_target = query_analysis.get("confidence_threshold", 0.8)
        quality_target = query_analysis.get("quality_threshold", 0.85)
        
        return f"""
QUALITY EXPECTATIONS:
Complexity Level: {complexity.value} - {complexity_guidance.get(complexity, '')}
Urgency Level: {urgency.value} - {urgency_guidance.get(urgency, '')}

Target Confidence: {confidence_target:.1%} - Be appropriately confident in your analysis
Target Quality: {quality_target:.1%} - Ensure high-quality, actionable insights
        """.strip()
    
    def _generate_context_section(self, context: Dict[str, Any]) -> str:
        """Generate previous context section."""
        
        previous_responses = context.get("previous_responses", [])
        files_analyzed = context.get("files_analyzed", [])
        
        context_parts = []
        
        if previous_responses:
            context_parts.append("PREVIOUS ANALYSIS:")
            for i, response in enumerate(previous_responses[-3:], 1):  # Last 3 responses
                model_name = response.get("model", "Unknown")
                content_preview = response.get("content", "")[:200] + "..."
                context_parts.append(f"{i}. {model_name}: {content_preview}")
        
        if files_analyzed:
            context_parts.append(f"\nFILES ANALYZED: {', '.join(files_analyzed[-10:])}")  # Last 10 files
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _generate_instructions_section(self, model_info: Dict[str, Any], role: str) -> str:
        """Generate final instructions section."""
        
        base_instructions = """
CRITICAL INSTRUCTIONS:
• Provide clear, well-structured analysis following your expertise and style
• Include confidence levels for key conclusions
• Reference specific evidence and reasoning
• Identify any limitations or uncertainties
• Focus on actionable insights and recommendations
        """.strip()
        
        role_specific = {
            "primary": "• Take ownership of the complete analysis and final recommendations",
            "supporting": "• Focus on your specialized expertise to complement the primary analysis", 
            "validator": "• Critically evaluate the analysis for accuracy and completeness",
            "synthesizer": "• Combine multiple perspectives into a coherent, comprehensive response"
        }
        
        if role in role_specific:
            base_instructions += f"\n{role_specific[role]}"
        
        return base_instructions
    
    def create_consensus_prompt(
        self, 
        responses: List[Dict[str, Any]], 
        disagreement_points: List[str]
    ) -> str:
        """Create prompt for consensus building across multiple responses."""
        
        return f"""
CONSENSUS BUILDING TASK:

You are analyzing {len(responses)} different AI model responses to build consensus and resolve disagreements.

MODEL RESPONSES:
{chr(10).join(f"{i+1}. {response.get('model', 'Unknown')}: {response.get('content', '')[:300]}..." for i, response in enumerate(responses))}

IDENTIFIED DISAGREEMENTS:
{chr(10).join(f"• {point}" for point in disagreement_points)}

INSTRUCTIONS:
1. Identify areas of strong agreement across models
2. Analyze disagreement points and evaluate each perspective  
3. Synthesize a balanced conclusion that considers all valid points
4. Highlight remaining uncertainties or areas requiring additional expertise
5. Provide a confidence score for your consensus conclusion

Focus on creating a response that captures the best insights from all models while resolving conflicts through evidence-based reasoning.
        """.strip()
    
    def create_refinement_prompt(
        self, 
        original_response: str, 
        feedback: str, 
        target_improvements: List[str]
    ) -> str:
        """Create prompt for iterative refinement based on feedback."""
        
        return f"""
ITERATIVE REFINEMENT TASK:

ORIGINAL RESPONSE:
{original_response}

FEEDBACK FOR IMPROVEMENT:
{feedback}

SPECIFIC IMPROVEMENT TARGETS:
{chr(10).join(f"• {improvement}" for improvement in target_improvements)}

INSTRUCTIONS:
1. Review the original response and identify areas for enhancement
2. Address the specific feedback while maintaining the core insights
3. Improve clarity, accuracy, and actionability based on targets
4. Ensure the refined response builds upon rather than replaces good elements
5. Maintain consistency with your model's expertise and style

Provide an enhanced response that incorporates the feedback while preserving the valuable aspects of the original analysis.
        """.strip()