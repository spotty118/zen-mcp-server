"""
Consensus Building System for Multi-Model Intelligence

This module provides weighted confidence scoring and iterative refinement
across multiple AI models to build consensus on complex decisions.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from ..models.orchestration_response import ModelResponse


class ConsensusPoint(BaseModel):
    """A point of agreement between models."""
    
    content: str = Field(..., description="The agreed-upon content")
    models: List[str] = Field(..., description="Models that agree on this point")
    confidence: float = Field(..., description="Combined confidence score")
    weight: float = Field(..., description="Importance weight of this consensus point")


class DisagreementPoint(BaseModel):
    """A point of disagreement between models."""
    
    topic: str = Field(..., description="Topic of disagreement")
    positions: Dict[str, str] = Field(..., description="Model positions {model: position}")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence in each position")
    resolution_suggestion: Optional[str] = Field(None, description="Suggested resolution")


class ConsensusAnalysis(BaseModel):
    """Analysis of consensus and disagreements across model responses."""
    
    consensus_score: float = Field(..., description="Overall consensus score (0-1)")
    agreement_points: List[ConsensusPoint] = Field(default_factory=list)
    disagreement_points: List[DisagreementPoint] = Field(default_factory=list)
    dominant_themes: List[str] = Field(default_factory=list)
    outlier_responses: List[str] = Field(default_factory=list)
    synthesis_recommendations: List[str] = Field(default_factory=list)


class ConsensusBuilder:
    """Builds consensus from multiple model responses using sophisticated analysis."""
    
    def __init__(self):
        """Initialize the consensus builder."""
        # Pattern matching for different types of content
        self.technical_patterns = [
            r'\b(function|class|method|variable|parameter)\s+(\w+)\b',
            r'\b(import|from|export|require)\s+[\w\.]+\b',
            r'\b(error|exception|bug|issue)\s+(?:with|in|at)\s+[\w\.]+\b',
        ]
        
        self.opinion_patterns = [
            r'\b(should|must|need to|have to|ought to)\b',
            r'\b(recommend|suggest|advise|propose)\b',
            r'\b(better|worse|best|worst|optimal|ideal)\b',
        ]
        
        self.fact_patterns = [
            r'\b(is|are|was|were|has|have|will)\s+\w+',
            r'\b\d+(\.\d+)?\s*(seconds|minutes|hours|days|MB|GB|TB|%)\b',
            r'\b(true|false|yes|no|correct|incorrect)\b',
        ]
    
    async def analyze_responses(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Analyze multiple model responses to find consensus and disagreements.
        
        Args:
            responses: List of model responses to analyze
            
        Returns:
            Dictionary containing consensus analysis
        """
        if not responses:
            return {
                "consensus_score": 0.0,
                "agreements": [],
                "disagreements": [],
                "responses": []
            }
        
        # Extract structured content from responses
        structured_responses = self._extract_structured_content(responses)
        
        # Find agreement points
        agreements = self._find_agreements(structured_responses)
        
        # Find disagreement points  
        disagreements = self._find_disagreements(structured_responses)
        
        # Calculate overall consensus score
        consensus_score = self._calculate_consensus_score(agreements, disagreements, responses)
        
        # Generate synthesis recommendations
        synthesis_recommendations = self._generate_synthesis_recommendations(
            agreements, disagreements, responses
        )
        
        # Build analysis object
        analysis = ConsensusAnalysis(
            consensus_score=consensus_score,
            agreement_points=agreements,
            disagreement_points=disagreements,
            dominant_themes=self._extract_dominant_themes(structured_responses),
            outlier_responses=self._identify_outliers(responses),
            synthesis_recommendations=synthesis_recommendations
        )
        
        return {
            "consensus_score": consensus_score,
            "agreements": agreements,
            "disagreements": disagreements,
            "analysis": analysis,
            "responses": responses
        }
    
    def _extract_structured_content(self, responses: List[ModelResponse]) -> List[Dict[str, Any]]:
        """Extract structured content from raw model responses."""
        structured = []
        
        for response in responses:
            if response.error or not response.content:
                continue
                
            # Extract different types of content
            content_analysis = {
                "model": response.model,
                "confidence": response.confidence_score,
                "raw_content": response.content,
                "technical_elements": self._extract_technical_elements(response.content),
                "opinions": self._extract_opinions(response.content),
                "facts": self._extract_facts(response.content),
                "key_phrases": self._extract_key_phrases(response.content),
                "sentiment": self._analyze_sentiment(response.content),
                "word_count": len(response.content.split()),
                "complexity_score": self._calculate_complexity_score(response.content)
            }
            
            structured.append(content_analysis)
        
        return structured
    
    def _extract_technical_elements(self, content: str) -> List[str]:
        """Extract technical elements like function names, imports, etc."""
        elements = []
        
        for pattern in self.technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                elements.extend([' '.join(match) for match in matches])
            else:
                elements.extend(matches)
        
        return list(set(elements))  # Remove duplicates
    
    def _extract_opinions(self, content: str) -> List[str]:
        """Extract opinion statements and recommendations."""
        opinions = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains opinion patterns
            for pattern in self.opinion_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    opinions.append(sentence)
                    break
        
        return opinions
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract factual statements."""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence contains fact patterns
            for pattern in self.fact_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    facts.append(sentence)
                    break
        
        return facts
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases and important terms."""
        # Simple extraction based on capitalized words and technical terms
        phrases = []
        
        # Find quoted text
        quoted = re.findall(r'"([^"]*)"', content)
        phrases.extend(quoted)
        
        # Find code snippets in backticks
        code_snippets = re.findall(r'`([^`]*)`', content)
        phrases.extend(code_snippets)
        
        # Find important technical terms (simplified)
        technical_terms = re.findall(
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b',  # CamelCase
            content
        )
        phrases.extend(technical_terms)
        
        return list(set(phrases))
    
    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis."""
        positive_words = ['good', 'great', 'excellent', 'optimal', 'best', 'recommend', 'should']
        negative_words = ['bad', 'poor', 'worst', 'avoid', 'problem', 'issue', 'error', 'bug']
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score based on various factors."""
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        # Factors contributing to complexity
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        long_words = sum(1 for word in words if len(word) > 6)
        technical_terms = len(self._extract_technical_elements(content))
        
        # Normalize to 0-1 scale
        complexity = min(1.0, (
            (avg_words_per_sentence / 20) * 0.3 +
            (long_words / len(words)) * 0.4 +
            (technical_terms / max(len(words), 1)) * 0.3
        ))
        
        return complexity
    
    def _find_agreements(self, structured_responses: List[Dict[str, Any]]) -> List[ConsensusPoint]:
        """Find points of agreement between models."""
        agreements = []
        
        if len(structured_responses) < 2:
            return agreements
        
        # Compare technical elements
        tech_agreements = self._find_element_agreements(
            structured_responses, 'technical_elements'
        )
        agreements.extend(tech_agreements)
        
        # Compare key phrases
        phrase_agreements = self._find_element_agreements(
            structured_responses, 'key_phrases'
        )
        agreements.extend(phrase_agreements)
        
        # Compare opinions
        opinion_agreements = self._find_semantic_agreements(
            structured_responses, 'opinions'
        )
        agreements.extend(opinion_agreements)
        
        return agreements
    
    def _find_element_agreements(
        self, 
        structured_responses: List[Dict[str, Any]], 
        element_type: str
    ) -> List[ConsensusPoint]:
        """Find agreements on specific elements."""
        agreements = []
        
        # Collect all elements from all responses
        all_elements = {}
        for response in structured_responses:
            elements = response.get(element_type, [])
            for element in elements:
                if element not in all_elements:
                    all_elements[element] = []
                all_elements[element].append({
                    'model': response['model'],
                    'confidence': response['confidence']
                })
        
        # Find elements mentioned by multiple models
        for element, mentions in all_elements.items():
            if len(mentions) >= 2:  # At least 2 models agree
                models = [m['model'] for m in mentions]
                avg_confidence = sum(m['confidence'] for m in mentions) / len(mentions)
                
                agreement = ConsensusPoint(
                    content=element,
                    models=models,
                    confidence=avg_confidence,
                    weight=len(mentions) / len(structured_responses)  # Weight by how many agree
                )
                agreements.append(agreement)
        
        return agreements
    
    def _find_semantic_agreements(
        self, 
        structured_responses: List[Dict[str, Any]], 
        element_type: str
    ) -> List[ConsensusPoint]:
        """Find semantic agreements in opinions or facts."""
        agreements = []
        
        # This is a simplified version - in production, you'd use more sophisticated NLP
        # For now, we'll look for similar phrases or keywords
        
        all_elements = []
        for response in structured_responses:
            elements = response.get(element_type, [])
            for element in elements:
                all_elements.append({
                    'content': element,
                    'model': response['model'],
                    'confidence': response['confidence']
                })
        
        # Group similar elements (simplified similarity check)
        groups = self._group_similar_elements(all_elements)
        
        for group in groups:
            if len(group) >= 2:  # At least 2 models have similar content
                models = [item['model'] for item in group]
                avg_confidence = sum(item['confidence'] for item in group) / len(group)
                
                # Take the most detailed content as representative
                representative_content = max(group, key=lambda x: len(x['content']))['content']
                
                agreement = ConsensusPoint(
                    content=representative_content,
                    models=models,
                    confidence=avg_confidence,
                    weight=len(group) / len(structured_responses)
                )
                agreements.append(agreement)
        
        return agreements
    
    def _group_similar_elements(self, elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group semantically similar elements."""
        groups = []
        used_indices = set()
        
        for i, element1 in enumerate(elements):
            if i in used_indices:
                continue
                
            group = [element1]
            used_indices.add(i)
            
            for j, element2 in enumerate(elements[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                # Simple similarity check based on common words
                similarity = self._calculate_text_similarity(
                    element1['content'], 
                    element2['content']
                )
                
                if similarity > 0.3:  # Threshold for similarity
                    group.append(element2)
                    used_indices.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_disagreements(self, structured_responses: List[Dict[str, Any]]) -> List[DisagreementPoint]:
        """Find points of disagreement between models."""
        disagreements = []
        
        if len(structured_responses) < 2:
            return disagreements
        
        # Find disagreements in opinions
        opinion_disagreements = self._find_opinion_disagreements(structured_responses)
        disagreements.extend(opinion_disagreements)
        
        # Find disagreements in sentiment
        sentiment_disagreement = self._find_sentiment_disagreement(structured_responses)
        if sentiment_disagreement:
            disagreements.append(sentiment_disagreement)
        
        return disagreements
    
    def _find_opinion_disagreements(self, structured_responses: List[Dict[str, Any]]) -> List[DisagreementPoint]:
        """Find disagreements in opinions and recommendations."""
        disagreements = []
        
        # Collect all opinions
        all_opinions = []
        for response in structured_responses:
            opinions = response.get('opinions', [])
            for opinion in opinions:
                all_opinions.append({
                    'content': opinion,
                    'model': response['model'],
                    'confidence': response['confidence']
                })
        
        # Look for conflicting opinions (simplified)
        conflict_keywords = [
            ('should', 'should not'),
            ('recommend', 'avoid'),
            ('best', 'worst'),
            ('good', 'bad'),
            ('use', 'avoid'),
            ('enable', 'disable')
        ]
        
        for positive_keyword, negative_keyword in conflict_keywords:
            positive_opinions = [
                op for op in all_opinions 
                if positive_keyword in op['content'].lower()
            ]
            negative_opinions = [
                op for op in all_opinions 
                if negative_keyword in op['content'].lower()
            ]
            
            if positive_opinions and negative_opinions:
                positions = {}
                confidence_scores = {}
                
                for op in positive_opinions:
                    positions[op['model']] = op['content']
                    confidence_scores[op['model']] = op['confidence']
                
                for op in negative_opinions:
                    positions[op['model']] = op['content']
                    confidence_scores[op['model']] = op['confidence']
                
                disagreement = DisagreementPoint(
                    topic=f"Opinion on {positive_keyword} vs {negative_keyword}",
                    positions=positions,
                    confidence_scores=confidence_scores,
                    resolution_suggestion=self._suggest_opinion_resolution(
                        positive_opinions, negative_opinions
                    )
                )
                disagreements.append(disagreement)
        
        return disagreements
    
    def _find_sentiment_disagreement(self, structured_responses: List[Dict[str, Any]]) -> Optional[DisagreementPoint]:
        """Find disagreement in overall sentiment."""
        sentiments = [(resp['model'], resp['sentiment']) for resp in structured_responses]
        
        unique_sentiments = set(sentiment for _, sentiment in sentiments)
        
        if len(unique_sentiments) > 1:
            positions = {model: sentiment for model, sentiment in sentiments}
            confidence_scores = {resp['model']: resp['confidence'] for resp in structured_responses}
            
            return DisagreementPoint(
                topic="Overall sentiment",
                positions=positions,
                confidence_scores=confidence_scores,
                resolution_suggestion="Consider the context and specific aspects being evaluated"
            )
        
        return None
    
    def _suggest_opinion_resolution(
        self, 
        positive_opinions: List[Dict[str, Any]], 
        negative_opinions: List[Dict[str, Any]]
    ) -> str:
        """Suggest resolution for conflicting opinions."""
        
        # Simple resolution based on confidence scores
        avg_positive_confidence = sum(op['confidence'] for op in positive_opinions) / len(positive_opinions)
        avg_negative_confidence = sum(op['confidence'] for op in negative_opinions) / len(negative_opinions)
        
        if avg_positive_confidence > avg_negative_confidence:
            return "Positive opinion has higher average confidence"
        elif avg_negative_confidence > avg_positive_confidence:
            return "Negative opinion has higher average confidence"
        else:
            return "Consider both perspectives and context-specific factors"
    
    def _calculate_consensus_score(
        self, 
        agreements: List[ConsensusPoint], 
        disagreements: List[DisagreementPoint], 
        responses: List[ModelResponse]
    ) -> float:
        """Calculate overall consensus score."""
        
        if not responses:
            return 0.0
        
        # Base score from agreements vs disagreements
        total_points = len(agreements) + len(disagreements)
        if total_points == 0:
            agreement_ratio = 0.5  # Neutral if no clear agreements or disagreements
        else:
            agreement_ratio = len(agreements) / total_points
        
        # Weight by confidence and model count
        if agreements:
            avg_agreement_confidence = sum(
                agreement.confidence * agreement.weight 
                for agreement in agreements
            ) / sum(agreement.weight for agreement in agreements)
        else:
            avg_agreement_confidence = 0.0
        
        # Factor in response count (more models = higher potential consensus)
        model_factor = min(1.0, len(responses) / 3.0)  # Optimal around 3 models
        
        # Combine factors
        consensus_score = (
            agreement_ratio * 0.4 +
            avg_agreement_confidence * 0.4 +
            model_factor * 0.2
        )
        
        return min(0.95, consensus_score)  # Cap at 0.95 to indicate uncertainty always exists
    
    def _extract_dominant_themes(self, structured_responses: List[Dict[str, Any]]) -> List[str]:
        """Extract dominant themes across all responses."""
        
        # Collect all key phrases
        all_phrases = []
        for response in structured_responses:
            all_phrases.extend(response.get('key_phrases', []))
        
        # Count frequency
        phrase_counts = {}
        for phrase in all_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Return most common themes
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in sorted_phrases[:5] if count > 1]
    
    def _identify_outliers(self, responses: List[ModelResponse]) -> List[str]:
        """Identify responses that are significantly different from others."""
        
        if len(responses) < 3:
            return []  # Need at least 3 responses to identify outliers
        
        outliers = []
        
        # Simple outlier detection based on response length and confidence
        lengths = [len(response.content.split()) for response in responses]
        confidences = [response.confidence_score for response in responses]
        
        avg_length = sum(lengths) / len(lengths)
        avg_confidence = sum(confidences) / len(confidences)
        
        for i, response in enumerate(responses):
            length_diff = abs(lengths[i] - avg_length) / avg_length
            confidence_diff = abs(confidences[i] - avg_confidence)
            
            # Mark as outlier if significantly different
            if length_diff > 0.5 or confidence_diff > 0.3:
                outliers.append(response.model)
        
        return outliers
    
    def _generate_synthesis_recommendations(
        self, 
        agreements: List[ConsensusPoint], 
        disagreements: List[DisagreementPoint], 
        responses: List[ModelResponse]
    ) -> List[str]:
        """Generate recommendations for synthesizing the responses."""
        
        recommendations = []
        
        # Recommend based on agreements
        if agreements:
            high_confidence_agreements = [
                a for a in agreements 
                if a.confidence > 0.8 and a.weight > 0.5
            ]
            if high_confidence_agreements:
                recommendations.append(
                    f"Emphasize {len(high_confidence_agreements)} high-confidence consensus points"
                )
        
        # Recommend based on disagreements
        if disagreements:
            recommendations.append(
                f"Address {len(disagreements)} areas of disagreement with balanced perspective"
            )
        
        # Recommend based on model performance
        best_model = max(responses, key=lambda r: r.confidence_score)
        if best_model.confidence_score > 0.9:
            recommendations.append(
                f"Weight {best_model.model} response higher due to exceptional confidence"
            )
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Synthesize responses with equal weighting")
        
        return recommendations