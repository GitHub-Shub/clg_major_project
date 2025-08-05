# backend/explainable_ai.py
"""
Phase 4: Explainable AI Module for Motor Vehicles Act RAG Pipeline
================================================================

This module provides comprehensive explainability features for the RAG pipeline,
making the AI decision-making process transparent and trustworthy.

Features:
- Source attribution with legal section mapping
- Confidence score explanations
- Cache level transparency
- Query processing path visualization
- Retrieval result explanations
- Legal reasoning insights
- Knowledge gap identification
- Multi-level explanation detail (basic, detailed, debug)
"""

import json
import os
import time
import logging
import jsonify
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
from pathlib import Path

from requests import request


class ExplanationLevel(Enum):
    """Different levels of explanation detail."""
    BASIC = "basic"           # End-user friendly explanations
    DETAILED = "detailed"     # More technical explanations
    DEBUG = "debug"          # Full technical details for developers


class CacheLevel(Enum):
    """Cache levels used in the RAG pipeline."""
    FAQ = "faq"
    EXACT_QUERY = "exact_query"
    SEMANTIC = "semantic"
    PRECOMPUTED = "precomputed"
    EMBEDDING_CACHED = "embedding_cached"
    RETRIEVAL_CACHED = "retrieval_cached"
    FULL_RAG = "full_rag"
    ERROR = "error"


@dataclass
class SourceAttribution:
    """Represents attribution to a source document chunk."""
    chunk_id: int
    text: str
    similarity_score: float
    legal_sections: List[str]
    page_reference: Optional[str] = None
    confidence_contribution: float = 0.0
    relevance_explanation: str = ""


@dataclass
class ConfidenceExplanation:
    """Explains the confidence score of a response."""
    overall_confidence: float
    cache_level_confidence: float
    source_quality_confidence: float
    legal_section_confidence: float
    query_clarity_confidence: float
    factors_boosting_confidence: List[str]
    factors_reducing_confidence: List[str]
    confidence_category: str  # "high", "medium", "low"


@dataclass
class RetrievalExplanation:
    """Explains the retrieval process and results."""
    query_embedding_used: bool
    search_strategy: str
    total_chunks_searched: int
    chunks_retrieved: int
    top_similarity_scores: List[float]
    retrieval_time_ms: float
    search_terms_identified: List[str]
    legal_topics_identified: List[str]


@dataclass
class CacheExplanation:
    """Explains which cache level was used and why."""
    cache_level_used: CacheLevel
    cache_hit: bool
    cache_lookup_time_ms: float
    cache_hierarchy_checked: List[str]
    why_this_cache_level: str
    cache_stats: Dict[str, Any]


@dataclass
class LegalAnalysis:
    """Legal-specific analysis of the query and response."""
    legal_sections_referenced: List[str]
    penalty_amounts_mentioned: List[str]
    legal_procedures_involved: List[str]
    applicable_laws: List[str]
    jurisdiction_scope: str
    legal_certainty: str  # "certain", "likely", "unclear"
    alternative_interpretations: List[str]


@dataclass
class QueryAnalysis:
    """Analysis of the user's query."""
    query_intent: str
    query_complexity: str  # "simple", "moderate", "complex"
    ambiguity_level: str   # "clear", "somewhat_ambiguous", "highly_ambiguous"
    legal_domain: str      # e.g., "traffic_violations", "licensing", "documentation"
    key_terms_extracted: List[str]
    implicit_assumptions: List[str]
    potential_follow_up_questions: List[str]


@dataclass
class KnowledgeGaps:
    """Identifies what the system doesn't know or is uncertain about."""
    missing_information: List[str]
    uncertain_areas: List[str]
    out_of_scope_aspects: List[str]
    recommendations_for_clarification: List[str]
    external_resources_suggested: List[str]


@dataclass
class ExplanationResult:
    """Complete explanation of an AI response."""
    query: str
    response: str
    timestamp: str
    processing_time_ms: float
    explanation_level: ExplanationLevel
    
    # Core explanations
    source_attribution: List[SourceAttribution]
    confidence_explanation: ConfidenceExplanation
    retrieval_explanation: RetrievalExplanation
    cache_explanation: CacheExplanation
    legal_analysis: LegalAnalysis
    query_analysis: QueryAnalysis
    knowledge_gaps: KnowledgeGaps
    
    # Meta information
    system_version: str = "Phase 4"
    explanation_id: str = ""


class ExplainableAI:
    """
    Main explainable AI module that provides transparency into RAG pipeline decisions.
    """
    
    def __init__(self, rag_pipeline, config: Optional[Dict] = None):
        """
        Initialize the explainable AI module.
        
        Args:
            rag_pipeline: The RAG pipeline instance to explain
            config: Configuration options
        """
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger('ExplainableAI')
        
        # Default configuration
        self.config = {
            'default_explanation_level': ExplanationLevel.BASIC,
            'max_source_attributions': 5,
            'min_confidence_threshold': 0.3,
            'enable_legal_analysis': True,
            'enable_knowledge_gap_detection': True,
            'save_explanations': True,
            'explanation_cache_size': 1000,
            'legal_section_extraction': True,
            'penalty_amount_extraction': True,
        }
        
        if config:
            self.config.update(config)
        
        # Initialize explanation storage
        self.explanation_cache = {}
        self.explanation_stats = {
            'total_explanations_generated': 0,
            'explanations_by_level': {level.value: 0 for level in ExplanationLevel},
            'explanations_by_cache_level': {level.value: 0 for level in CacheLevel},
            'avg_explanation_generation_time': 0.0
        }
        
        # Legal patterns for analysis
        self.legal_patterns = {
            'section_pattern': r'Section\s+(\d+[A-Z]*)',
            'penalty_pattern': r'₹\s*(\d+(?:,\d+)*)',
            'fine_pattern': r'fine\s+(?:of\s+)?₹?\s*(\d+(?:,\d+)*)',
            'imprisonment_pattern': r'imprisonment\s+(?:up\s+to\s+)?(\d+)\s+(months?|years?)',
            'license_pattern': r'license\s+(suspension|cancellation|impoundment)',
        }
        
        self.logger.info("Explainable AI module initialized")
    
    def explain_response(self, query: str, response: str, processing_metadata: Dict[str, Any], 
                        explanation_level: ExplanationLevel = None) -> ExplanationResult:
        """
        Generate a comprehensive explanation for an AI response.
        
        Args:
            query: The user's original query
            response: The AI's response
            processing_metadata: Metadata from the RAG pipeline processing
            explanation_level: Level of detail for explanation
            
        Returns:
            Complete explanation result
        """
        start_time = time.time()
        
        if explanation_level is None:
            explanation_level = self.config['default_explanation_level']
        
        self.logger.info(f"Generating {explanation_level.value} explanation for query: {query[:50]}...")
        
        try:
            # Generate unique explanation ID
            explanation_id = self._generate_explanation_id(query)
            
            # Analyze the query
            query_analysis = self._analyze_query(query)
            
            # Analyze cache usage
            cache_explanation = self._explain_cache_usage(processing_metadata)
            
            # Analyze source attribution
            source_attribution = self._explain_source_attribution(
                query, response, processing_metadata
            )
            
            # Analyze confidence
            confidence_explanation = self._explain_confidence(
                query, response, processing_metadata, source_attribution
            )
            
            # Analyze retrieval process
            retrieval_explanation = self._explain_retrieval_process(processing_metadata)
            
            # Perform legal analysis
            legal_analysis = self._analyze_legal_aspects(query, response, source_attribution)
            
            # Identify knowledge gaps
            knowledge_gaps = self._identify_knowledge_gaps(
                query, response, source_attribution, legal_analysis
            )
            
            # Create complete explanation
            explanation = ExplanationResult(
                query=query,
                response=response,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
                explanation_level=explanation_level,
                source_attribution=source_attribution,
                confidence_explanation=confidence_explanation,
                retrieval_explanation=retrieval_explanation,
                cache_explanation=cache_explanation,
                legal_analysis=legal_analysis,
                query_analysis=query_analysis,
                knowledge_gaps=knowledge_gaps,
                explanation_id=explanation_id
            )
            
            # Update statistics
            self._update_explanation_stats(explanation, start_time)
            
            # Cache explanation if enabled
            if self.config['save_explanations']:
                self._cache_explanation(explanation)
            
            self.logger.info(f"Explanation generated in {explanation.processing_time_ms:.1f}ms")
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {str(e)}")
            # Return minimal explanation on error
            return self._create_error_explanation(query, response, str(e))
    
    def _generate_explanation_id(self, query: str) -> str:
        """Generate unique ID for explanation."""
        timestamp = str(int(time.time() * 1000))
        query_hash = hash(query.lower().strip()) % 10000
        return f"exp_{timestamp}_{query_hash:04d}"
    
    def _analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze the user's query for intent and characteristics."""
        query_lower = query.lower().strip()
        
        # Determine query intent
        intent_patterns = {
            'penalty_inquiry': ['penalty', 'fine', 'punishment', 'charges'],
            'procedure_inquiry': ['how to', 'process', 'procedure', 'steps'],
            'document_inquiry': ['documents', 'papers', 'certificate', 'required'],
            'legal_clarification': ['what is', 'define', 'meaning', 'explain'],
            'compliance_check': ['can i', 'allowed', 'legal', 'permitted'],
            'consequence_inquiry': ['what happens', 'consequences', 'result', 'outcome']
        }
        
        query_intent = "general_inquiry"
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                query_intent = intent
                break
        
        # Determine complexity
        complexity_indicators = {
            'simple': len(query.split()) <= 8,
            'moderate': 8 < len(query.split()) <= 15,
            'complex': len(query.split()) > 15
        }
        
        query_complexity = "simple"
        for complexity, condition in complexity_indicators.items():
            if condition:
                query_complexity = complexity
                break
        
        # Determine ambiguity
        ambiguity_indicators = [
            'or', 'either', 'maybe', 'might', 'could', 'possibly', 'what if'
        ]
        ambiguity_count = sum(1 for indicator in ambiguity_indicators if indicator in query_lower)
        
        if ambiguity_count == 0:
            ambiguity_level = "clear"
        elif ambiguity_count <= 2:
            ambiguity_level = "somewhat_ambiguous"
        else:
            ambiguity_level = "highly_ambiguous"
        
        # Determine legal domain
        domain_patterns = {
            'licensing': ['license', 'driving', 'learner', 'age', 'permit'],
            'traffic_violations': ['speed', 'signal', 'helmet', 'seatbelt', 'mobile'],
            'penalties_fines': ['penalty', 'fine', 'punishment', 'charges', 'amount'],
            'documentation': ['registration', 'insurance', 'certificate', 'documents'],
            'safety_provisions': ['helmet', 'seatbelt', 'safety', 'protection'],
            'drunk_driving': ['alcohol', 'drunk', 'intoxicated', 'blood alcohol'],
            'vehicle_regulations': ['registration', 'fitness', 'pollution', 'emission'],
            'emergency_provisions': ['golden hour', 'accident', 'emergency', 'medical']
        }
        
        legal_domain = "general"
        max_matches = 0
        for domain, keywords in domain_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > max_matches:
                max_matches = matches
                legal_domain = domain
        
        # Extract key terms
        key_terms = []
        important_words = [
            'penalty', 'fine', 'license', 'helmet', 'registration', 'insurance',
            'drunk', 'speed', 'alcohol', 'certificate', 'documents', 'section'
        ]
        for word in important_words:
            if word in query_lower:
                key_terms.append(word)
        
        # Identify implicit assumptions
        implicit_assumptions = []
        if 'penalty' in query_lower or 'fine' in query_lower:
            implicit_assumptions.append("User assumes violation has occurred")
        if 'can i' in query_lower or 'allowed' in query_lower:
            implicit_assumptions.append("User wants to know legal permissibility")
        
        # Generate potential follow-up questions
        follow_up_questions = self._generate_follow_up_questions(query_intent, legal_domain)
        
        return QueryAnalysis(
            query_intent=query_intent,
            query_complexity=query_complexity,
            ambiguity_level=ambiguity_level,
            legal_domain=legal_domain,
            key_terms_extracted=key_terms,
            implicit_assumptions=implicit_assumptions,
            potential_follow_up_questions=follow_up_questions
        )
    
    def _generate_follow_up_questions(self, intent: str, domain: str) -> List[str]:
        """Generate relevant follow-up questions based on intent and domain."""
        follow_ups = []
        
        if intent == "penalty_inquiry":
            follow_ups.extend([
                "What are the procedures for paying this fine?",
                "Can this penalty be appealed?",
                "Are there repeat offender provisions?"
            ])
        
        if domain == "licensing":
            follow_ups.extend([
                "What documents are required for license application?",
                "How long is the license valid?",
                "What is the renewal process?"
            ])
        
        return follow_ups[:3]  # Limit to top 3
    
    def _explain_cache_usage(self, processing_metadata: Dict[str, Any]) -> CacheExplanation:
        """Explain which cache level was used and why."""
        cache_level_used = processing_metadata.get('cache_level', 'unknown')
        cache_hit = cache_level_used != 'full_rag'
        cache_lookup_time = processing_metadata.get('cache_lookup_time', 0.0)
        
        # Determine cache hierarchy that was checked
        cache_hierarchy_checked = [
            "FAQ Cache",
            "Exact Query Cache",
            "Semantic Cache",
            "Pre-computed Cache",
            "Embedding Cache",
            "Retrieval Cache"
        ]
        
        # Explain why this cache level was used
        cache_explanations = {
            'faq': "Found exact match in FAQ database for common questions",
            'exact_query': "Found exact match for previously asked identical question",
            'semantic': "Found semantically similar question with high confidence",
            'precomputed': "Found pre-computed response for this legal topic",
            'embedding_cached': "Used cached query embedding but retrieved new chunks",
            'retrieval_cached': "Used cached chunk retrieval results",
            'full_rag': "No cache hits - generated fresh response using full RAG pipeline",
            'error': "Error occurred during processing"
        }
        
        why_explanation = cache_explanations.get(cache_level_used, "Unknown cache behavior")
        
        return CacheExplanation(
            cache_level_used=CacheLevel(cache_level_used) if cache_level_used in [e.value for e in CacheLevel] else CacheLevel.ERROR,
            cache_hit=cache_hit,
            cache_lookup_time_ms=round(cache_lookup_time * 1000, 2),
            cache_hierarchy_checked=cache_hierarchy_checked,
            why_this_cache_level=why_explanation,
            cache_stats=processing_metadata.get('cache_stats', {})
        )
    
    def _explain_source_attribution(self, query: str, response: str, 
                                  processing_metadata: Dict[str, Any]) -> List[SourceAttribution]:
        """Generate source attribution explanations."""
        retrieved_chunks = processing_metadata.get('retrieved_chunks', [])
        
        if not retrieved_chunks:
            # For cached responses, try to infer sources from response content
            return self._infer_sources_from_response(response)
        
        attributions = []
        
        for i, chunk in enumerate(retrieved_chunks[:self.config['max_source_attributions']]):
            # Extract legal sections
            legal_sections = self._extract_legal_sections(chunk.get('text', ''))
            
            # Calculate relevance explanation
            relevance_explanation = self._explain_chunk_relevance(
                query, chunk.get('text', ''), chunk.get('similarity_score', 0)
            )
            
            attribution = SourceAttribution(
                chunk_id=chunk.get('metadata', {}).get('chunk_id', i),
                text=chunk.get('text', '')[:300] + "..." if len(chunk.get('text', '')) > 300 else chunk.get('text', ''),
                similarity_score=chunk.get('similarity_score', 0.0),
                legal_sections=legal_sections,
                confidence_contribution=self._calculate_confidence_contribution(chunk, retrieved_chunks),
                relevance_explanation=relevance_explanation
            )
            
            attributions.append(attribution)
        
        return attributions
    
    def _infer_sources_from_response(self, response: str) -> List[SourceAttribution]:
        """Infer likely sources from cached response content."""
        legal_sections = self._extract_legal_sections(response)
        
        if legal_sections:
            # Create synthetic attribution for cached responses
            return [SourceAttribution(
                chunk_id=-1,  # Indicates cached response
                text="Response from cache - original source chunks not available",
                similarity_score=1.0,
                legal_sections=legal_sections,
                confidence_contribution=1.0,
                relevance_explanation="Cached response based on Motor Vehicles Act sections"
            )]
        
        return []
    
    def _extract_legal_sections(self, text: str) -> List[str]:
        """Extract legal section references from text."""
        if not self.config['legal_section_extraction']:
            return []
        
        matches = re.findall(self.legal_patterns['section_pattern'], text, re.IGNORECASE)
        return [f"Section {match}" for match in matches]
    
    def _explain_chunk_relevance(self, query: str, chunk_text: str, similarity_score: float) -> str:
        """Explain why a specific chunk is relevant to the query."""
        query_words = set(query.lower().split())
        chunk_words = set(chunk_text.lower().split())
        
        common_words = query_words.intersection(chunk_words)
        legal_sections = self._extract_legal_sections(chunk_text)
        
        explanations = []
        
        if similarity_score > 0.8:
            explanations.append("High semantic similarity to your question")
        elif similarity_score > 0.6:
            explanations.append("Moderate semantic similarity to your question")
        
        if len(common_words) > 3:
            explanations.append(f"Contains key terms: {', '.join(list(common_words)[:3])}")
        
        if legal_sections:
            explanations.append(f"Contains relevant legal sections: {', '.join(legal_sections[:2])}")
        
        if not explanations:
            explanations.append("General topical relevance to Motor Vehicles Act")
        
        return "; ".join(explanations)
    
    def _calculate_confidence_contribution(self, chunk: Dict, all_chunks: List[Dict]) -> float:
        """Calculate how much this chunk contributes to overall confidence."""
        if not all_chunks:
            return 1.0
        
        chunk_score = chunk.get('similarity_score', 0)
        total_score = sum(c.get('similarity_score', 0) for c in all_chunks)
        
        if total_score == 0:
            return 1.0 / len(all_chunks)
        
        return chunk_score / total_score
    
    def _explain_confidence(self, query: str, response: str, processing_metadata: Dict[str, Any],
                          source_attribution: List[SourceAttribution]) -> ConfidenceExplanation:
        """Generate detailed confidence explanation."""
        # Base confidence from processing metadata
        cache_level = processing_metadata.get('cache_level', 'unknown')
        processing_time = processing_metadata.get('processing_time', 0)
        
        # Calculate component confidence scores
        cache_level_confidence = self._calculate_cache_level_confidence(cache_level)
        source_quality_confidence = self._calculate_source_quality_confidence(source_attribution)
        legal_section_confidence = self._calculate_legal_section_confidence(response)
        query_clarity_confidence = self._calculate_query_clarity_confidence(query)
        
        # Calculate overall confidence
        weights = {'cache': 0.3, 'source': 0.3, 'legal': 0.2, 'clarity': 0.2}
        overall_confidence = (
            cache_level_confidence * weights['cache'] +
            source_quality_confidence * weights['source'] +
            legal_section_confidence * weights['legal'] +
            query_clarity_confidence * weights['clarity']
        )
        
        # Identify confidence factors
        factors_boosting = []
        factors_reducing = []
        
        if cache_level_confidence > 0.8:
            factors_boosting.append("High-confidence cache level used")
        elif cache_level_confidence < 0.5:
            factors_reducing.append("Low-confidence cache level")
        
        if source_quality_confidence > 0.8:
            factors_boosting.append("High-quality source attribution")
        elif source_quality_confidence < 0.5:
            factors_reducing.append("Limited or low-quality sources")
        
        if legal_section_confidence > 0.8:
            factors_boosting.append("Clear legal section references")
        elif legal_section_confidence < 0.5:
            factors_reducing.append("Vague or missing legal references")
        
        if query_clarity_confidence > 0.8:
            factors_boosting.append("Clear and specific query")
        elif query_clarity_confidence < 0.5:
            factors_reducing.append("Ambiguous or unclear query")
        
        if processing_time < 0.1:
            factors_boosting.append("Fast response from high-confidence cache")
        elif processing_time > 2.0:
            factors_reducing.append("Slow processing indicating complex generation")
        
        # Determine confidence category
        if overall_confidence >= 0.9:
            confidence_category = "high"
        elif overall_confidence >= 0.7:
            confidence_category = "medium"
        else:
            confidence_category = "low"
        
        return ConfidenceExplanation(
            overall_confidence=round(overall_confidence, 3),
            cache_level_confidence=round(cache_level_confidence, 3),
            source_quality_confidence=round(source_quality_confidence, 3),
            legal_section_confidence=round(legal_section_confidence, 3),
            query_clarity_confidence=round(query_clarity_confidence, 3),
            factors_boosting_confidence=factors_boosting,
            factors_reducing_confidence=factors_reducing,
            confidence_category=confidence_category
        )
    
    def _calculate_cache_level_confidence(self, cache_level: str) -> float:
        """Calculate confidence based on which cache level was used."""
        confidence_by_level = {
            'faq': 1.0,              # FAQ entries are manually curated
            'exact_query': 0.95,     # Exact matches are very reliable
            'semantic': 0.85,        # Semantic matches are quite reliable
            'precomputed': 0.8,      # Pre-computed responses are topic-based
            'embedding_cached': 0.75, # Partial cache hit
            'retrieval_cached': 0.75, # Partial cache hit
            'full_rag': 0.7,         # Fresh generation, moderate confidence
            'error': 0.0             # Error state
        }
        return confidence_by_level.get(cache_level, 0.5)
    
    def _calculate_source_quality_confidence(self, source_attribution: List[SourceAttribution]) -> float:
        """Calculate confidence based on source quality."""
        if not source_attribution:
            return 0.3
        
        # Factor in similarity scores and legal section presence
        avg_similarity = np.mean([attr.similarity_score for attr in source_attribution])
        sections_present = any(attr.legal_sections for attr in source_attribution)
        
        base_confidence = avg_similarity
        if sections_present:
            base_confidence += 0.2  # Boost for legal sections
        
        return min(1.0, base_confidence)
    
    def _calculate_legal_section_confidence(self, response: str) -> float:
        """Calculate confidence based on legal section specificity."""
        legal_sections = self._extract_legal_sections(response)
        penalty_amounts = re.findall(self.legal_patterns['penalty_pattern'], response)
        
        confidence = 0.5  # Base confidence
        
        if legal_sections:
            confidence += 0.3 * min(1.0, len(legal_sections) / 2)
        
        if penalty_amounts:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_query_clarity_confidence(self, query: str) -> float:
        """Calculate confidence based on query clarity."""
        query_lower = query.lower().strip()
        
        # Factor in question specificity
        specific_terms = [
            'penalty', 'fine', 'section', 'license', 'helmet', 'registration',
            'insurance', 'alcohol', 'speed', 'documents'
        ]
        
        specificity_score = sum(1 for term in specific_terms if term in query_lower)
        specificity_confidence = min(1.0, specificity_score / 3)
        
        # Factor in question structure
        question_words = ['what', 'how', 'when', 'where', 'why', 'which']
        has_question_structure = any(word in query_lower for word in question_words)
        
        structure_confidence = 0.8 if has_question_structure else 0.6
        
        # Combine factors
        overall_clarity = (specificity_confidence * 0.6) + (structure_confidence * 0.4)
        
        return overall_clarity
    
    def _explain_retrieval_process(self, processing_metadata: Dict[str, Any]) -> RetrievalExplanation:
        """Explain the chunk retrieval process."""
        retrieved_chunks = processing_metadata.get('retrieved_chunks', [])
        
        # Determine if embedding was cached
        query_embedding_used = processing_metadata.get('embedding_cache_hit', False)
        
        # Determine search strategy
        search_strategy = "semantic_similarity"
        if processing_metadata.get('retrieval_cache_hit', False):
            search_strategy = "cached_retrieval"
        
        # Extract similarity scores
        similarity_scores = [
            chunk.get('similarity_score', 0) 
            for chunk in retrieved_chunks
        ]
        
        # Identify search terms and topics
        search_terms = processing_metadata.get('key_terms', [])
        legal_topics = processing_metadata.get('legal_topics', [])
        
        return RetrievalExplanation(
            query_embedding_used=query_embedding_used,
            search_strategy=search_strategy,
            total_chunks_searched=processing_metadata.get('total_chunks', 0),
            chunks_retrieved=len(retrieved_chunks),
            top_similarity_scores=similarity_scores[:3],
            retrieval_time_ms=processing_metadata.get('retrieval_time', 0) * 1000,
            search_terms_identified=search_terms,
            legal_topics_identified=legal_topics
        )
    
    def _analyze_legal_aspects(self, query: str, response: str, 
                             source_attribution: List[SourceAttribution]) -> LegalAnalysis:
        """Perform legal-specific analysis of the response."""
        # Extract legal information
        legal_sections = self._extract_legal_sections(response)
        penalty_amounts = re.findall(self.legal_patterns['penalty_pattern'], response)
        
        # Identify legal procedures
        procedure_indicators = [
            'application', 'renewal', 'registration', 'verification', 
            'inspection', 'appeal', 'hearing', 'court'
        ]
        legal_procedures = [
            proc for proc in procedure_indicators 
            if proc in response.lower()
        ]
        
        # Identify applicable laws
        applicable_laws = ["Motor Vehicles Act 1988"]
        if 'bharatiya nyaya sanhita' in response.lower():
            applicable_laws.append("Bharatiya Nyaya Sanhita")
        if 'state rules' in response.lower():
            applicable_laws.append("State Motor Vehicle Rules")
        
        # Determine legal certainty
        certainty_indicators = {
            'certain': ['as per section', 'under section', 'according to'],
            'likely': ['typically', 'usually', 'generally', 'may'],
            'unclear': ['might', 'could', 'possibly', 'varies', 'depends']
        }
        
        legal_certainty = "unclear"
        for certainty, indicators in certainty_indicators.items():
            if any(indicator in response.lower() for indicator in indicators):
                legal_certainty = certainty
                break
        
        # Generate alternative interpretations
        alternative_interpretations = self._generate_alternative_interpretations(query, response)
        
        return LegalAnalysis(
            legal_sections_referenced=legal_sections,
            penalty_amounts_mentioned=[f"₹{amount}" for amount in penalty_amounts],
            legal_procedures_involved=legal_procedures,
            applicable_laws=applicable_laws,
            jurisdiction_scope="India (Central Act + State Rules)",
            legal_certainty=legal_certainty,
            alternative_interpretations=alternative_interpretations
        )
    
    def _generate_alternative_interpretations(self, query: str, response: str) -> List[str]:
        """Generate alternative ways the query could be interpreted."""
        alternatives = []
        
        query_lower = query.lower()
        
        # Check for ambiguous terms
        if 'penalty' in query_lower:
            alternatives.append("Query could refer to civil penalties, criminal charges, or administrative actions")
        
        if 'documents' in query_lower:
            alternatives.append("Could refer to documents required while driving vs. documents needed for registration/license")
        
        if 'fine' in query_lower:
            alternatives.append("Could refer to monetary fine amount vs. fine payment procedures")
        
        return alternatives[:3]  # Limit to most relevant
    
    def _identify_knowledge_gaps(self, query: str, response: str, 
                               source_attribution: List[SourceAttribution],
                               legal_analysis: LegalAnalysis) -> KnowledgeGaps:
        """Identify what the system doesn't know or is uncertain about."""
        missing_information = []
        uncertain_areas = []
        out_of_scope = []
        recommendations = []
        external_resources = []
        
        # Check for missing penalty amounts
        if 'penalty' in query.lower() or 'fine' in query.lower():
            if not legal_analysis.penalty_amounts_mentioned:
                missing_information.append("Specific penalty amounts not found")
                recommendations.append("Check latest state motor vehicle rules for current penalty amounts")
        
        # Check for missing procedural information
        if 'how to' in query.lower() or 'procedure' in query.lower():
            if not legal_analysis.legal_procedures_involved:
                missing_information.append("Detailed procedures not available in source documents")
                recommendations.append("Consult official RTO website for step-by-step procedures")
        
        # Check for state-specific variations
        if 'varies' in response.lower() or 'state rules' in response.lower():
            uncertain_areas.append("State-specific variations may apply")
            recommendations.append("Check your state's motor vehicle rules for specific requirements")
        
        # Check for recent amendments
        if not source_attribution or all(attr.chunk_id == -1 for attr in source_attribution):
            uncertain_areas.append("Information may not reflect latest amendments")
            recommendations.append("Verify with latest Motor Vehicles Act amendments")
        
        # Identify out-of-scope aspects
        if legal_analysis.legal_certainty == "unclear":
            out_of_scope.append("Some aspects may require legal consultation")
            external_resources.append("Consult traffic lawyer for complex cases")
        
        # Standard external resources
        external_resources.extend([
            "Official website: parivahan.gov.in",
            "Local RTO office",
            "Motor Vehicles Act 1988 (full text)"
        ])
        
        return KnowledgeGaps(
            missing_information=missing_information,
            uncertain_areas=uncertain_areas,
            out_of_scope_aspects=out_of_scope,
            recommendations_for_clarification=recommendations,
            external_resources_suggested=external_resources[:4]  # Limit to most relevant
        )
    
    def _update_explanation_stats(self, explanation: ExplanationResult, start_time: float):
        """Update explanation generation statistics."""
        generation_time = time.time() - start_time
        
        self.explanation_stats['total_explanations_generated'] += 1
        self.explanation_stats['explanations_by_level'][explanation.explanation_level.value] += 1
        self.explanation_stats['explanations_by_cache_level'][explanation.cache_explanation.cache_level_used.value] += 1
        
        # Update average generation time
        total_explanations = self.explanation_stats['total_explanations_generated']
        current_avg = self.explanation_stats['avg_explanation_generation_time']
        self.explanation_stats['avg_explanation_generation_time'] = (
            (current_avg * (total_explanations - 1) + generation_time) / total_explanations
        )
    
    def _cache_explanation(self, explanation: ExplanationResult):
        """Cache explanation for potential reuse."""
        if len(self.explanation_cache) >= self.config['explanation_cache_size']:
            # Remove oldest explanation
            oldest_key = min(self.explanation_cache.keys())
            del self.explanation_cache[oldest_key]
        
        self.explanation_cache[explanation.explanation_id] = explanation
    
    def _create_error_explanation(self, query: str, response: str, error: str) -> ExplanationResult:
        """Create minimal explanation for error cases."""
        return ExplanationResult(
            query=query,
            response=response,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0.0,
            explanation_level=ExplanationLevel.BASIC,
            source_attribution=[],
            confidence_explanation=ConfidenceExplanation(
                overall_confidence=0.0,
                cache_level_confidence=0.0,
                source_quality_confidence=0.0,
                legal_section_confidence=0.0,
                query_clarity_confidence=0.0,
                factors_boosting_confidence=[],
                factors_reducing_confidence=[f"System error: {error}"],
                confidence_category="error"
            ),
            retrieval_explanation=RetrievalExplanation(
                query_embedding_used=False,
                search_strategy="error",
                total_chunks_searched=0,
                chunks_retrieved=0,
                top_similarity_scores=[],
                retrieval_time_ms=0.0,
                search_terms_identified=[],
                legal_topics_identified=[]
            ),
            cache_explanation=CacheExplanation(
                cache_level_used=CacheLevel.ERROR,
                cache_hit=False,
                cache_lookup_time_ms=0.0,
                cache_hierarchy_checked=[],
                why_this_cache_level=f"Error occurred: {error}",
                cache_stats={}
            ),
            legal_analysis=LegalAnalysis(
                legal_sections_referenced=[],
                penalty_amounts_mentioned=[],
                legal_procedures_involved=[],
                applicable_laws=[],
                jurisdiction_scope="Unknown",
                legal_certainty="error",
                alternative_interpretations=[]
            ),
            query_analysis=QueryAnalysis(
                query_intent="unknown",
                query_complexity="unknown",
                ambiguity_level="unknown",
                legal_domain="unknown",
                key_terms_extracted=[],
                implicit_assumptions=[],
                potential_follow_up_questions=[]
            ),
            knowledge_gaps=KnowledgeGaps(
                missing_information=[f"System error prevented processing: {error}"],
                uncertain_areas=[],
                out_of_scope_aspects=[],
                recommendations_for_clarification=["Contact system administrator"],
                external_resources_suggested=[]
            ),
            explanation_id=f"error_{int(time.time())}"
        )
    
    def format_explanation(self, explanation: ExplanationResult, 
                         format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Format explanation for different output types.
        
        Args:
            explanation: The explanation result to format
            format_type: "json", "html", "markdown", or "plain_text"
            
        Returns:
            Formatted explanation
        """
        if format_type == "json":
            return self._format_as_json(explanation)
        elif format_type == "html":
            return self._format_as_html(explanation)
        elif format_type == "markdown":
            return self._format_as_markdown(explanation)
        elif format_type == "plain_text":
            return self._format_as_plain_text(explanation)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_as_json(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Format explanation as JSON (for API responses)."""
        # Convert dataclass to dict with proper serialization
        result = asdict(explanation)
        
        # Convert enums to strings
        result['explanation_level'] = explanation.explanation_level.value
        result['cache_explanation']['cache_level_used'] = explanation.cache_explanation.cache_level_used.value
        
        return result
    
    def _format_as_markdown(self, explanation: ExplanationResult) -> str:
        """Format explanation as Markdown (for documentation)."""
        md = f"""# AI Response Explanation

## Query
**Original Question:** {explanation.query}

**AI Response:** {explanation.response}

## Confidence Analysis
- **Overall Confidence:** {explanation.confidence_explanation.overall_confidence:.1%} ({explanation.confidence_explanation.confidence_category})
- **Processing Method:** {explanation.cache_explanation.cache_level_used.value.replace('_', ' ').title()}
- **Processing Time:** {explanation.processing_time_ms:.1f}ms

### Confidence Factors
**Boosting Confidence:**
{chr(10).join(f"- {factor}" for factor in explanation.confidence_explanation.factors_boosting_confidence)}

**Reducing Confidence:**
{chr(10).join(f"- {factor}" for factor in explanation.confidence_explanation.factors_reducing_confidence)}

## Source Attribution
"""
        
        if explanation.source_attribution:
            for i, attr in enumerate(explanation.source_attribution, 1):
                md += f"""
### Source {i}
- **Similarity Score:** {attr.similarity_score:.3f}
- **Legal Sections:** {', '.join(attr.legal_sections) if attr.legal_sections else 'None'}
- **Relevance:** {attr.relevance_explanation}
- **Text:** {attr.text[:200]}...
"""
        else:
            md += "\n*No specific source chunks available (cached response)*\n"
        
        # Add legal analysis if detailed
        if explanation.explanation_level != ExplanationLevel.BASIC:
            md += f"""
## Legal Analysis
- **Legal Sections:** {', '.join(explanation.legal_analysis.legal_sections_referenced)}
- **Penalty Amounts:** {', '.join(explanation.legal_analysis.penalty_amounts_mentioned)}
- **Legal Certainty:** {explanation.legal_analysis.legal_certainty}
- **Applicable Laws:** {', '.join(explanation.legal_analysis.applicable_laws)}
"""
        
        return md
    
    def _format_as_plain_text(self, explanation: ExplanationResult) -> str:
        """Format explanation as plain text (for simple interfaces)."""
        text = f"EXPLANATION FOR: {explanation.query}\n"
        text += f"RESPONSE: {explanation.response}\n\n"
        
        # Basic confidence information
        confidence = explanation.confidence_explanation
        text += f"CONFIDENCE: {confidence.overall_confidence:.1%} ({confidence.confidence_category})\n"
        text += f"PROCESSING: {explanation.cache_explanation.cache_level_used.value.replace('_', ' ').title()}\n"
        text += f"TIME: {explanation.processing_time_ms:.1f}ms\n\n"
        
        # Legal sections
        if explanation.legal_analysis.legal_sections_referenced:
            text += f"LEGAL SECTIONS: {', '.join(explanation.legal_analysis.legal_sections_referenced)}\n"
        
        # Key confidence factors
        if confidence.factors_boosting_confidence:
            text += f"CONFIDENCE BOOSTERS: {'; '.join(confidence.factors_boosting_confidence)}\n"
        
        if confidence.factors_reducing_confidence:
            text += f"CONFIDENCE REDUCERS: {'; '.join(confidence.factors_reducing_confidence)}\n"
        
        return text
    
    def _format_as_html(self, explanation: ExplanationResult) -> str:
        """Format explanation as HTML (for web interfaces)."""
        confidence = explanation.confidence_explanation
        cache_info = explanation.cache_explanation
        
        # Determine confidence color
        confidence_colors = {
            'high': '#28a745',    # Green
            'medium': '#ffc107',  # Yellow
            'low': '#dc3545',     # Red
            'error': '#6c757d'    # Gray
        }
        confidence_color = confidence_colors.get(confidence.confidence_category, '#6c757d')
        
        html = f"""
<div class="ai-explanation" style="font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0;">
    <h3 style="color: #495057; margin-top: 0;">🔍 AI Response Explanation</h3>
    
    <div class="confidence-section" style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
        <h4 style="color: #495057; margin-top: 0;">📊 Confidence Analysis</h4>
        <div style="display: flex; align-items: center; margin: 10px 0;">
            <span style="background: {confidence_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold;">
                {confidence.overall_confidence:.1%} Confidence
            </span>
            <span style="margin-left: 15px; color: #6c757d;">
                {cache_info.cache_level_used.value.replace('_', ' ').title()} • {explanation.processing_time_ms:.1f}ms
            </span>
        </div>
        <p style="color: #495057; margin: 10px 0;"><strong>Why this confidence level:</strong> {cache_info.why_this_cache_level}</p>
    </div>
"""
        
        # Add source attribution if available
        if explanation.source_attribution:
            html += """
    <div class="sources-section" style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
        <h4 style="color: #495057; margin-top: 0;">📚 Source Information</h4>
"""
            for i, attr in enumerate(explanation.source_attribution[:3], 1):
                sections_text = ', '.join(attr.legal_sections) if attr.legal_sections else 'General content'
                html += f"""
        <div style="border-left: 3px solid #007bff; padding-left: 10px; margin: 10px 0;">
            <strong>Source {i}:</strong> {sections_text}<br>
            <span style="color: #6c757d;">Similarity: {attr.similarity_score:.3f} • {attr.relevance_explanation}</span>
        </div>
"""
            html += "    </div>"
        
        # Add legal analysis
        if explanation.legal_analysis.legal_sections_referenced:
            html += f"""
    <div class="legal-section" style="background: white; padding: 15px; border-radius: 6px; margin: 10px 0;">
        <h4 style="color: #495057; margin-top: 0;">⚖️ Legal References</h4>
        <p style="color: #495057;"><strong>Sections:</strong> {', '.join(explanation.legal_analysis.legal_sections_referenced)}</p>
        {f'<p style="color: #495057;"><strong>Penalties:</strong> {", ".join(explanation.legal_analysis.penalty_amounts_mentioned)}</p>' if explanation.legal_analysis.penalty_amounts_mentioned else ''}
        <p style="color: #495057;"><strong>Legal Certainty:</strong> {explanation.legal_analysis.legal_certainty.replace('_', ' ').title()}</p>
    </div>
"""
        
        # Add knowledge gaps for detailed explanations
        if explanation.explanation_level != ExplanationLevel.BASIC and explanation.knowledge_gaps.recommendations_for_clarification:
            html += """
    <div class="knowledge-gaps" style="background: #fff3cd; padding: 15px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #ffc107;">
        <h4 style="color: #856404; margin-top: 0;">💡 Additional Information</h4>
"""
            for rec in explanation.knowledge_gaps.recommendations_for_clarification:
                html += f'        <p style="color: #856404; margin: 5px 0;">• {rec}</p>'
            html += "    </div>"
        
        html += "</div>"
        return html
    
    def get_explanation_stats(self) -> Dict[str, Any]:
        """Get statistics about explanation generation."""
        return {
            'stats': self.explanation_stats,
            'cache_size': len(self.explanation_cache),
            'config': self.config
        }
    
    def save_explanation_history(self, file_path: str):
        """Save explanation history to file."""
        try:
            explanations_data = {
                'explanations': {
                    exp_id: asdict(explanation) 
                    for exp_id, explanation in self.explanation_cache.items()
                },
                'stats': self.explanation_stats,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(explanations_data, f, indent=2, default=str)
            
            self.logger.info(f"Explanation history saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save explanation history: {str(e)}")


class RAGPipelineWithExplanations:
    """
    Wrapper that adds explainable AI capabilities to the existing RAG pipeline.
    """
    
    def __init__(self, rag_pipeline, explanation_config: Optional[Dict] = None):
        """
        Initialize RAG pipeline with explanation capabilities.
        
        Args:
            rag_pipeline: Existing RAG pipeline instance
            explanation_config: Configuration for explanation features
        """
        self.rag_pipeline = rag_pipeline
        self.explainer = ExplainableAI(rag_pipeline, explanation_config)
        self.logger = logging.getLogger('RAGPipelineWithExplanations')
        
        # Track processing metadata for explanations
        self.current_processing_metadata = {}
        
        self.logger.info("RAG Pipeline with Explanations initialized")
    
    def process_query_with_explanation(self, query: str, 
                                     explanation_level: ExplanationLevel = ExplanationLevel.BASIC,
                                     include_explanation: bool = True) -> Dict[str, Any]:
        """
        Process query and generate explanation.
        
        Args:
            query: User's query
            explanation_level: Level of explanation detail
            include_explanation: Whether to include explanation in response
            
        Returns:
            Dictionary with response and optional explanation
        """
        start_time = time.time()
        
        try:
            # Prepare to capture processing metadata
            self._prepare_metadata_capture()
            
            # Process query through RAG pipeline
            response = self.rag_pipeline.process_query(query)
            
            # Capture final processing metadata
            processing_metadata = self._capture_processing_metadata(query, response, start_time)
            
            result = {
                'query': query,
                'response': response,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate explanation if requested
            if include_explanation:
                explanation = self.explainer.explain_response(
                    query, response, processing_metadata, explanation_level
                )
                result['explanation'] = self.explainer.format_explanation(explanation, "json")
                result['explanation_summary'] = self._create_explanation_summary(explanation)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in process_query_with_explanation: {str(e)}")
            return {
                'query': query,
                'response': f"Error processing query: {str(e)}",
                'error': str(e),
                'explanation': None if not include_explanation else self.explainer._create_error_explanation(query, "", str(e))
            }
    
    def _prepare_metadata_capture(self):
        """Prepare to capture processing metadata."""
        # Patch RAG pipeline methods to capture metadata
        # This is a simplified approach - in practice, you'd modify the RAG pipeline directly
        self.current_processing_metadata = {
            'cache_level': 'unknown',
            'retrieved_chunks': [],
            'processing_time': 0,
            'embedding_cache_hit': False,
            'retrieval_cache_hit': False,
            'total_chunks': getattr(self.rag_pipeline, 'pipeline_stats', {}).get('total_chunks', 0)
        }
    
    def _capture_processing_metadata(self, query: str, response: str, start_time: float) -> Dict[str, Any]:
        """Capture metadata from the processing pipeline."""
        processing_time = time.time() - start_time
        
        # Get stats from RAG pipeline
        if hasattr(self.rag_pipeline, 'get_pipeline_stats'):
            pipeline_stats = self.rag_pipeline.get_pipeline_stats()
        else:
            pipeline_stats = {}
        
        # Infer cache level from processing time and pipeline state
        cache_level = self._infer_cache_level(processing_time, pipeline_stats)
        
        metadata = {
            'cache_level': cache_level,
            'processing_time': processing_time,
            'pipeline_stats': pipeline_stats,
            'query_length': len(query),
            'response_length': len(response),
            'timestamp': time.time(),
            
            # Try to get retrieval information
            'retrieved_chunks': self._extract_retrieval_info(response),
            'embedding_cache_hit': processing_time < 0.05,  # Very fast suggests cache hit
            'retrieval_cache_hit': processing_time < 0.2,   # Fast suggests retrieval cache
            'total_chunks': pipeline_stats.get('total_chunks', 0),
            
            # Extract legal information
            'legal_sections': self._extract_legal_sections_from_response(response),
            'penalty_amounts': self._extract_penalty_amounts(response),
            'key_terms': self._extract_key_terms(query),
            'legal_topics': self._identify_legal_topics(query)
        }
        
        return metadata
    
    def _infer_cache_level(self, processing_time: float, pipeline_stats: Dict) -> str:
        """Infer which cache level was likely used based on processing time."""
        if processing_time < 0.01:
            return 'faq'
        elif processing_time < 0.05:
            return 'exact_query'
        elif processing_time < 0.1:
            return 'semantic'
        elif processing_time < 0.2:
            return 'precomputed'
        elif processing_time < 0.5:
            return 'embedding_cached'
        elif processing_time < 1.0:
            return 'retrieval_cached'
        else:
            return 'full_rag'
    
    def _extract_retrieval_info(self, response: str) -> List[Dict]:
        """Extract retrieval information from response patterns."""
        # This is a simplified version - in practice, you'd capture this during processing
        legal_sections = self._extract_legal_sections_from_response(response)
        
        if legal_sections:
            return [{
                'text': f"Content related to {', '.join(legal_sections)}",
                'similarity_score': 0.8,  # Estimated
                'metadata': {'legal_sections': legal_sections}
            }]
        
        return []
    
    def _extract_legal_sections_from_response(self, response: str) -> List[str]:
        """Extract legal section references from response."""
        pattern = r'Section\s+(\d+[A-Z]*)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        return [f"Section {match}" for match in matches]
    
    def _extract_penalty_amounts(self, response: str) -> List[str]:
        """Extract penalty amounts from response."""
        pattern = r'₹\s*(\d+(?:,\d+)*)'
        matches = re.findall(pattern, response)
        return [f"₹{match}" for match in matches]
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query."""
        important_terms = [
            'penalty', 'fine', 'license', 'helmet', 'registration', 'insurance',
            'drunk', 'speed', 'alcohol', 'certificate', 'documents', 'section',
            'violation', 'traffic', 'driving', 'vehicle'
        ]
        
        query_lower = query.lower()
        return [term for term in important_terms if term in query_lower]
    
    def _identify_legal_topics(self, query: str) -> List[str]:
        """Identify legal topics from query."""
        topic_keywords = {
            'licensing': ['license', 'permit', 'driving'],
            'traffic_violations': ['speed', 'signal', 'violation'],
            'safety': ['helmet', 'seatbelt', 'safety'],
            'documentation': ['documents', 'papers', 'certificate'],
            'penalties': ['penalty', 'fine', 'punishment']
        }
        
        identified_topics = []
        query_lower = query.lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                identified_topics.append(topic)
        
        return identified_topics
    
    def _create_explanation_summary(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """Create a brief summary of the explanation for quick viewing."""
        confidence = explanation.confidence_explanation
        cache_info = explanation.cache_explanation
        
        return {
            'confidence_level': confidence.confidence_category,
            'confidence_percentage': round(confidence.overall_confidence * 100, 1),
            'processing_method': cache_info.cache_level_used.value.replace('_', ' ').title(),
            'processing_time_ms': explanation.processing_time_ms,
            'legal_sections': explanation.legal_analysis.legal_sections_referenced,
            'penalty_amounts': explanation.legal_analysis.penalty_amounts_mentioned,
            'sources_used': len(explanation.source_attribution),
            'key_confidence_factors': explanation.confidence_explanation.factors_boosting_confidence[:2],
            'recommendations': explanation.knowledge_gaps.recommendations_for_clarification[:2]
        }
    
    def get_explanation_by_id(self, explanation_id: str) -> Optional[ExplanationResult]:
        """Retrieve a cached explanation by ID."""
        return self.explainer.explanation_cache.get(explanation_id)
    
    def export_explanations(self, file_path: str, format_type: str = "json"):
        """Export all explanations to file."""
        self.explainer.save_explanation_history(file_path)
    
    # Delegate other methods to the original RAG pipeline
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped RAG pipeline."""
        return getattr(self.rag_pipeline, name)


def integrate_explainable_ai(rag_pipeline, explanation_config: Optional[Dict] = None) -> RAGPipelineWithExplanations:
    """
    Easy integration function to add explainable AI to existing RAG pipeline.
    
    Args:
        rag_pipeline: Your existing RAG pipeline
        explanation_config: Optional configuration for explanation features
        
    Returns:
        Enhanced RAG pipeline with explanation capabilities
    """
    return RAGPipelineWithExplanations(rag_pipeline, explanation_config)


# Flask route integration helpers
def add_explanation_routes(app, explainable_rag_pipeline):
    """
    Add explanation-related routes to Flask app.
    
    Args:
        app: Flask application instance
        explainable_rag_pipeline: RAGPipelineWithExplanations instance
    """
    
    @app.route('/query/explained', methods=['POST'])
    def handle_explained_query():
        """Handle query with detailed explanation."""
        try:
            data = request.json
            query = data.get('query', '').strip()
            explanation_level = data.get('explanation_level', 'basic')
            
            if not query:
                return jsonify({'error': 'Query is required'}), 400
            
            # Convert string to enum
            try:
                exp_level = ExplanationLevel(explanation_level)
            except ValueError:
                exp_level = ExplanationLevel.BASIC
            
            # Process with explanation
            result = explainable_rag_pipeline.process_query_with_explanation(
                query, exp_level, include_explanation=True
            )
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/explanation/<explanation_id>', methods=['GET'])
    def get_explanation(explanation_id):
        """Get a specific explanation by ID."""
        try:
            explanation = explainable_rag_pipeline.get_explanation_by_id(explanation_id)
            
            if not explanation:
                return jsonify({'error': 'Explanation not found'}), 404
            
            # Get format from query parameter
            format_type = request.args.get('format', 'json')
            
            if format_type == 'json':
                return jsonify(explainable_rag_pipeline.explainer.format_explanation(explanation, 'json'))
            elif format_type == 'html':
                html_content = explainable_rag_pipeline.explainer.format_explanation(explanation, 'html')
                return html_content, 200, {'Content-Type': 'text/html'}
            elif format_type == 'markdown':
                md_content = explainable_rag_pipeline.explainer.format_explanation(explanation, 'markdown')
                return md_content, 200, {'Content-Type': 'text/markdown'}
            else:
                return jsonify({'error': f'Unsupported format: {format_type}'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/explanations/stats', methods=['GET'])
    def get_explanation_stats():
        """Get explanation generation statistics."""
        try:
            stats = explainable_rag_pipeline.explainer.get_explanation_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/explanations/export', methods=['POST'])
    def export_explanations():
        """Export explanation history."""
        try:
            data = request.json or {}
            format_type = data.get('format', 'json')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"data/exports/explanations_{timestamp}.json"
            
            # Ensure export directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            explainable_rag_pipeline.export_explanations(file_path, format_type)
            
            return jsonify({
                'message': 'Explanations exported successfully',
                'file_path': file_path,
                'timestamp': timestamp
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500


def main():
    """Test the explainable AI module."""
    print("🧪 Testing Explainable AI Module")
    print("=" * 50)
    
    # Mock RAG pipeline for testing
    class MockRAGPipeline:
        def __init__(self):
            self.pipeline_stats = {
                'total_chunks': 1000,
                'total_queries_processed': 50,
                'overall_cache_hit_rate': 75.5
            }
        
        def process_query(self, query):
            time.sleep(0.1)  # Simulate processing
            return "As per Section 181, driving without a valid license is punishable with a fine of ₹5,000."
        
        def get_pipeline_stats(self):
            return self.pipeline_stats
    
    # Test the explainable AI
    mock_rag = MockRAGPipeline()
    explainable_rag = integrate_explainable_ai(mock_rag)
    
    # Test query
    test_query = "What is the penalty for driving without a license?"
    
    print(f"Testing query: {test_query}")
    
    # Test different explanation levels
    for level in ExplanationLevel:
        print(f"\n--- {level.value.upper()} EXPLANATION ---")
        
        result = explainable_rag.process_query_with_explanation(
            test_query, level, include_explanation=True
        )
        
        print(f"Response: {result['response']}")
        print(f"Processing time: {result['processing_time_ms']}ms")
        
        if result.get('explanation_summary'):
            summary = result['explanation_summary']
            print(f"Confidence: {summary['confidence_percentage']}% ({summary['confidence_level']})")
            print(f"Method: {summary['processing_method']}")
            print(f"Legal sections: {', '.join(summary['legal_sections'])}")
    
    print(f"\n✅ Explainable AI module test completed!")


if __name__ == '__main__':
    main()