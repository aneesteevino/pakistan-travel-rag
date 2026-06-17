"""
Enhanced RAG Pipeline for Pakistan Travel RAG System.
Implements: Intent Detection → Metadata Filtering → Retrieval → 
Reranking → Context Compression → Generation with Fallback Layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re

from src.retriever import get_retriever
from src.generator import generate_answer
from src.accommodation_service import accommodation_service
from src.transport_service import transport_service
from src.city_normalizer import normalize_city_name, is_valid_pakistan_city

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Detected query intents."""
    ACCOMMODATION_SEARCH = "accommodation_search"
    TRANSPORT_SEARCH = "transport_search"
    DESTINATION_COMPARISON = "destination_comparison"
    ITINERARY_PLANNING = "itinerary_planning"
    BUDGET_INQUIRY = "budget_inquiry"
    GENERAL_TRAVEL = "general_travel"
    WEATHER_INQUIRY = "weather_inquiry"
    ACTIVITY_SEARCH = "activity_search"

@dataclass
class RetrievalContext:
    """Structured retrieval context."""
    documents: List[Dict[str, Any]]
    confidence_score: float
    retrieval_method: str
    metadata_filters: Dict[str, Any]
    reranked: bool

@dataclass
class QueryAnalysis:
    """Query analysis result."""
    intent: QueryIntent
    entities: Dict[str, List[str]]
    confidence: float
    metadata_filters: Dict[str, Any]

class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with intent detection and fallback."""
    
    def __init__(self):
        self.retriever = get_retriever()
        self.confidence_threshold = 0.6
        
    def process_query(self, query: str, top_k: int = 6) -> Dict[str, Any]:
        """
        Process user query through enhanced RAG pipeline.
        
        Args:
            query: User's travel query
            top_k: Number of documents to retrieve
            
        Returns:
            Complete response with metadata
        """
        
        try:
            # Step 1: Intent Detection
            analysis = self._detect_intent(query)
            logger.info(f"Intent detected: {analysis.intent.value} (confidence: {analysis.confidence:.2f})")
            
            # Step 2: Entity-based Service Routing
            if analysis.intent in [QueryIntent.ACCOMMODATION_SEARCH, QueryIntent.TRANSPORT_SEARCH]:
                return self._handle_service_query(query, analysis)
            
            # Step 3: Metadata Filtering & Retrieval
            retrieval_context = self._retrieve_with_filters(query, analysis.metadata_filters, top_k)
            
            # Step 4: Confidence Check & Fallback Decision
            if retrieval_context.confidence_score < self.confidence_threshold:
                return self._handle_fallback(query, analysis, retrieval_context)
            
            # Step 5: Context Compression & Generation
            response = self._generate_response(query, retrieval_context, analysis)
            
            return {
                "response": response,
                "intent": analysis.intent.value,
                "confidence": retrieval_context.confidence_score,
                "source": "knowledge_base",
                "metadata": {
                    "documents_used": len(retrieval_context.documents),
                    "retrieval_method": retrieval_context.retrieval_method,
                    "filters_applied": retrieval_context.metadata_filters
                }
            }
            
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try rephrasing your question about Pakistan travel.",
                "intent": "error",
                "confidence": 0.0,
                "source": "error_fallback"
            }
    
    def _detect_intent(self, query: str) -> QueryAnalysis:
        """Detect query intent and extract entities."""
        
        query_lower = query.lower()
        
        # Intent patterns
        intent_patterns = {
            QueryIntent.ACCOMMODATION_SEARCH: [
                r'\b(hotel|accommodation|stay|lodge|guest house|room|booking)\b',
                r'\bwhere to stay\b',
                r'\bplace to sleep\b'
            ],
            QueryIntent.TRANSPORT_SEARCH: [
                r'\b(transport|travel|bus|flight|train|route|how to reach|how to get)\b',
                r'\btravel time\b',
                r'\bfare|ticket price\b'
            ],
            QueryIntent.DESTINATION_COMPARISON: [
                r'\b(compare|vs|versus|difference|better|choose between)\b',
                r'\bwhich is better\b',
                r'\bor\b.*\b(skardu|hunza|swat|karachi|lahore)\b'
            ],
            QueryIntent.BUDGET_INQUIRY: [
                r'\b(budget|cost|expense|price|how much|money|pkr|rupees)\b',
                r'\baffordable\b',
                r'\bcheap\b'
            ],
            QueryIntent.ITINERARY_PLANNING: [
                r'\b(itinerary|plan|schedule|day|trip plan|travel plan)\b',
                r'\bdays?\s+(in|at|to)\b',
                r'\bwhat to do\b'
            ],
            QueryIntent.WEATHER_INQUIRY: [
                r'\b(weather|climate|temperature|rain|snow|season)\b',
                r'\bbest time to visit\b',
                r'\bwhen to go\b'
            ],
            QueryIntent.ACTIVITY_SEARCH: [
                r'\b(activity|activities|things to do|attractions|sightseeing)\b',
                r'\bwhat to see\b',
                r'\bplaces to visit\b'
            ]
        }
        
        # Score intents
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            if score > 0:
                intent_scores[intent] = score / len(patterns)
        
        # Determine primary intent
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[primary_intent]
        else:
            primary_intent = QueryIntent.GENERAL_TRAVEL
            confidence = 0.5
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Build metadata filters
        metadata_filters = self._build_metadata_filters(entities, primary_intent)
        
        return QueryAnalysis(
            intent=primary_intent,
            entities=entities,
            confidence=confidence,
            metadata_filters=metadata_filters
        )
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract travel-related entities from query."""
        
        entities = {
            "cities": [],
            "provinces": [],
            "accommodation_types": [],
            "transport_modes": [],
            "activities": [],
            "durations": []
        }
        
        # City extraction
        pakistan_cities = [
            "karachi", "lahore", "islamabad", "peshawar", "quetta", "multan",
            "faisalabad", "rawalpindi", "hyderabad", "skardu", "gilgit", "hunza",
            "swat", "chitral", "murree", "naran", "kaghan", "fairy meadows"
        ]
        
        query_lower = query.lower()
        for city in pakistan_cities:
            if city in query_lower:
                entities["cities"].append(normalize_city_name(city))
        
        # Province extraction
        provinces = ["punjab", "sindh", "kpk", "balochistan", "gilgit baltistan", "azad kashmir"]
        for province in provinces:
            if province in query_lower:
                entities["provinces"].append(province.title())
        
        # Accommodation types
        if re.search(r'\bhotel\b', query_lower):
            entities["accommodation_types"].append("Hotel")
        if re.search(r'\bguest house\b', query_lower):
            entities["accommodation_types"].append("Guest House")
        
        # Transport modes
        if re.search(r'\b(flight|air|plane)\b', query_lower):
            entities["transport_modes"].append("Air")
        if re.search(r'\b(bus|road|drive|car)\b', query_lower):
            entities["transport_modes"].append("Road")
        
        # Duration extraction
        duration_match = re.search(r'(\d+)\s*(day|week|month)', query_lower)
        if duration_match:
            entities["durations"].append(duration_match.group(0))
        
        return entities
    
    def _build_metadata_filters(self, entities: Dict[str, List[str]], intent: QueryIntent) -> Dict[str, Any]:
        """Build metadata filters for retrieval."""
        
        filters = {}
        
        # City filtering
        if entities["cities"]:
            filters["cities"] = entities["cities"]
        
        # Intent-specific filtering
        if intent == QueryIntent.ACCOMMODATION_SEARCH:
            filters["dataset_types"] = ["hotels", "guest_house", "accommodation"]
            if entities["accommodation_types"]:
                filters["accommodation_types"] = entities["accommodation_types"]
        
        elif intent == QueryIntent.TRANSPORT_SEARCH:
            filters["dataset_types"] = ["flight_route", "road_transport", "transport"]
            if entities["transport_modes"]:
                filters["transport_modes"] = entities["transport_modes"]
        
        elif intent == QueryIntent.DESTINATION_COMPARISON:
            filters["dataset_types"] = ["destinations", "tourism", "activities"]
        
        elif intent == QueryIntent.BUDGET_INQUIRY:
            filters["dataset_types"] = ["tourism", "accommodation", "activities"]
        
        return filters
    
    def _retrieve_with_filters(self, query: str, filters: Dict[str, Any], top_k: int) -> RetrievalContext:
        """Retrieve documents with metadata filtering."""
        
        try:
            # Use existing retriever
            documents = self.retriever.retrieve(query, top_k * 2)  # Retrieve more for filtering
            
            # Apply metadata filters
            if filters:
                documents = self._apply_metadata_filters(documents, filters)
            
            # Limit to top_k after filtering
            documents = documents[:top_k]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(documents, query)
            
            return RetrievalContext(
                documents=documents,
                confidence_score=confidence,
                retrieval_method="vector_search_with_filters",
                metadata_filters=filters,
                reranked=False
            )
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return RetrievalContext(
                documents=[],
                confidence_score=0.0,
                retrieval_method="error",
                metadata_filters=filters,
                reranked=False
            )
    
    def _apply_metadata_filters(self, documents: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply metadata filters to retrieved documents."""
        
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            # Check dataset type filter
            if "dataset_types" in filters:
                doc_type = metadata.get("dataset_type", "")
                if not any(dtype in doc_type for dtype in filters["dataset_types"]):
                    continue
            
            # Check city filter
            if "cities" in filters:
                doc_city = metadata.get("city", "")
                if doc_city and not any(city.lower() in doc_city.lower() for city in filters["cities"]):
                    continue
            
            # Check accommodation type filter
            if "accommodation_types" in filters:
                doc_type = metadata.get("type", "")
                if doc_type and doc_type not in filters["accommodation_types"]:
                    continue
            
            filtered_docs.append(doc)
        
        return filtered_docs
    
    def _calculate_confidence(self, documents: List[Dict], query: str) -> float:
        """Calculate retrieval confidence score."""
        
        if not documents:
            return 0.0
        
        # Use similarity scores if available
        scores = [doc.get("score", 0.5) for doc in documents]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # Boost confidence if we have multiple relevant documents
        document_bonus = min(0.2, len(documents) * 0.05)
        
        return min(1.0, avg_score + document_bonus)
    
    def _handle_service_query(self, query: str, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Handle queries that should use dedicated services."""
        
        cities = analysis.entities["cities"]
        
        if analysis.intent == QueryIntent.ACCOMMODATION_SEARCH and len(cities) >= 1:
            city = cities[0]
            acc_type = analysis.entities["accommodation_types"][0] if analysis.entities["accommodation_types"] else None
            
            accommodations = accommodation_service.search_accommodations(city, acc_type)
            
            if accommodations:
                response = self._format_accommodation_response(accommodations, city, acc_type)
                return {
                    "response": response,
                    "intent": analysis.intent.value,
                    "confidence": 1.0,
                    "source": "accommodation_service",
                    "metadata": {"accommodations_found": len(accommodations)}
                }
        
        elif analysis.intent == QueryIntent.TRANSPORT_SEARCH and len(cities) >= 2:
            departure, arrival = cities[0], cities[1]
            
            comparison = transport_service.compare_transport(departure, arrival)
            response = self._format_transport_response(comparison, departure, arrival)
            
            return {
                "response": response,
                "intent": analysis.intent.value,
                "confidence": 1.0,
                "source": "transport_service",
                "metadata": {"route_available": comparison.get("available", False)}
            }
        
        # Fall back to RAG if service query doesn't match expected pattern
        retrieval_context = self._retrieve_with_filters(query, analysis.metadata_filters, 6)
        return self._generate_response(query, retrieval_context, analysis)
    
    def _handle_fallback(self, query: str, analysis: QueryAnalysis, 
                        retrieval_context: RetrievalContext) -> Dict[str, Any]:
        """Handle queries with low confidence using LLM fallback."""
        
        # Generate fallback response using LLM general knowledge
        fallback_prompt = f"""
Based on general travel knowledge about Pakistan, please provide helpful information for: {query}

Important guidelines:
- Clearly indicate this is based on general knowledge, not specific database information
- Focus on Pakistan travel destinations, culture, and practical advice
- Suggest contacting local tourism offices for current information
- Be helpful but acknowledge limitations

Query: {query}
"""
        
        response = generate_answer(fallback_prompt, [])
        
        # Add fallback disclaimer
        disclaimer = "\n\n📝 *Note: This response is based on general travel knowledge. For the most current information, please verify with local tourism authorities or service providers.*"
        
        return {
            "response": response + disclaimer,
            "intent": analysis.intent.value,
            "confidence": retrieval_context.confidence_score,
            "source": "llm_fallback",
            "metadata": {
                "fallback_reason": "low_confidence",
                "original_confidence": retrieval_context.confidence_score
            }
        }
    
    def _generate_response(self, query: str, context: RetrievalContext, analysis: QueryAnalysis) -> str:
        """Generate response using retrieved context."""
        
        # Prepare context for generation
        context_text = "\n".join([doc.get("content", "") for doc in context.documents])
        
        # Generate response
        response = generate_answer(query, context.documents)
        
        # Add source attribution (without exposing filenames)
        if context.documents:
            response += "\n\n📊 *Response based on available travel records in our Pakistan tourism database.*"
        
        return response
    
    def _format_accommodation_response(self, accommodations: List, city: str, acc_type: Optional[str]) -> str:
        """Format accommodation search response."""
        
        type_text = f" {acc_type.lower()}s" if acc_type else " accommodations"
        
        response = f"🏨 **{type_text.title()} in {city}**\n\n"
        
        for i, acc in enumerate(accommodations[:5], 1):  # Limit to 5 results
            response += f"**{i}. {acc.name}**\n"
            response += f"   • Type: {acc.type}\n"
            response += f"   • Price Range: {acc.price_range}\n"
            response += f"   • Rating: {acc.rating}/5.0\n"
            if acc.contact.get("phone"):
                response += f"   • Phone: {acc.contact['phone']}\n"
            response += "\n"
        
        response += "📊 *Information sourced from accommodation database.*"
        return response
    
    def _format_transport_response(self, comparison: Dict, departure: str, arrival: str) -> str:
        """Format transport comparison response."""
        
        if not comparison.get("available"):
            return f"🚫 {comparison.get('message', 'No transport information available.')}"
        
        response = f"🚗 **Transport Options: {departure} → {arrival}**\n\n"
        
        air_info = comparison["options"].get("air", {})
        road_info = comparison["options"].get("road", {})
        
        if air_info.get("available"):
            response += "✈️ **Air Travel**\n"
            response += f"   • Fare Range: {air_info['fare_range']}\n"
            response += f"   • Duration: {air_info['avg_duration']}\n"
            response += f"   • Airlines: {', '.join(air_info['operators'])}\n\n"
        
        if road_info.get("available"):
            response += "🚌 **Road Transport**\n"
            response += f"   • Fare Range: {road_info['fare_range']}\n" 
            response += f"   • Duration: {road_info['avg_duration']}\n"
            response += f"   • Operators: {', '.join(road_info['operators'])}\n\n"
        
        response += f"💡 **Recommendation**: {comparison['recommendation']}\n\n"
        response += "📊 *Transport information from operator databases.*"
        
        return response

# Global pipeline instance
rag_pipeline = EnhancedRAGPipeline()