"""
LLM Fallback Layer for Pakistan Travel RAG System.
Provides general travel knowledge when retrieval confidence is insufficient.
"""

import logging
from typing import Dict, Any, Optional
from src.config import GROQ_API_KEY, GEMINI_API_KEY, GROQ_MODEL, GEMINI_MODEL

logger = logging.getLogger(__name__)

class LLMFallbackService:
    """Service for LLM-based fallback responses when RAG confidence is low."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
    def should_use_fallback(self, confidence_score: float, retrieved_docs: int = 0) -> bool:
        """Determine if fallback should be used based on confidence and document count."""
        return (confidence_score < self.confidence_threshold or 
                retrieved_docs == 0)
    
    def generate_fallback_response(self, query: str, context_info: Optional[Dict[str, Any]] = None) -> str:
        """Generate fallback response using LLM general knowledge."""
        
        # Enhanced fallback prompt with Pakistan focus
        fallback_prompt = f"""You are a Pakistan travel expert assistant. A user asked: "{query}"

The local knowledge base has insufficient specific information for this query.

Please provide helpful general information about Pakistan travel while:

1. **Clearly indicating** this is based on general travel knowledge, not specific database records
2. **Focusing on Pakistan** destinations, culture, and practical travel advice  
3. **Using Pakistani Rupees (PKR)** for any cost estimates
4. **Suggesting verification** with local tourism authorities for current information
5. **Being helpful and informative** while acknowledging limitations
6. **Mentioning reliable sources** like PTDC, local tourism offices, or official websites

For accommodation questions: Focus on Hotels and Guest Houses (our system's supported types)
For transport questions: Include both air and road options with major Pakistani operators
For budget questions: Provide realistic PKR estimates based on general Pakistan travel costs
For destination questions: Highlight Pakistan's natural beauty, culture, and unique experiences

Be encouraging about Pakistan travel while being honest about information limitations.

Response:"""

        try:
            response = self._call_llm(fallback_prompt)
            
            # Add fallback disclaimer
            disclaimer = "\n\n📝 *Note: This response is based on general travel knowledge about Pakistan. For the most current and specific information, please verify with:*\n" \
                        "• **Pakistan Tourism Development Corporation (PTDC)**: ptdc.gov.pk\n" \
                        "• **Local tourism offices** in Pakistani cities\n" \
                        "• **Hotels and transport operators** directly"
            
            return response + disclaimer
            
        except Exception as e:
            logger.error(f"LLM fallback generation failed: {e}")
            return self._generate_default_fallback(query)
    
    def _call_llm(self, prompt: str) -> str:
        """Call available LLM service."""
        
        # Try Groq first (primary provider)
        if GROQ_API_KEY:
            try:
                from groq import Groq
                client = Groq(api_key=GROQ_API_KEY)
                
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.4,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"Groq fallback failed: {e}")
        
        # Try Gemini as backup
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(GEMINI_MODEL)
                
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=600,
                        temperature=0.4,
                    )
                )
                
                return response.text.strip()
                
            except Exception as e:
                logger.warning(f"Gemini fallback failed: {e}")
        
        # If both fail, raise exception to trigger default fallback
        raise Exception("No LLM service available for fallback")
    
    def _generate_default_fallback(self, query: str) -> str:
        """Generate default fallback when LLM services fail."""
        
        # Categorize query type for better default responses
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['hotel', 'accommodation', 'stay', 'lodge']):
            return self._accommodation_fallback(query)
        elif any(word in query_lower for word in ['transport', 'travel', 'bus', 'flight', 'route']):
            return self._transport_fallback(query)
        elif any(word in query_lower for word in ['budget', 'cost', 'price', 'money']):
            return self._budget_fallback(query)
        elif any(word in query_lower for word in ['destination', 'place', 'visit', 'tourism']):
            return self._destination_fallback(query)
        else:
            return self._general_fallback(query)
    
    def _accommodation_fallback(self, query: str) -> str:
        """Default accommodation fallback."""
        return f"""I don't have specific accommodation information for "{query}" in our current database.

**For Hotels & Guest Houses in Pakistan:**
• **Major hotel chains**: Pearl Continental, Serena Hotels, Marriott (in major cities)
• **Guest houses**: Available in most tourist destinations, typically PKR 3,000-8,000/night
• **Budget range**: PKR 2,000-15,000+ depending on location and amenities

**Recommended booking sources:**
• Contact hotels directly via phone
• Visit local tourism offices
• Check with travel agencies in major cities

📝 *Our database focuses on Hotels and Guest Houses. For current availability and rates, please contact accommodations directly.*"""

    def _transport_fallback(self, query: str) -> str:
        """Default transport fallback."""
        return f"""I don't have specific transport information for "{query}" in our current database.

**Pakistan Transport Options:**
• **Domestic flights**: PIA, Airblue, AirSial, Fly Jinnah, Serene Air
• **Road transport**: Daewoo Express, NATCO, Bilal Travels, local operators
• **Typical costs**: Flights PKR 8,000-25,000, buses PKR 1,500-5,000 depending on route

**For current schedules and booking:**
• Airlines: Check official websites or visit airport counters
• Road transport: Visit bus terminals or contact operators directly
• Local transport: Available in all major cities

📝 *Transport schedules and prices change frequently. Please verify current information with operators.*"""

    def _budget_fallback(self, query: str) -> str:
        """Default budget fallback."""
        return f"""I don't have specific budget information for "{query}" in our current database.

**General Pakistan Travel Budget (per person/day):**
• **Budget travel**: PKR 2,000-4,000 (basic accommodation, local food, public transport)
• **Mid-range travel**: PKR 5,000-12,000 (decent hotels, mixed dining, private transport)
• **Luxury travel**: PKR 15,000+ (premium accommodation, fine dining, guided tours)

**Cost factors:**
• Accommodation: PKR 1,500-15,000+ per night
• Food: PKR 500-3,000+ per day
• Transport: varies by distance and mode

📝 *Costs vary significantly by destination and season. Contact local tourism offices for current pricing.*"""

    def _destination_fallback(self, query: str) -> str:
        """Default destination fallback."""
        return f"""I don't have specific destination information for "{query}" in our current database.

**Popular Pakistan destinations:**
• **Northern Areas**: Hunza, Skardu, Swat, Chitral, Gilgit
• **Punjab**: Lahore, Islamabad, Murree, Multan
• **Sindh**: Karachi, Hyderabad, Thatta
• **KPK**: Peshawar, Kumrat Valley, Malam Jabba
• **Balochistan**: Quetta, Gwadar

**Best travel seasons:**
• Mountains: April-October
• Plains: October-March
• Coastal: November-February

📝 *For detailed destination information, contact Pakistan Tourism Development Corporation (PTDC) at ptdc.gov.pk*"""

    def _general_fallback(self, query: str) -> str:
        """Default general fallback."""
        return f"""I don't have specific information about "{query}" in our Pakistan travel database.

**For Pakistan travel assistance:**
• **Pakistan Tourism Development Corporation (PTDC)**: ptdc.gov.pk
• **Provincial tourism departments**: Available in each province
• **Local tourism offices**: Found in major cities and tourist areas
• **Travel agencies**: Specialized Pakistan tour operators

**Popular resources:**
• Official tourism websites
• Local guidebooks
• Hotel concierge services
• Tourist information centers

📝 *Our database is focused on Pakistan accommodations and destinations. For specialized queries, local experts and official sources are your best resource.*"""

# Global fallback service instance
llm_fallback = LLMFallbackService()