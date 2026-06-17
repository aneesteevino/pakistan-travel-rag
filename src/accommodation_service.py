"""
Accommodation Service for Pakistan Travel RAG System.
Handles Hotel and Guest House search, comparison, and recommendations.
Restricted to data/new data/Hotels data/ and data/new data/Guest houses data/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from src.config import DATA_DIR
from src.city_normalizer import normalize_city_name

logger = logging.getLogger(__name__)

# Data paths
HOTELS_DATA_PATH = DATA_DIR / "new data" / "Hotels data" / "hotels.csv"
GUEST_HOUSES_DATA_PATH = DATA_DIR / "new data" / "Guest houses data" / "guest_houses.csv"

@dataclass
class Accommodation:
    """Standardized accommodation representation."""
    name: str
    type: str  # "Hotel" or "Guest House"
    city: str
    province: str
    price_range: str
    rating: float
    amenities: List[str]
    contact: Dict[str, str]
    address: str
    description: str
    source: str

class AccommodationService:
    """Service for accommodation search and comparison."""
    
    def __init__(self):
        self.hotels_df = None
        self.guest_houses_df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load accommodation datasets."""
        try:
            if HOTELS_DATA_PATH.exists():
                self.hotels_df = pd.read_csv(HOTELS_DATA_PATH)
                self.hotels_df['city'] = self.hotels_df['city'].apply(normalize_city_name)
                logger.info(f"Loaded {len(self.hotels_df)} hotels")
            else:
                logger.warning(f"Hotels data not found: {HOTELS_DATA_PATH}")
                
            if GUEST_HOUSES_DATA_PATH.exists():
                self.guest_houses_df = pd.read_csv(GUEST_HOUSES_DATA_PATH)  
                self.guest_houses_df['city'] = self.guest_houses_df['city'].apply(normalize_city_name)
                logger.info(f"Loaded {len(self.guest_houses_df)} guest houses")
            else:
                logger.warning(f"Guest houses data not found: {GUEST_HOUSES_DATA_PATH}")
                
        except Exception as e:
            logger.error(f"Error loading accommodation data: {e}")
    
    def search_accommodations(self, city: str, accommodation_type: Optional[str] = None) -> List[Accommodation]:
        """
        Search accommodations by city and optionally by type.
        
        Args:
            city: Target city name
            accommodation_type: "Hotel", "Guest House", or None for both
            
        Returns:
            List of matching accommodations
        """
        city = normalize_city_name(city)
        results = []
        
        # Search hotels
        if accommodation_type != "Guest House" and self.hotels_df is not None:
            hotel_matches = self.hotels_df[
                self.hotels_df['city'].str.lower() == city.lower()
            ]
            
            for _, row in hotel_matches.iterrows():
                results.append(Accommodation(
                    name=str(row.get('name', '')),
                    type="Hotel",
                    city=str(row.get('city', '')),
                    province=str(row.get('province', '')),
                    price_range=str(row.get('price_range', 'PKR 5,000-15,000')),
                    rating=float(row.get('rating', 0.0)),
                    amenities=str(row.get('amenities', '')).split(','),
                    contact={
                        'phone': str(row.get('phone', '')),
                        'email': str(row.get('email', ''))
                    },
                    address=str(row.get('address', '')),
                    description=str(row.get('description', '')),
                    source="hotels.csv"
                ))
        
        # Search guest houses
        if accommodation_type != "Hotel" and self.guest_houses_df is not None:
            gh_matches = self.guest_houses_df[
                self.guest_houses_df['city'].str.lower() == city.lower()
            ]
            
            for _, row in gh_matches.iterrows():
                results.append(Accommodation(
                    name=str(row.get('name', '')),
                    type="Guest House", 
                    city=str(row.get('city', '')),
                    province=str(row.get('province', '')),
                    price_range=str(row.get('price_range', 'PKR 3,000-8,000')),
                    rating=float(row.get('rating', 0.0)),
                    amenities=str(row.get('amenities', '')).split(','),
                    contact={
                        'phone': str(row.get('phone', '')),
                        'email': str(row.get('email', ''))
                    },
                    address=str(row.get('address', '')),
                    description=str(row.get('description', '')),
                    source="guest_houses.csv"
                ))
        
        return results
    
    def compare_accommodations(self, city1: str, city2: str) -> Dict[str, Any]:
        """
        Compare accommodations between two cities.
        
        Returns:
            Comparison data with recommendations
        """
        city1_accoms = self.search_accommodations(city1)
        city2_accoms = self.search_accommodations(city2)
        
        def analyze_city(accommodations: List[Accommodation], city_name: str) -> Dict[str, Any]:
            if not accommodations:
                return {
                    "city": city_name,
                    "available": False,
                    "message": f"No accommodations available in our database for {city_name}"
                }
            
            hotels = [a for a in accommodations if a.type == "Hotel"]
            guest_houses = [a for a in accommodations if a.type == "Guest House"]
            
            avg_rating = sum(a.rating for a in accommodations) / len(accommodations) if accommodations else 0
            
            return {
                "city": city_name,
                "available": True,
                "total_count": len(accommodations),
                "hotels_count": len(hotels),
                "guest_houses_count": len(guest_houses),
                "avg_rating": round(avg_rating, 1),
                "top_recommendation": accommodations[0] if accommodations else None,
                "budget_option": min(accommodations, key=lambda x: x.rating) if accommodations else None,
                "luxury_option": max(accommodations, key=lambda x: x.rating) if accommodations else None
            }
        
        return {
            "comparison": {
                city1: analyze_city(city1_accoms, city1),
                city2: analyze_city(city2_accoms, city2)
            },
            "recommendation": self._generate_recommendation(city1_accoms, city2_accoms, city1, city2)
        }
    
    def _generate_recommendation(self, city1_accoms: List[Accommodation], 
                               city2_accoms: List[Accommodation], city1: str, city2: str) -> str:
        """Generate accommodation recommendation between cities."""
        
        if not city1_accoms and not city2_accoms:
            return f"We currently don't have accommodation data for {city1} or {city2}. Please check with local tourism offices."
        
        if not city1_accoms:
            return f"Based on available data, {city2} has more accommodation options in our database."
        
        if not city2_accoms:
            return f"Based on available data, {city1} has more accommodation options in our database."
        
        city1_avg = sum(a.rating for a in city1_accoms) / len(city1_accoms)
        city2_avg = sum(a.rating for a in city2_accoms) / len(city2_accoms)
        
        if city1_avg > city2_avg:
            return f"For accommodation quality, {city1} shows higher average ratings ({city1_avg:.1f}/5.0 vs {city2_avg:.1f}/5.0)."
        elif city2_avg > city1_avg:
            return f"For accommodation quality, {city2} shows higher average ratings ({city2_avg:.1f}/5.0 vs {city1_avg:.1f}/5.0)."
        else:
            return f"Both {city1} and {city2} have similar accommodation quality ratings in our database."
    
    def get_available_cities(self) -> List[str]:
        """Get list of cities with available accommodations."""
        cities = set()
        
        if self.hotels_df is not None:
            cities.update(self.hotels_df['city'].dropna().unique())
            
        if self.guest_houses_df is not None:
            cities.update(self.guest_houses_df['city'].dropna().unique())
        
        return sorted(list(cities))

# Global service instance
accommodation_service = AccommodationService()