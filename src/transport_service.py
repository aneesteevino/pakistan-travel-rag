"""
Transport Service for Pakistan Travel RAG System.
Handles route comparison, fare comparison, and transport recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

from src.city_normalizer import normalize_city_name
from src.flights import search_flights, AIRPORTS, AIRLINES

logger = logging.getLogger(__name__)

class TransportMode(Enum):
    """Available transport modes."""
    AIR = "Air"
    ROAD = "Road"
    RAIL = "Rail"

@dataclass 
class TransportOption:
    """Standardized transport option representation."""
    mode: TransportMode
    operator: str
    departure_city: str
    arrival_city: str
    fare_pkr: int
    duration_hours: float
    contact: str
    availability: str
    notes: str

class TransportService:
    """Service for transport search and comparison."""
    
    def __init__(self):
        self.flight_data = {}
        self.road_operators = self._load_road_operators()
        
    def _load_road_operators(self) -> List[Dict[str, Any]]:
        """Load available road transport operators."""
        # Standard Pakistan transport operators
        return [
            {
                "name": "Daewoo Express",
                "routes": ["Karachi-Lahore", "Lahore-Islamabad", "Karachi-Multan"],
                "fare_range": "2500-4000",
                "contact": "0800-DAEWOO"
            },
            {
                "name": "Bilal Travels", 
                "routes": ["Lahore-Skardu", "Islamabad-Gilgit", "Rawalpindi-Hunza"],
                "fare_range": "3000-5000",
                "contact": "051-123-4567"
            },
            {
                "name": "NATCO",
                "routes": ["Karachi-Quetta", "Lahore-Peshawar", "Islamabad-Chitral"], 
                "fare_range": "2000-4500",
                "contact": "021-111-NATCO"
            }
        ]
    
    def search_transport(self, departure: str, arrival: str, mode: Optional[TransportMode] = None) -> List[TransportOption]:
        """
        Search transport options between two cities.
        
        Args:
            departure: Departure city
            arrival: Arrival city
            mode: Specific transport mode or None for all
            
        Returns:
            List of available transport options
        """
        departure = normalize_city_name(departure)
        arrival = normalize_city_name(arrival)
        options = []
        
        # Search flights if air mode requested or no mode specified
        if mode in [TransportMode.AIR, None]:
            flight_options = self._search_flights(departure, arrival)
            options.extend(flight_options)
        
        # Search road transport if road mode requested or no mode specified  
        if mode in [TransportMode.ROAD, None]:
            road_options = self._search_road_transport(departure, arrival)
            options.extend(road_options)
            
        return options
    
    def _search_flights(self, departure: str, arrival: str) -> List[TransportOption]:
        """Search flight options."""
        options = []
        
        # Check if cities have airports
        dep_airport = AIRPORTS.get(departure)
        arr_airport = AIRPORTS.get(arrival)
        
        if not dep_airport or not arr_airport:
            # Provide fallback information
            return [TransportOption(
                mode=TransportMode.AIR,
                operator="Multiple Airlines",
                departure_city=departure,
                arrival_city=arrival,
                fare_pkr=0,
                duration_hours=0,
                contact="See airline websites",
                availability="Flight schedules change frequently",
                notes=f"No direct airport data available. Check with PIA, Airblue, AirSial, Fly Jinnah, or Serene Air for current schedules and pricing."
            )]
        
        # Use flight search service
        flight_result = search_flights(departure, arrival)
        
        if flight_result.get("found"):
            for airline in AIRLINES:
                options.append(TransportOption(
                    mode=TransportMode.AIR,
                    operator=airline,
                    departure_city=departure,
                    arrival_city=arrival,
                    fare_pkr=int(flight_result["avg_price_usd"] * 280),
                    duration_hours=flight_result["avg_duration_hours"],
                    contact=f"Contact {airline} directly",
                    availability="Daily flights available",
                    notes="Prices and schedules subject to change"
                ))
        
        return options
    
    def _search_road_transport(self, departure: str, arrival: str) -> List[TransportOption]:
        """Search road transport options."""
        options = []
        
        # Check each operator for route coverage
        for operator in self.road_operators:
            route_key = f"{departure}-{arrival}"
            reverse_route = f"{arrival}-{departure}"
            
            if any(route_key in route or reverse_route in route for route in operator["routes"]):
                fare_range = operator["fare_range"].split("-")
                avg_fare = (int(fare_range[0]) + int(fare_range[1])) // 2
                
                # Estimate duration based on distance (rough approximation)
                duration = self._estimate_road_duration(departure, arrival)
                
                options.append(TransportOption(
                    mode=TransportMode.ROAD,
                    operator=operator["name"],
                    departure_city=departure,
                    arrival_city=arrival,
                    fare_pkr=avg_fare,
                    duration_hours=duration,
                    contact=operator["contact"],
                    availability="Daily service",
                    notes="Schedule and fare subject to operator confirmation"
                ))
        
        # If no specific operators found, provide general information
        if not options:
            options.append(TransportOption(
                mode=TransportMode.ROAD,
                operator="Local Transport Services",
                departure_city=departure,
                arrival_city=arrival, 
                fare_pkr=2500,
                duration_hours=self._estimate_road_duration(departure, arrival),
                contact="Local bus stations",
                availability="Multiple daily services",
                notes="Contact local transport operators for current schedules and pricing"
            ))
        
        return options
    
    def _estimate_road_duration(self, departure: str, arrival: str) -> float:
        """Estimate road travel duration between cities."""
        
        # Rough duration estimates for major routes (hours)
        duration_map = {
            ("Karachi", "Lahore"): 20,
            ("Lahore", "Islamabad"): 5,
            ("Islamabad", "Gilgit"): 18,
            ("Gilgit", "Skardu"): 6,
            ("Lahore", "Peshawar"): 6,
            ("Karachi", "Quetta"): 12,
            ("Islamabad", "Murree"): 2,
            ("Lahore", "Multan"): 6,
            ("Islamabad", "Swat"): 5
        }
        
        # Check both directions
        key1 = (departure, arrival)
        key2 = (arrival, departure) 
        
        if key1 in duration_map:
            return duration_map[key1]
        elif key2 in duration_map:
            return duration_map[key2]
        else:
            # Default estimate based on city importance
            return 8.0
    
    def compare_transport(self, departure: str, arrival: str) -> Dict[str, Any]:
        """
        Compare all available transport options between two cities.
        
        Returns:
            Comprehensive comparison with recommendations
        """
        all_options = self.search_transport(departure, arrival)
        
        if not all_options:
            return {
                "available": False,
                "message": f"We currently do not have route information for {departure} to {arrival}. You can check the latest schedules directly from available transport operators.",
                "operators": {
                    "airlines": ["PIA", "Airblue", "AirSial", "Fly Jinnah", "Serene Air"],
                    "road_transport": [op["name"] for op in self.road_operators]
                }
            }
        
        # Group by transport mode
        air_options = [opt for opt in all_options if opt.mode == TransportMode.AIR]
        road_options = [opt for opt in all_options if opt.mode == TransportMode.ROAD]
        
        # Generate comparison
        comparison = {
            "available": True,
            "route": f"{departure} → {arrival}",
            "options": {
                "air": self._analyze_options(air_options),
                "road": self._analyze_options(road_options)
            },
            "recommendation": self._generate_transport_recommendation(air_options, road_options)
        }
        
        return comparison
    
    def _analyze_options(self, options: List[TransportOption]) -> Dict[str, Any]:
        """Analyze transport options and provide summary."""
        if not options:
            return {"available": False}
        
        fares = [opt.fare_pkr for opt in options if opt.fare_pkr > 0]
        durations = [opt.duration_hours for opt in options if opt.duration_hours > 0]
        
        return {
            "available": True,
            "count": len(options),
            "fare_range": f"PKR {min(fares)}-{max(fares)}" if fares else "Contact operator",
            "avg_duration": f"{sum(durations)/len(durations):.1f} hours" if durations else "Contact operator",
            "operators": [opt.operator for opt in options],
            "fastest_option": min(options, key=lambda x: x.duration_hours) if durations else None,
            "cheapest_option": min(options, key=lambda x: x.fare_pkr) if fares else None
        }
    
    def _generate_transport_recommendation(self, air_options: List[TransportOption], 
                                         road_options: List[TransportOption]) -> str:
        """Generate transport recommendation."""
        
        if not air_options and not road_options:
            return "Contact local transport operators for the most current route information."
        
        if air_options and not road_options:
            return "Air travel is the primary option for this route. Check with airlines for current schedules."
        
        if road_options and not air_options:
            return "Road transport is the main option for this route. Multiple operators serve this connection."
        
        # Both options available
        air_duration = min(opt.duration_hours for opt in air_options if opt.duration_hours > 0)
        road_duration = min(opt.duration_hours for opt in road_options if opt.duration_hours > 0)
        
        if air_duration < road_duration * 0.3:
            return "Air travel is significantly faster for this route, though road transport offers scenic journey options."
        else:
            return "Both air and road transport are viable. Choose based on budget, schedule, and travel preference."

# Global service instance
transport_service = TransportService()