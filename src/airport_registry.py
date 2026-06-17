"""
Authoritative Airport Registry for Pakistan Travel RAG System.
Contains only verified Pakistani airports with correct classifications.
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Airport:
    """Airport information."""
    code: str
    name: str
    city: str
    province: str
    type: str  # "International" or "Domestic" 
    status: str  # "Active" or "Limited"

# Authoritative Pakistan Airport Registry
PAKISTAN_AIRPORTS: Dict[str, Airport] = {
    # International Airports (verified)
    "KHI": Airport("KHI", "Jinnah International Airport", "Karachi", "Sindh", "International", "Active"),
    "LHE": Airport("LHE", "Allama Iqbal International Airport", "Lahore", "Punjab", "International", "Active"),
    "ISB": Airport("ISB", "Islamabad International Airport", "Islamabad", "Islamabad Capital Territory", "International", "Active"),
    "PEW": Airport("PEW", "Peshawar International Airport", "Peshawar", "Khyber Pakhtunkhwa", "International", "Active"),
    "UET": Airport("UET", "Quetta International Airport", "Quetta", "Balochistan", "International", "Active"),
    "MUX": Airport("MUX", "Multan International Airport", "Multan", "Punjab", "International", "Active"),
    "CJL": Airport("CJL", "Chitral Airport", "Chitral", "Khyber Pakhtunkhwa", "Domestic", "Active"),
    
    # Major Domestic Airports
    "GIL": Airport("GIL", "Gilgit Airport", "Gilgit", "Gilgit Baltistan", "Domestic", "Active"),
    "SKD": Airport("SKD", "Skardu Airport", "Skardu", "Gilgit Baltistan", "Domestic", "Active"),
    "SWV": Airport("SWV", "Swat Airport", "Swat", "Khyber Pakhtunkhwa", "Domestic", "Limited"),
    "BWP": Airport("BWP", "Bahawalpur Airport", "Bahawalpur", "Punjab", "Domestic", "Active"),
    "RYK": Airport("RYK", "Rahim Yar Khan Airport", "Rahim Yar Khan", "Punjab", "Domestic", "Active"),
    "SUL": Airport("SUL", "Sui Airport", "Sui", "Balochistan", "Domestic", "Limited"),
}

# City to Airport Code mapping
CITY_AIRPORTS: Dict[str, str] = {
    "Karachi": "KHI",
    "Lahore": "LHE", 
    "Islamabad": "ISB",
    "Peshawar": "PEW",
    "Quetta": "UET",
    "Multan": "MUX",
    "Chitral": "CJL",
    "Gilgit": "GIL",
    "Skardu": "SKD",
    "Swat": "SWV",
    "Bahawalpur": "BWP",
    "Rahim Yar Khan": "RYK",
    "Sui": "SUL"
}

# Nearest airports for cities without direct airport access
NEAREST_AIRPORTS: Dict[str, str] = {
    "Hunza": "GIL",
    "Naran": "ISB", 
    "Kaghan": "ISB",
    "Murree": "ISB",
    "Fairy Meadows": "GIL",
    "Kumrat Valley": "PEW",
    "Shogran": "ISB",
    "Malam Jabba": "PEW",
    "Kalash Valley": "CJL",
    "Neelum Valley": "ISB",
    "Attabad Lake": "GIL",
    "Shigar": "SKD",
    "Khaplu": "SKD",
    "Deosai Plains": "SKD",
    "Naltar Valley": "GIL"
}

class AirportRegistry:
    """Airport registry service."""
    
    @staticmethod
    def get_airport_by_city(city: str) -> Airport:
        """Get airport information by city name."""
        code = CITY_AIRPORTS.get(city)
        if code:
            return PAKISTAN_AIRPORTS[code]
        return None
    
    @staticmethod 
    def get_nearest_airport(city: str) -> Airport:
        """Get nearest airport for cities without direct access."""
        nearest_city = NEAREST_AIRPORTS.get(city)
        if nearest_city:
            code = CITY_AIRPORTS.get(nearest_city)
            if code:
                return PAKISTAN_AIRPORTS[code]
        return None
    
    @staticmethod
    def get_international_airports() -> List[Airport]:
        """Get all international airports."""
        return [airport for airport in PAKISTAN_AIRPORTS.values() 
                if airport.type == "International"]
    
    @staticmethod
    def get_domestic_airports() -> List[Airport]:
        """Get all domestic airports.""" 
        return [airport for airport in PAKISTAN_AIRPORTS.values()
                if airport.type == "Domestic"]
    
    @staticmethod
    def is_international_airport(city: str) -> bool:
        """Check if city has international airport."""
        airport = AirportRegistry.get_airport_by_city(city)
        return airport and airport.type == "International"
    
    @staticmethod
    def get_flight_routing(departure: str, arrival: str) -> Dict[str, any]:
        """Get flight routing information between cities."""
        
        dep_airport = AirportRegistry.get_airport_by_city(departure)
        arr_airport = AirportRegistry.get_airport_by_city(arrival)
        
        # Check for nearest airports if direct not available
        if not dep_airport:
            dep_airport = AirportRegistry.get_nearest_airport(departure)
        if not arr_airport:
            arr_airport = AirportRegistry.get_nearest_airport(arrival)
        
        routing_info = {
            "departure_city": departure,
            "arrival_city": arrival,
            "departure_airport": dep_airport,
            "arrival_airport": arr_airport,
            "direct_flight_possible": bool(dep_airport and arr_airport),
            "road_segments": []
        }
        
        # Add road segments if nearest airports used
        if dep_airport and dep_airport.city != departure:
            routing_info["road_segments"].append({
                "from": departure,
                "to": dep_airport.city,
                "type": "Road to Airport"
            })
        
        if arr_airport and arr_airport.city != arrival:
            routing_info["road_segments"].append({
                "from": arr_airport.city, 
                "to": arrival,
                "type": "Road from Airport"
            })
        
        return routing_info

# Global registry instance
airport_registry = AirportRegistry()