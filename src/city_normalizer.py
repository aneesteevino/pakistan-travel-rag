"""
City Name Normalization Service.
Standardizes city names across the Pakistan Travel RAG System.
"""

import re
from typing import Dict

# City normalization mappings
CITY_MAPPINGS: Dict[str, str] = {
    # Common variations
    "islamabad city": "Islamabad",
    "lahore city": "Lahore", 
    "karachi city": "Karachi",
    "rawalpindi city": "Rawalpindi",
    
    # Valley and region mappings
    "naran kaghan": "Naran",
    "kaghan valley": "Kaghan",
    "fairy meadows base camp": "Fairy Meadows",
    "karimabad village": "Hunza",
    "altit village": "Hunza", 
    "passu village": "Hunza",
    "attabad lake shore": "Hunza",
    
    # City bazaar mappings
    "skardu bazaar": "Skardu",
    "mingora city": "Swat",
    "kalam valley": "Swat",
    "chitral bazaar": "Chitral",
    "bumburet valley": "Chitral",
    
    # Regional variations
    "shigar village": "Shigar",
    "khaplu valley": "Khaplu", 
    "nagar valley": "Nagar",
    "gilgit city": "Gilgit",
    "astore valley": "Astore",
    "minimarg village": "Minimarg",
    
    # Common misspellings
    "islamabd": "Islamabad",
    "lahor": "Lahore",
    "karachi": "Karachi",
    "peshawar": "Peshawar",
    "queta": "Quetta",
    "multan": "Multan"
}

def normalize_city_name(city: str) -> str:
    """
    Normalize city name for consistent matching.
    
    Args:
        city: Raw city name from dataset
        
    Returns:
        Normalized city name
    """
    if not city or not isinstance(city, str):
        return ""
    
    # Clean input
    city = city.strip()
    if not city:
        return ""
    
    # Remove common prefixes/suffixes
    city = re.sub(r'\s*,\s*Pakistan$', '', city, flags=re.IGNORECASE)
    city = re.sub(r'\s*,\s*PK$', '', city, flags=re.IGNORECASE) 
    city = re.sub(r'\s+District$', '', city, flags=re.IGNORECASE)
    city = re.sub(r'\s+City$', '', city, flags=re.IGNORECASE)
    
    # Convert to lowercase for lookup
    city_lower = city.lower().strip()
    
    # Check mappings
    if city_lower in CITY_MAPPINGS:
        return CITY_MAPPINGS[city_lower]
    
    # Extract main city name from compound names
    if "," in city:
        city = city.split(",")[0].strip()
    
    # Title case the result
    return city.title()

def get_city_aliases(city: str) -> list[str]:
    """
    Get all known aliases for a city.
    
    Args:
        city: Normalized city name
        
    Returns:
        List of aliases including the original
    """
    normalized = normalize_city_name(city)
    aliases = [normalized]
    
    # Find reverse mappings
    for alias, canonical in CITY_MAPPINGS.items():
        if canonical.lower() == normalized.lower():
            aliases.append(alias.title())
    
    return list(set(aliases))

def is_valid_pakistan_city(city: str) -> bool:
    """
    Check if city name is a recognized Pakistan city.
    
    Args:
        city: City name to validate
        
    Returns:
        True if recognized Pakistan city
    """
    normalized = normalize_city_name(city)
    
    # Major Pakistan cities
    major_cities = {
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", 
        "Quetta", "Multan", "Faisalabad", "Hyderabad", "Sialkot",
        "Gujranwala", "Skardu", "Gilgit", "Hunza", "Swat", "Murree",
        "Abbottabad", "Mansehra", "Chitral", "Bahawalpur", "Sukkur",
        "Larkana", "Mirpur", "Muzaffarabad", "Naran", "Kaghan",
        "Fairy Meadows", "Shigar", "Khaplu", "Nagar", "Astore"
    }
    
    return normalized in major_cities or any(
        normalized.lower() in city.lower() for city in major_cities
    )