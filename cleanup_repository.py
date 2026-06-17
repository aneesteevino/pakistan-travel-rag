#!/usr/bin/env python3
"""
Repository cleanup and refactoring script for Pakistan Travel RAG System.
Removes unused files, consolidates datasets, and prepares for architecture upgrade.
"""

import os
import shutil
from pathlib import Path

def cleanup_unused_files():
    """Remove unused legacy datasets and files."""
    
    # Files to remove
    unused_files = [
        "data/destinations.csv",
        "data/hotels.csv", 
        "data/activities.csv",
        "data/visa_requirements.csv",
        "data/weather_data.csv",
        "data/combined_data.csv",
        "data/pakistan_tourism_dataset.csv",
        "data/PIA_2026_Advanced_Kaggle_Dataset.csv",
        "data/PIA_2026_Advanced_Kaggle_Dataset.xlsx",
        "data/airbnb-listings-in-pakistan.csv",
        "data/sample-data-Guest_houses.csv", 
        "data/road_transport.csv",
        "data/Pakistan Cities and Zip Codes.csv"
    ]
    
    # Directories to remove
    unused_dirs = [
        "ralph/",
        "logs/",
        "__pycache__/",
        "src/__pycache__/",
        "vector_db/",
        "cache/"
    ]
    
    # Remove unused files
    for file_path in unused_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")
    
    # Remove unused directories (skip if files are in use)
    for dir_path in unused_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"✅ Removed directory: {dir_path}")
            except PermissionError:
                print(f"⚠️ Skipping {dir_path} - files in use")

def create_service_structure():
    """Create new service architecture."""
    
    services = [
        "src/accommodation_service.py",
        "src/transport_service.py", 
        "src/email_service.py",
        "src/airport_registry.py",
        "src/llm_fallback.py",
        "src/rag_pipeline.py",
        "src/city_normalizer.py"
    ]
    
    for service in services:
        Path(service).parent.mkdir(parents=True, exist_ok=True)
        if not Path(service).exists():
            Path(service).touch()
            print(f"✅ Created: {service}")

def audit_dataset_usage():
    """Audit current dataset usage."""
    
    # Current dataset to feature mapping
    dataset_usage = {
        "data/new data/Hotels data/hotels.csv": "accommodation_search",
        "data/new data/Guest houses data/guest_houses.csv": "accommodation_search"
    }
    
    # Features to API endpoints
    feature_to_api = {
        "accommodation_search": ["/accommodations", "/accommodations/compare"],
        "transport_comparison": ["/transport/routes", "/transport/compare"],
        "itinerary_generation": ["/planner/generate", "/planner/send-document"]
    }
    
    # UI components mapping
    ui_components = {
        "accommodation_search": ["AccommodationSearch", "AccommodationCompare"],
        "transport_comparison": ["TransportCompare", "RouteCompare"],  
        "itinerary_generation": ["PlannerWizard", "ProgressiveDisclosure"]
    }
    
    print("📊 Dataset Usage Audit:")
    for dataset, feature in dataset_usage.items():
        if os.path.exists(dataset):
            print(f"   ✅ {dataset} → {feature}")
            apis = feature_to_api.get(feature, [])
            components = ui_components.get(feature, [])
            print(f"      APIs: {apis}")
            print(f"      UI Components: {components}")
        else:
            print(f"   ❌ {dataset} → {feature} (MISSING)")

if __name__ == "__main__":
    print("🧹 Starting Repository Cleanup...")
    cleanup_unused_files()
    create_service_structure()
    audit_dataset_usage()
    print("✅ Repository cleanup complete!")