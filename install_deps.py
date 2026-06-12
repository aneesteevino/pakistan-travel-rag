#!/usr/bin/env python3
"""
Pakistan Travel RAG System - Dependency Installer
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        print(f"🔄 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, 
                               capture_output=True, text=True)
        print(f"✅ Success: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("================================================================")
    print("Pakistan Travel Intelligence RAG System - Installation")
    print("================================================================")
    
    # List of packages to install
    packages = [
        "streamlit>=1.35.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "groq>=0.9.0",
        "google-generativeai>=0.7.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.7.0", 
        "faiss-cpu>=1.8.0"
    ]
    
    print(f"📦 Installing {len(packages)} packages...")
    print()
    
    success_count = 0
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if run_command(f"pip install {package}"):
            success_count += 1
        else:
            failed_packages.append(package)
        print()
    
    print("================================================================")
    print("Installation Summary:")
    print(f"✅ Successful: {success_count}/{len(packages)}")
    
    if failed_packages:
        print(f"❌ Failed: {len(failed_packages)}")
        print("Failed packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print()
        print("Try installing failed packages manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
    else:
        print("🎉 All packages installed successfully!")
        print()
        print("You can now run the full system:")
        print("  streamlit run app.py")
    
    print("================================================================")

if __name__ == "__main__":
    main()