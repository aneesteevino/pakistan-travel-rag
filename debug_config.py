#!/usr/bin/env python3
"""
Debug configuration - Check current environment variables and test API
"""
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print("=== Debug Configuration ===")

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded successfully")
except Exception as e:
    print(f"❌ dotenv error: {e}")

# Show current environment variables
print(f"\nEnvironment Variables:")
print(f"GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not set'}")
print(f"GROQ_MODEL: {os.getenv('GROQ_MODEL')}")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")

# Test config import
try:
    from src.config import GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER
    print(f"\nConfig Import Test:")
    print(f"GROQ_API_KEY: {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")  
    print(f"GROQ_MODEL: {GROQ_MODEL}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
except Exception as e:
    print(f"❌ Config import error: {e}")

# Test API call
if os.getenv('GROQ_API_KEY'):
    try:
        from groq import Groq
        
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        print(f"\nTesting Groq API with model: {os.getenv('GROQ_MODEL')}")
        
        completion = client.chat.completions.create(
            model=os.getenv('GROQ_MODEL'),
            messages=[
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ],
            max_tokens=50,
            temperature=0,
        )
        
        if completion.choices:
            response = completion.choices[0].message.content
            print(f"✅ API Test Result: {response}")
        else:
            print("❌ No response from API")
            
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
else:
    print("❌ No API key to test")