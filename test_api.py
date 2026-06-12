#!/usr/bin/env python3
"""
Test API configuration
"""
import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ dotenv loaded")
except ImportError:
    print("❌ dotenv not installed")
    sys.exit(1)

# Test environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL")

print(f"GROQ_API_KEY: {'✅ Set' if groq_api_key else '❌ Not set'}")
print(f"GROQ_MODEL: {groq_model}")

if groq_api_key:
    print(f"Key length: {len(groq_api_key)}")
    print(f"Key starts with: {groq_api_key[:10]}...")

# Test Groq API
if groq_api_key:
    try:
        from groq import Groq
        
        client = Groq(api_key=groq_api_key)
        
        completion = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "user", "content": "How much budget should I need to visit Karachi for 3 days? Answer in PKR."}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        
        if completion.choices:
            response = completion.choices[0].message.content
            print("✅ Groq API working!")
            print(f"Response: {response[:200]}...")
        else:
            print("❌ No response from Groq")
            
    except Exception as e:
        print(f"❌ Groq API error: {e}")
else:
    print("❌ Cannot test API - no key")