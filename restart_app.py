#!/usr/bin/env python3
"""
Restart and test the Pakistan Travel RAG system
"""
import os
import sys
from pathlib import Path

print("🔄 Restarting Pakistan Travel Intelligence System...")

# Clear Python cache
if hasattr(sys, 'modules'):
    modules_to_clear = [m for m in sys.modules.keys() if m.startswith('src.')]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

print("✅ Python modules cleared")

# Add src to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment loaded")
except Exception as e:
    print(f"❌ Environment error: {e}")
    sys.exit(1)

# Verify configuration
groq_key = os.getenv('GROQ_API_KEY')
groq_model = os.getenv('GROQ_MODEL')

print(f"🔧 Configuration Check:")
print(f"   GROQ_API_KEY: {'✅ Set' if groq_key else '❌ Missing'}")
print(f"   GROQ_MODEL: {groq_model}")

if not groq_key:
    print("❌ Missing API key - check .env file")
    sys.exit(1)

# Test API
try:
    from groq import Groq
    
    client = Groq(api_key=groq_key)
    
    completion = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "user", "content": "What's a good budget for visiting Karachi for 2 days? Answer in PKR."}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    
    if completion.choices:
        response = completion.choices[0].message.content
        print("✅ API Test Successful!")
        print(f"Sample Response: {response[:100]}...")
        
        print("\n🚀 System is ready! Now restart Streamlit:")
        print("   streamlit run app.py")
        
    else:
        print("❌ API test failed - no response")
        
except Exception as e:
    print(f"❌ API test failed: {e}")
    if "llama-3.1-70b-versatile" in str(e):
        print("⚠️  Old model still being used - check environment variables")
    elif "decommissioned" in str(e):
        print("⚠️  Model is deprecated - using backup model")
        # Try with a different model
        try:
            completion = client.chat.completions.create(
                model="llama3-8b-8192",  # Fallback model
                messages=[
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=10,
            )
            print("✅ Fallback model works - update .env to use 'llama3-8b-8192'")
        except:
            print("❌ Fallback also failed")