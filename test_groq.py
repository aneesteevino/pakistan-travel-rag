#!/usr/bin/env python3
"""
Test script to verify Groq API configuration with updated model
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    """Test Groq API with the new model"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL")
    
    print(f"GROQ_API_KEY configured: {'✅' if groq_api_key else '❌'}")
    print(f"GROQ_MODEL: {groq_model}")
    
    if not groq_api_key:
        print("❌ No Groq API key found")
        return False
        
    try:
        from groq import Groq
        
        client = Groq(api_key=groq_api_key)
        
        # Test with a simple travel question
        completion = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "user", "content": "How much budget should I need to visit Karachi? Provide answer in PKR."}
            ],
            max_tokens=1000,
            temperature=0.7,
        )
        
        if completion.choices and len(completion.choices) > 0:
            response = completion.choices[0].message.content
            print("✅ Groq API test successful!")
            print(f"Response: {response}")
            return True
        else:
            print("❌ No response from Groq API")
            return False
            
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

if __name__ == "__main__":
    test_groq_api()