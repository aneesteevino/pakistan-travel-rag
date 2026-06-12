"""
Simplified Pakistan Travel Assistant - No RAG dependencies required
Just tests the LLM functionality with Groq API
"""

import streamlit as st
import os
import os

# Simple config - Load from environment or use placeholder
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

st.set_page_config(
    page_title="Pakistan Travel Assistant - Simple",
    page_icon="🇵🇰",
    layout="wide"
)

st.title("🇵🇰 Pakistan Travel Assistant - Simple Version")
st.write("Testing Groq API functionality for Pakistan travel questions")

# Test Groq API
def test_groq_api(question: str) -> str:
    """Test Groq API with travel question"""
    try:
        from groq import Groq
        
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""You are a Pakistan travel assistant. Help with travel questions about Pakistan.
        
        For budget questions, provide realistic estimates in Pakistani Rupees (PKR):
        - Budget travel: 2,000-4,000 PKR per day
        - Mid-range travel: 5,000-12,000 PKR per day  
        - Luxury travel: 15,000+ PKR per day
        
        Popular Pakistan destinations include:
        - Karachi (beaches, food, culture)
        - Lahore (history, architecture, cuisine)
        - Islamabad (modern capital, Margalla Hills)
        - Hunza Valley (mountains, culture)
        - Skardu (K2 base, lakes)
        
        Question: {question}
        
        Please provide a helpful response about Pakistan travel."""
        
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        
        if completion.choices:
            return completion.choices[0].message.content
        else:
            return "❌ No response from API"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Show current config
st.sidebar.title("Configuration")
st.sidebar.write(f"**Model:** {GROQ_MODEL}")
st.sidebar.write(f"**API Key:** {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")

# Test questions
st.subheader("🧪 Quick Tests")
col1, col2 = st.columns(2)

with col1:
    if st.button("Test: Karachi Budget"):
        with st.spinner("Testing..."):
            response = test_groq_api("How much budget should I need to visit Karachi for 3 days?")
        st.write("**Response:**")
        st.write(response)

with col2:
    if st.button("Test: Lahore Attractions"):
        with st.spinner("Testing..."):
            response = test_groq_api("What are the top attractions to visit in Lahore?")
        st.write("**Response:**")
        st.write(response)

# Custom question
st.subheader("💬 Ask Your Question")
user_question = st.text_input("Ask about Pakistan travel:", placeholder="e.g., What's the best time to visit Hunza Valley?")

if user_question:
    if st.button("Get Answer"):
        with st.spinner("Getting response..."):
            response = test_groq_api(user_question)
        
        st.write("**Answer:**")
        st.write(response)

# Installation note
st.markdown("---")
st.info("""
**Note:** This is a simplified version to test the API. 
For the full RAG system, install dependencies:
```
pip install -r requirements.txt
```
Then run: `streamlit run app.py`
""")