@echo off
echo Installing Pakistan Travel RAG dependencies...
echo.

echo Installing core dependencies...
pip install streamlit pandas numpy python-dotenv

echo Installing ML dependencies...
pip install sentence-transformers torch

echo Installing vector database...
pip install faiss-cpu

echo Installing LLM providers...
pip install google-generativeai groq

echo.
echo Installation complete! Now run:
echo streamlit run app.py
pause