@echo off
echo ================================================================
echo Pakistan Travel Intelligence RAG System - Full Installation
echo ================================================================
echo.

echo Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo ✅ Virtual environment found
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

echo.
echo Installing dependencies from requirements.txt...
echo ================================================================

echo 📦 Installing core packages (streamlit, pandas, numpy)...
pip install streamlit>=1.35.0 pandas>=2.0.0 numpy>=1.24.0

echo.
echo 📦 Installing python-dotenv...
pip install python-dotenv>=1.0.0

echo.
echo 📦 Installing Groq API client...
pip install groq>=0.9.0

echo.
echo 📦 Installing Google Generative AI (optional)...
pip install google-generativeai>=0.7.0

echo.
echo 🧠 Installing ML/AI packages (this may take a while)...
echo Installing PyTorch...
pip install torch>=2.0.0

echo.
echo Installing sentence-transformers...
pip install sentence-transformers>=2.7.0

echo.
echo 🗄️ Installing vector database...
echo Installing FAISS (CPU version)...
pip install faiss-cpu>=1.8.0

echo.
echo ================================================================
echo ✅ Installation complete!
echo ================================================================
echo.
echo Now you can run the full Pakistan Travel RAG system:
echo   streamlit run app.py
echo.
echo The system includes:
echo - 🤖 AI Travel Assistant with Pakistan knowledge base
echo - 📅 Itinerary Builder for Pakistan destinations  
echo - ⚖️ Destination Comparison tool
echo - 🧳 Trip Planner with budget optimization
echo - 📊 Knowledge Base browser
echo.
pause