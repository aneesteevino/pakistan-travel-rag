Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Pakistan Travel Intelligence RAG System - Full Installation" -ForegroundColor Cyan  
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host

Write-Host "Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    & .\venv\Scripts\Activate.ps1
}

Write-Host
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host "📦 Installing core packages (streamlit, pandas, numpy)..." -ForegroundColor Blue
pip install streamlit pandas numpy

Write-Host
Write-Host "📦 Installing python-dotenv..." -ForegroundColor Blue  
pip install python-dotenv

Write-Host
Write-Host "📦 Installing Groq API client..." -ForegroundColor Blue
pip install groq

Write-Host
Write-Host "📦 Installing Google Generative AI (optional)..." -ForegroundColor Blue
pip install google-generativeai

Write-Host
Write-Host "🧠 Installing ML/AI packages (this may take a while)..." -ForegroundColor Magenta
Write-Host "Installing PyTorch..." -ForegroundColor Blue
pip install torch

Write-Host
Write-Host "Installing sentence-transformers..." -ForegroundColor Blue
pip install sentence-transformers

Write-Host  
Write-Host "🗄️ Installing vector database..." -ForegroundColor Magenta
Write-Host "Installing FAISS (CPU version)..." -ForegroundColor Blue
pip install faiss-cpu

Write-Host
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "✅ Installation complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host

Write-Host "Now you can run the full Pakistan Travel RAG system:" -ForegroundColor Yellow
Write-Host "  streamlit run app.py" -ForegroundColor White

Write-Host
Write-Host "The system includes:" -ForegroundColor Yellow
Write-Host "- 🤖 AI Travel Assistant with Pakistan knowledge base" -ForegroundColor White
Write-Host "- 📅 Itinerary Builder for Pakistan destinations" -ForegroundColor White  
Write-Host "- ⚖️ Destination Comparison tool" -ForegroundColor White
Write-Host "- 🧳 Trip Planner with budget optimization" -ForegroundColor White
Write-Host "- 📊 Knowledge Base browser" -ForegroundColor White

Write-Host
Read-Host "Press Enter to continue"