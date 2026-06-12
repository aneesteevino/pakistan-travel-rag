#!/usr/bin/env powershell

Write-Host "🚀 Setting up Pakistan Travel Intelligence RAG Git Repository" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Initialize git repository
if (Test-Path ".git") {
    Write-Host "✅ Git repository already initialized" -ForegroundColor Green
} else {
    Write-Host "📂 Initializing git repository..." -ForegroundColor Yellow
    git init
    Write-Host "✅ Git repository initialized" -ForegroundColor Green
}

# Add all files
Write-Host "📝 Adding files to git..." -ForegroundColor Yellow
git add .

# Create initial commit
Write-Host "💾 Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Pakistan Travel Intelligence RAG System

Features:
- 🤖 AI Travel Assistant for Pakistan
- 📅 Itinerary Builder with budget planning
- ⚖️ Destination comparison tool
- 🧳 Personalized trip planner
- 📊 Knowledge base with 174+ documents
- 🔍 Semantic search with FAISS
- 🎯 Groq/Gemini LLM integration

Includes:
- Complete Pakistan travel dataset
- RAG architecture with vector similarity
- Streamlit web interface
- Docker support
- CI/CD workflows
- Comprehensive documentation"

Write-Host "✅ Initial commit created" -ForegroundColor Green

# Set up main branch
Write-Host "🌿 Setting up main branch..." -ForegroundColor Yellow
git branch -M main

Write-Host
Write-Host "🎉 Repository setup complete!" -ForegroundColor Green
Write-Host
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create repository on GitHub:" -ForegroundColor White
Write-Host "   https://github.com/new" -ForegroundColor Blue
Write-Host
Write-Host "2. Add remote origin:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/yourusername/pakistan-travel-rag.git" -ForegroundColor Blue
Write-Host
Write-Host "3. Push to GitHub:" -ForegroundColor White
Write-Host "   git push -u origin main" -ForegroundColor Blue
Write-Host
Write-Host "4. Set up environment:" -ForegroundColor White
Write-Host "   cp .env.example .env" -ForegroundColor Blue
Write-Host "   # Add your API keys to .env" -ForegroundColor Gray
Write-Host
Write-Host "5. Install and run:" -ForegroundColor White
Write-Host "   pip install -r requirements.txt" -ForegroundColor Blue
Write-Host "   streamlit run app.py" -ForegroundColor Blue
Write-Host

Read-Host "Press Enter to continue"