#!/bin/bash

echo "🚀 Setting up Pakistan Travel Intelligence RAG Git Repository"
echo "================================================================"

# Initialize git repository
if [ -d ".git" ]; then
    echo "✅ Git repository already initialized"
else
    echo "📂 Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
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

echo "✅ Initial commit created"

# Set up main branch
echo "🌿 Setting up main branch..."
git branch -M main

echo
echo "🎉 Repository setup complete!"
echo
echo "Next steps:"
echo "1. Create repository on GitHub: https://github.com/new"
echo "2. Add remote origin: git remote add origin https://github.com/yourusername/pakistan-travel-rag.git"
echo "3. Push to GitHub: git push -u origin main"
echo "4. Set up environment: cp .env.example .env && # Add your API keys"
echo "5. Install and run: pip install -r requirements.txt && streamlit run app.py"
echo

read -p "Press Enter to continue..."