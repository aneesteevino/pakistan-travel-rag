# 🇵🇰 Pakistan Travel Intelligence RAG System

A comprehensive **Retrieval-Augmented Generation (RAG)** system for Pakistan travel planning, powered by semantic search and LLM generation. Get personalized travel recommendations, itineraries, and budget planning for Pakistan destinations.

![Pakistan Travel RAG](https://img.shields.io/badge/Travel-Pakistan-green?style=for-the-badge&logo=map)
![AI Powered](https://img.shields.io/badge/AI-Powered-blue?style=for-the-badge&logo=robot)
![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red?style=for-the-badge&logo=streamlit)

## ✨ Features

### 🤖 **AI Travel Assistant**
- Get instant answers to Pakistan travel questions
- Budget recommendations in Pakistani Rupees (PKR)
- Destination-specific advice and tips
- Cultural insights and local recommendations

### 📅 **Itinerary Builder**
- Generate day-wise travel plans for Pakistan destinations
- Customize by budget, travel style, and interests
- Accommodation and activity suggestions
- Transportation recommendations

### ⚖️ **Destination Comparison**
- Compare Pakistan destinations side-by-side
- Budget analysis, safety ratings, and best times to visit
- Activities, accommodation, and cultural experiences
- Data-driven recommendations

### 🧳 **Trip Planner**
- Personalized Pakistan travel plans based on preferences
- Multi-destination recommendations
- Budget optimization and breakdown
- Interest-based activity suggestions

### 📊 **Knowledge Base Browser**
- Explore the Pakistan travel dataset
- View destinations, hotels, activities, and weather data
- Understand the AI's knowledge sources
- Real data transparency

## 🏗️ Architecture

```
User Query
    │
    ▼
Semantic Search     ← Sentence Transformers
    │
    ▼  
FAISS Vector DB    ← 174 Pakistan travel documents
    │
    ▼
Context Retrieval  ← Top-K similar documents
    │
    ▼
Generator          ← Groq LLaMA / Google Gemini
    │
    ▼
Grounded Response  ← Pakistan-specific, accurate answer
```

## 🚀 Quick Start

### 1️⃣ **Clone Repository**
```bash
git clone https://github.com/yourusername/pakistan-travel-rag.git
cd pakistan-travel-rag
```

### 2️⃣ **Create Virtual Environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Configure API Keys**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# Get Groq API key from: https://console.groq.com/
# Get Gemini API key from: https://makersuite.google.com/
```

### 5️⃣ **Run the Application**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to start planning your Pakistan adventure! 🎉

## ⚙️ Configuration

### **Environment Variables**
```bash
# LLM Provider (groq | gemini)
LLM_PROVIDER=groq

# Groq Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Google Gemini Configuration  
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# RAG Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=512
TOP_K=6
SIMILARITY_THRESHOLD=0.03
```

### **Supported Models**
- **Groq**: `llama-3.1-8b-instant`, `mixtral-8x7b-32768`
- **Google Gemini**: `gemini-1.5-flash`, `gemini-1.5-pro`

## 📊 Dataset

The system includes comprehensive Pakistan travel data:

| Dataset | Records | Description |
|---------|---------|-------------|
| **Destinations** | 20+ | Major Pakistan cities and tourist spots |
| **Hotels** | 50+ | Accommodation across different budgets |  
| **Activities** | 100+ | Things to do, attractions, experiences |
| **Weather** | 12 months | Climate data for travel planning |
| **Visa Requirements** | Multiple | Entry requirements and procedures |

### **Data Sources**
- Pakistan Tourism Development Corporation (PTDC)
- Local travel operators and guides
- Government tourism websites
- Travel community contributions

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Vector Database** | FAISS |
| **Embeddings** | Sentence Transformers |
| **LLM Providers** | Groq, Google Gemini |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.8+ |

## 📁 Project Structure

```
pakistan-travel-rag/
├── 📁 src/                    # Source code
│   ├── config.py              # Configuration management
│   ├── data_loader.py         # CSV data processing
│   ├── vector_store.py        # FAISS vector database
│   ├── retriever.py           # Semantic search
│   ├── generator.py           # LLM integration
│   └── itinerary.py           # Travel planning logic
├── 📁 data/                   # Pakistan travel datasets
│   ├── destinations.csv       # Cities and attractions
│   ├── hotels.csv             # Accommodation data
│   ├── activities.csv         # Things to do
│   ├── weather_data.csv       # Climate information
│   └── visa_requirements.csv  # Entry procedures
├── 📄 app.py                  # Main Streamlit application
├── 📄 requirements.txt        # Python dependencies
├── 📄 .env.example            # Environment template
└── 📄 README.md               # This file
```

## 🧪 Testing

### **Quick API Test**
```bash
# Test with simplified version (no dependencies)
streamlit run simple_app.py
```

### **Full System Test**
```bash
# Run installation checker
python install_deps.py

# Test all components
python debug_config.py
```

## 🌟 Usage Examples

### **Budget Planning**
> "How much budget should I need to visit Karachi for 5 days?"

### **Itinerary Generation**
> Generate a 7-day adventure itinerary for Hunza Valley with ₨15,000 daily budget

### **Destination Comparison**  
> Compare Lahore vs Islamabad for cultural tourism

### **Trip Planning**
> Plan a 10-day Pakistan trip for ₨200,000 focusing on mountains and photography

## 🤝 Contributing

We welcome contributions to improve Pakistan travel intelligence! 

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Areas for Contribution**
- 📊 **Dataset Expansion**: Add more destinations, hotels, activities
- 🔧 **Feature Development**: New travel planning tools
- 🐛 **Bug Fixes**: Improve reliability and performance
- 📚 **Documentation**: Enhance setup guides and examples
- 🌍 **Localization**: Support for Urdu and regional languages

## 📋 Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **API Keys**: Groq or Google Gemini account

## 🔧 Troubleshooting

### **Common Issues**

**🚨 "No module named 'faiss'"**
```bash
pip install faiss-cpu
```

**🚨 "API key not configured"**  
```bash
# Check .env file exists and contains valid API keys
cat .env
```

**🚨 "Model decommissioned error"**
```bash
# Update to supported model in .env
GROQ_MODEL=llama-3.1-8b-instant
```

### **Getting Help**
- 📖 Check the [Issues](../../issues) for common problems
- 💬 Start a [Discussion](../../discussions) for questions
- 📧 Contact maintainers for urgent issues

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pakistan Tourism Development Corporation** for destination data
- **Local travel communities** for authentic recommendations  
- **Open source contributors** for making this project possible
- **Groq & Google** for providing AI infrastructure

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **Documentation**: [Wiki](../../wiki)
- **API Reference**: [Docs](docs/)
- **Dataset**: [Pakistan Travel Data](data/)

---

<div align="center">

**Made with ❤️ for Pakistan Travel**

[Report Bug](../../issues) • [Request Feature](../../issues) • [Contribute](../../pulls)

</div>