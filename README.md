# 🇵🇰 Pakistan Travel Intelligence RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive **Retrieval-Augmented Generation (RAG)** system for Pakistan travel planning, powered by semantic search and AI generation. Get personalized travel recommendations, itineraries, and budget planning for destinations across Pakistan.

## 🌟 Features

### 🤖 **AI Travel Assistant**
- Interactive chatbot for Pakistan travel questions
- Budget recommendations in Pakistani Rupees (PKR)
- Real-time responses about destinations, costs, and activities

### 📅 **Itinerary Builder**
- Generate day-wise travel plans for Pakistan destinations
- Customize by duration, budget, travel style, and interests
- Includes accommodation, activities, and cultural experiences

### ⚖️ **Destination Comparison**
- Compare Pakistan destinations side-by-side
- Safety ratings, costs, best visiting times
- Activity recommendations and cultural highlights

### 🧳 **Trip Planner**
- Personalized trip recommendations based on preferences
- Multi-destination planning within budget constraints
- Transportation and accommodation suggestions

### 📊 **Knowledge Base Browser**
- Explore the underlying Pakistan travel dataset
- 174 documents covering destinations, hotels, activities
- Transparent source attribution and similarity scores

## 🏗️ Architecture

```
User Query
    ↓
Embedding Model (sentence-transformers)
    ↓
FAISS Vector Search
    ↓
Retrieved Context
    ↓
LLM Generation (Groq/Gemini)
    ↓
Grounded Response
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pakistan-travel-rag.git
cd pakistan-travel-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. **Run the application**
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 🔑 Configuration

### Required API Keys

Get your API keys from:
- **Groq**: [console.groq.com](https://console.groq.com) (Recommended)
- **Google Gemini**: [ai.google.dev](https://ai.google.dev) (Optional)

Add to `.env` file:
```env
# Primary LLM Provider
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Optional: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# RAG Settings
LLM_PROVIDER=groq
TOP_K=6
SIMILARITY_THRESHOLD=0.03
```

### Supported Models
- **Groq**: `llama-3.1-8b-instant`, `llama3-8b-8192`, `mixtral-8x7b-32768`
- **Gemini**: `gemini-1.5-flash`, `gemini-1.5-pro`

## 📁 Project Structure

```
pakistan-travel-rag/
├── app.py                 # Main Streamlit application
├── src/
│   ├── config.py          # Configuration settings
│   ├── generator.py       # LLM response generation
│   ├── retriever.py       # Vector similarity search
│   ├── vector_store.py    # FAISS vector database
│   ├── data_loader.py     # CSV data processing
│   └── itinerary.py       # Travel planning logic
├── data/                  # Pakistan travel datasets
│   ├── destinations.csv   # Tourist destinations
│   ├── hotels.csv         # Accommodation data
│   ├── activities.csv     # Things to do
│   ├── weather_data.csv   # Climate information
│   └── visa_requirements.csv
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── README.md             # This file
```

## 🗂️ Dataset

The knowledge base includes **174 documents** covering:

- **🏔️ Destinations**: 50+ Pakistan tourist spots (Karachi, Lahore, Hunza, Skardu, etc.)
- **🏨 Hotels**: Accommodation options across budget ranges
- **🎯 Activities**: Adventure, cultural, historical, and nature activities
- **🌤️ Weather**: Best visiting times and climate data
- **📋 Visa Info**: Entry requirements and travel regulations

All data is Pakistan-focused with costs in Pakistani Rupees (PKR).

## 💡 Usage Examples

### Travel Assistant
```
Q: "How much budget should I need to visit Karachi for 3 days?"
A: "For a 3-day trip to Karachi, budget approximately:
   • Budget travel: 6,000-12,000 PKR total
   • Mid-range: 15,000-36,000 PKR total
   • Luxury: 45,000+ PKR total
   
   This includes accommodation, meals, local transport, and activities..."
```

### Itinerary Builder
- **Destination**: Hunza Valley
- **Duration**: 5 days
- **Budget**: 8,000 PKR/day
- **Style**: Adventure
- **Interests**: Mountains, Photography, Culture

→ Generates detailed day-wise plan with activities, costs, and recommendations

## 🛠️ Development

### Running Tests
```bash
python debug_config.py    # Test API configuration
python simple_app.py      # Run simplified version
```

### Adding New Data
1. Add CSV files to `/data` directory
2. Follow existing schema format
3. System will automatically reindex on next run

### Customization
- Modify prompts in `src/generator.py`
- Adjust retrieval parameters in `src/config.py`
- Add new travel categories in `TRAVEL_INTERESTS`

## 🔧 Troubleshooting

### Common Issues

**"No module named 'faiss'"**
```bash
pip install faiss-cpu
```

**"No AI service available"**
- Check API keys in `.env` file
- Verify model names are correct
- Test with `python debug_config.py`

**"Could not generate response"**
- Check internet connection
- Verify API key permissions
- Try different model (update `GROQ_MODEL`)

### Performance Tips
- First run takes longer (building vector index)
- Increase `TOP_K` for more context (may increase costs)
- Use `faiss-gpu` for faster similarity search (requires CUDA)

## 📊 System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies and models
- **Internet**: Required for LLM API calls
- **OS**: Windows, macOS, Linux

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution
- 🗺️ Additional Pakistan travel datasets
- 🌐 Multi-language support (Urdu, regional languages)
- 📱 Mobile-responsive UI improvements
- 🔍 Advanced search and filtering
- 📈 Travel analytics and insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Pakistan Tourism Development Corporation** for destination insights
- **OpenAI community** for RAG architecture patterns
- **Streamlit** for the amazing web framework
- **FAISS team** for efficient vector search
- **Contributors** who help improve Pakistan travel accessibility

## 📞 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/pakistan-travel-rag/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/pakistan-travel-rag/discussions)
- 📧 **Email**: your-email@example.com

---

**Made with ❤️ for Pakistan Travel**

*Promoting sustainable and informed tourism in Pakistan through AI-powered travel assistance.*