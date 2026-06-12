# Contributing to Pakistan Travel Intelligence RAG System

Thank you for your interest in contributing to the Pakistan Travel Intelligence RAG System! 🇵🇰

## How to Contribute

### 🐛 Bug Reports
- Use the [GitHub Issues](https://github.com/yourusername/pakistan-travel-rag/issues) page
- Search existing issues before creating new ones
- Include system information, error messages, and steps to reproduce
- Use the bug report template

### 💡 Feature Requests
- Check [existing feature requests](https://github.com/yourusername/pakistan-travel-rag/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)
- Describe the problem and proposed solution
- Include use cases and examples
- Use the feature request template

### 🔧 Code Contributions

#### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/pakistan-travel-rag.git
   cd pakistan-travel-rag
   ```
3. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Set up pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

#### Making Changes
1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test your changes:
   ```bash
   python debug_config.py  # Test configuration
   streamlit run app.py     # Test full system
   ```
4. Commit with descriptive messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request

#### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions focused and small
- Add type hints where appropriate

#### Testing
- Test your changes thoroughly
- Include both positive and negative test cases
- Test with different API providers (Groq/Gemini)
- Verify the system works without API keys (graceful degradation)

### 📊 Data Contributions

We welcome contributions to expand the Pakistan travel dataset:

#### Adding Destinations
1. Add entries to `data/destinations.csv`
2. Include: name, description, location, categories, safety_rating, budget_level
3. Ensure data accuracy and cultural sensitivity

#### Adding Activities
1. Update `data/activities.csv`
2. Include: activity_name, location, category, cost_pkr, duration, description
3. Focus on authentic Pakistan experiences

#### Data Quality Guidelines
- Verify information accuracy
- Use Pakistani Rupees (PKR) for all costs
- Include seasonal variations where relevant
- Respect local culture and customs
- Cite reliable sources when possible

### 🌐 Documentation

Help improve documentation:
- Fix typos and grammatical errors
- Add examples and use cases
- Improve installation instructions
- Translate to other languages (Urdu, regional languages)
- Add screenshots and videos

### 💬 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and experiences
- Focus on constructive feedback
- Celebrate diversity in travel experiences

### 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation updates
- `data`: Dataset-related issues
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `pakistan-tourism`: Pakistan-specific content

### 📝 Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Request review from maintainers
6. Address feedback promptly
7. Maintain clean commit history

### 🎯 Priority Areas

We're especially looking for help with:

1. **Dataset Expansion**
   - More Pakistan destinations
   - Regional cuisines and food costs
   - Transportation options and costs
   - Accommodation data

2. **Internationalization**
   - Urdu language support
   - Regional language translations
   - Currency conversion features

3. **User Experience**
   - Mobile-responsive design
   - Accessibility improvements
   - Performance optimizations

4. **Advanced Features**
   - Real-time price updates
   - Weather integration
   - Social features (reviews, ratings)
   - Offline mode support

### 🔄 Release Process

1. Features are merged to `develop` branch
2. Regular releases from `develop` to `main`
3. Semantic versioning (e.g., v1.2.3)
4. Release notes for each version

### ❓ Getting Help

- Join discussions in [GitHub Discussions](https://github.com/yourusername/pakistan-travel-rag/discussions)
- Ask questions in issues with `question` label
- Contact maintainers directly for sensitive issues

### 🙏 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Given credit for significant contributions
- Invited to be maintainers for consistent contributors

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

---

**Thank you for contributing to making Pakistan travel more accessible and enjoyable for everyone!** 🚀