# EXAIM Gradio Demo - Deployment Guide

This guide explains how to deploy the EXAIM demo on Hugging Face Spaces or run it locally.

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.9 or higher
- API keys for your chosen LLM provider (OpenAI, Google Gemini, or Groq)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/EXAIM-MedGemma-Impact.git
   cd EXAIM-MedGemma-Impact
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements-gradio.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

5. **Run the Gradio app:**
   ```bash
   python app_gradio.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:7860`

## ☁️ Deployment to Hugging Face Spaces

### Method 1: Using the Web Interface

1. **Create a new Space:**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select Gradio SDK
   - Choose a name for your Space

2. **Upload files:**
   Upload these files to your Space:
   - `app_gradio.py`
   - `requirements-gradio.txt` (rename to `requirements.txt`)
   - `README_GRADIO.md` (rename to `README.md`)
   - All files from `exaim_core/`, `demos/cdss_example/`, and `infra/` directories

3. **Configure secrets:**
   - Go to Settings → Repository secrets
   - Add your API keys:
     - `OPENAI_API_KEY` (if using OpenAI)
     - `GOOGLE_API_KEY` (if using Gemini)
     - `GROQ_API_KEY` (if using Groq)

4. **Deploy:**
   Your Space will automatically build and deploy!

### Method 2: Using Git

1. **Create a new Space and clone it:**
   ```bash
   git clone https://huggingface.co/spaces/your-username/your-space-name
   cd your-space-name
   ```

2. **Copy necessary files:**
   ```bash
   # Copy from your EXAIM project
   cp ../EXAIM-MedGemma-Impact/app_gradio.py .
   cp ../EXAIM-MedGemma-Impact/requirements-gradio.txt requirements.txt
   cp ../EXAIM-MedGemma-Impact/README_GRADIO.md README.md
   
   # Copy source directories
   cp -r ../EXAIM-MedGemma-Impact/exaim_core .
   cp -r ../EXAIM-MedGemma-Impact/demos .
   cp -r ../EXAIM-MedGemma-Impact/infra .
   ```

3. **Create a `.gitignore`:**
   ```bash
   echo "__pycache__/" >> .gitignore
   echo "*.pyc" >> .gitignore
   echo ".env" >> .gitignore
   echo ".venv/" >> .gitignore
   ```

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "Initial EXAIM demo deployment"
   git push
   ```

5. **Configure secrets in the Hugging Face web interface** (see Method 1, step 3)

## 🔧 Configuration

### Environment Variables

The app requires LLM API keys. Set these as environment variables:

**For local development (.env file):**
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

**For Hugging Face Spaces:**
Configure these in Settings → Repository secrets

### Model Configuration

EXAIM uses models configured in `infra/model_configs.yaml`. By default, it uses:
- Gemini models for agents (cost-effective, fast)
- You can modify this to use OpenAI or Groq models

To change models, edit the configuration in your CDSS initialization or update `model_configs.yaml`.

## 📊 Resource Requirements

### Local Deployment
- **RAM:** 4GB minimum, 8GB recommended
- **CPU:** 2 cores minimum, 4 cores recommended
- **Storage:** 1GB for dependencies

### Hugging Face Spaces
- Free tier should work for demos
- Consider upgrading to persistent storage for production use

## 🐛 Troubleshooting

### "API key not found" error
- Ensure your API keys are properly set in environment variables
- For Hugging Face Spaces, check Repository secrets are configured
- Verify the key names match exactly (case-sensitive)

### "Module not found" error
- Ensure all required directories are copied (`exaim_core`, `demos`, `infra`)
- Check that `requirements.txt` includes all dependencies
- Try reinstalling: `pip install -r requirements-gradio.txt --force-reinstall`

### Slow performance
- EXAIM processes cases through multiple LLM calls, which can take 30-60 seconds
- Consider using faster models (Groq) or parallel processing
- Hugging Face Spaces free tier may have rate limits

### Out of memory errors
- Reduce the number of concurrent users
- Consider upgrading to a paid Hugging Face Spaces tier
- Optimize model configuration to use smaller models

## 🔒 Security Notes

- Never commit API keys to Git repositories
- Use environment variables or secrets management
- For production deployments, implement rate limiting
- Consider adding user authentication for controlled access

## 📝 Customization

### Adding More Example Cases
Edit the `EXAMPLE_CASES` list in `app_gradio.py`:
```python
EXAMPLE_CASES = [
    "Your new example case here...",
    # ... more cases
]
```

### Changing the UI Theme
Modify the `theme` parameter in `gr.Blocks()`:
```python
theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")
```

### Adjusting Output Formatting
Modify the `format_raw_traces()` and `format_summaries()` methods in `GradioStreamingHandler`.

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs/)
- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [LangChain Documentation](https://python.langchain.com/)
- [EXAIM Paper](link-to-your-paper)

## 🤝 Support

For issues or questions:
- Open an issue on GitHub
- Contact: your-email@example.com
- Discussion: Hugging Face Spaces discussion tab

## 📄 License

MIT License - See LICENSE file for details
