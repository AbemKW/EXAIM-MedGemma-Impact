# EXAIM Gradio Demo - Quick Reference

## 📁 Files Created

### Main Application
- **`app_gradio.py`** - Main Gradio application with streaming UI
  - Patient case input interface
  - Side-by-side raw traces and EXAIM summaries display
  - Three example clinical cases included
  - Async processing with real-time updates

### Configuration
- **`requirements-gradio.txt`** - Python dependencies for Gradio demo
- **`.env.example`** - Template for environment variables (API keys)
- **`README_GRADIO.md`** - Comprehensive README for Hugging Face Spaces

### Deployment
- **`GRADIO_DEPLOYMENT.md`** - Detailed deployment guide
  - Local deployment instructions
  - Hugging Face Spaces deployment (2 methods)
  - Troubleshooting guide
  
- **`start-gradio.ps1`** - Windows quick start script
- **`start-gradio.sh`** - Unix/Mac/Linux quick start script

## 🚀 Quick Start

### Option 1: Using Quick Start Script

**Windows:**
```powershell
./start-gradio.ps1
```

**Unix/Mac/Linux:**
```bash
chmod +x start-gradio.sh
./start-gradio.sh
```

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-gradio.txt
   ```

3. **Set up API keys:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

4. **Run the app:**
   ```bash
   python app_gradio.py
   ```

5. **Open browser:**
   Navigate to http://localhost:7860

## ☁️ Deploy to Hugging Face Spaces

### Quick Deploy

1. **Create new Space** at https://huggingface.co/spaces
2. **Select Gradio SDK**
3. **Upload files:**
   - `app_gradio.py`
   - Rename `requirements-gradio.txt` → `requirements.txt`
   - Rename `README_GRADIO.md` → `README.md`
   - Copy entire directories: `exaim_core/`, `demos/`, `infra/`

4. **Configure secrets** (Settings → Repository secrets):
   - Add `GOOGLE_API_KEY` (or `OPENAI_API_KEY` or `GROQ_API_KEY`)

5. **Deploy** - Space will auto-build!

## 🔧 Configuration

### Required API Keys (choose one)

- **Google Gemini** (Recommended - fast, cheap)
  - Get key: https://makersuite.google.com/app/apikey
  - Set: `GOOGLE_API_KEY=your_key`

- **OpenAI** (GPT-4/3.5)
  - Get key: https://platform.openai.com/api-keys
  - Set: `OPENAI_API_KEY=your_key`

- **Groq** (Fastest)
  - Get key: https://console.groq.com/keys
  - Set: `GROQ_API_KEY=your_key`

## 📊 What the App Does

### Input
- Patient case description (chief complaint, history, vitals, exam)

### Processing
1. **Orchestrator** analyzes case and coordinates specialists
2. **Specialist Agents** (cardiology, neurology, etc.) provide expert analysis
3. **EXAIM** monitors all agent activity in real-time:
   - **Token Gate**: Buffers agent output
   - **Buffer Agent**: Decides when to summarize
   - **Summarizer**: Generates clinical summaries

### Output (Side-by-Side)

**Left Panel - Raw Traces:**
- Complete agent reasoning
- All specialist outputs
- Full diagnostic thought process
- Typically 10,000+ tokens

**Right Panel - EXAIM Summaries:**
- Compressed clinical insights
- Status/Action, Key Findings, Differentials
- Recommendations and next steps
- Typically 500-1,500 tokens (85-95% compression)

## 💡 Key Features

### Real-time Streaming
- See agent reasoning as it happens
- Summaries generated dynamically
- No waiting for full completion

### Intelligent Compression
- Removes redundancy
- Preserves critical information
- Maintains clinical accuracy
- Structured output format

### Transparency
- Both raw and compressed views
- Full auditability
- Compare compression quality
- Educational tool

## 🎯 Use Cases

### Research & Development
- Study multi-agent compression techniques
- Evaluate clinical AI systems
- Benchmark summarization quality

### Education
- Demonstrate AI clinical reasoning
- Show explainability techniques
- Teaching tool for medical AI

### Demonstration
- Showcase EXAIM capabilities
- Clinical decision support prototype
- Stakeholder presentations

## ⚠️ Important Notes

### Not for Clinical Use
- Research prototype only
- Do not enter real patient data
- Not FDA approved
- For demonstration purposes

### Performance
- Processing takes 30-60 seconds
- Multiple LLM API calls
- Requires stable internet
- API rate limits may apply

### Cost
- ~$0.05-0.15 per case (Gemini)
- Free tier available for testing
- Monitor your API usage

## 📚 Documentation

- **`GRADIO_DEPLOYMENT.md`** - Full deployment guide
- **`README_GRADIO.md`** - Hugging Face Spaces README
- **`docs/DOCUMENTATION.md`** - EXAIM technical docs

## 🐛 Troubleshooting

### "API key not found"
→ Check .env file or HF Spaces secrets

### "Module not found"
→ Ensure all directories copied (exaim_core, demos, infra)

### Slow performance
→ Use faster models (Groq) or check API rate limits

### Out of memory
→ Reduce concurrent users or upgrade HF tier

## 🤝 Support

- GitHub Issues: Report bugs
- Discussions: Feature requests
- Email: Questions and feedback

## 📄 License

MIT License - Free for research and educational use

---

**Created:** February 2026
**Version:** 1.0
**Status:** ✅ Ready for deployment

Happy deploying! 🚀
