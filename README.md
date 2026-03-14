---
title: CT Orchestrator
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: frontend/app.py
pinned: false
license: apache-2.0
short_description: AI-powered creative testing for video ads
---

# 🎬 Creative Testing Orchestrator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-00A67E.svg)](https://langchain-ai.github.io/langgraph/)

A **multi-agent AI system** for automating creative testing workflows in media agencies. Analyzes video ads using local vision models, predicts brand lift study outcomes, and generates actionable insights—replacing manual processes that typically cost $10K-$25K per study.

## 🌐 Try the Live Demo

**[▶️ Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/akshargupta84/ct-orchestrator)**

The demo uses pre-scored sample videos so you can explore all features instantly. For live video scoring with your own creatives, follow the local installation guide below.

---

## 🎯 Problem Statement

Brand lift studies are the gold standard for measuring creative effectiveness, but they're:
- **Expensive**: $10K-$25K per study
- **Slow**: 2-4 weeks for results
- **High failure rate**: ~65% of creatives fail to show significant lift

**CT Orchestrator** enables pre-screening of video creatives using AI, providing:
- Pass/fail predictions before committing budget
- Diagnostic scores (attention, brand recall, message clarity, etc.)
- Actionable recommendations for creative optimization

**Potential savings**: ~$150K/year by avoiding tests on creatives predicted to fail.

---

## ✨ Features

### 🤖 Multi-Agent Architecture
- **Planning Agent**: Validates test plans against agency rules
- **Analysis Agent**: Parses results, diagnoses issues, recommends actions
- **Video Analyzer**: Extracts creative features using local vision AI
- **Knowledge Agent**: RAG over rules and past learnings

### 🎥 Video Analysis Pipeline
- Local vision model analysis using **Ollama/LLaVA**
- Frame-by-frame feature extraction (humans, logos, emotions, CTAs)
- **Privacy-first**: No data leaves your machine

### 📊 Predictive Modeling
- Multi-task learning: Pass/fail + 5 diagnostic scores
- Ensemble classifier (Logistic Regression + Random Forest)
- LOOCV validation for honest evaluation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Streamlit)                           │
│   Welcome │ Multi-Agent Hub │ CT Planner │ Results │ Insights │ Admin      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER (LangGraph)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Planning   │  │   Analysis   │  │    Video     │  │  Knowledge   │    │
│  │    Agent     │  │    Agent     │  │   Analyzer   │  │    Agent     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
│   SQLite │ ChromaDB (RAG) │ Video Files │ Pre-scored Cache                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Try the Demo (No Installation)

Visit the **[Live Demo](https://huggingface.co/spaces/akshargupta84/ct-orchestrator)** to explore with pre-scored sample videos.

### Option 2: Run Locally with Full Features

#### Prerequisites

| Requirement | Purpose | Notes |
|-------------|---------|-------|
| **Python 3.10+** | Runtime | Required |
| **[Ollama](https://ollama.ai/)** | Local video analysis | Must be running in background |
| **Anthropic API key** | LLM for agents | [Get one here](https://console.anthropic.com/) |

> ⚠️ **Important**: Ollama must be installed and running locally for video analysis. This is by design — your video data never leaves your machine.

#### Step 1: Install Ollama

| Platform | Installation |
|----------|--------------|
| **macOS** | `brew install ollama` or [download from ollama.ai](https://ollama.ai/) |
| **Windows** | [Download installer from ollama.ai](https://ollama.ai/) |
| **Linux** | `curl -fsSL https://ollama.ai/install.sh \| sh` |

```bash
# Start Ollama and download the vision model
ollama serve
ollama pull llava:13b  # Or llava:7b for less RAM
```

#### Step 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/akshargupta84/ct-orchestrator.git
cd ct-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FULL dependencies (includes Ollama, ChromaDB, etc.)
pip install -r requirements-local.txt

# Set up environment
cp .env.example .env
# Edit .env: Add your ANTHROPIC_API_KEY and set DEMO_MODE=false
```

#### Step 3: Run the Application

```bash
cd frontend
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🎥 Scoring Your Own Videos

This is the core feature of CT Orchestrator. Here's how to analyze your video creatives:

### Quick Python Script

```python
from services.video_ingestion import VideoIngestionService
from services.video_analysis import LocalVisionService

# Initialize services (requires Ollama running)
vision = LocalVisionService()
ingestion = VideoIngestionService(vision_service=vision)

# Analyze a single video
result = ingestion.analyze_video("path/to/your/creative.mp4")

# View results
print(f"Video: {result.filename}")
print(f"Duration: {result.duration_seconds}s")
print(f"Pass Probability: {result.pass_probability:.1%}")
print(f"Recommendation: {result.recommendation}")

# View extracted features
print("\nExtracted Features:")
print(f"  Human presence: {result.human_frame_ratio:.0%} of frames")
print(f"  Logo in first 3s: {result.logo_in_first_3_sec}")
print(f"  Has CTA: {result.has_cta}")
print(f"  Emotional content: {result.has_emotional_content}")

# View diagnostic scores
print("\nDiagnostic Scores:")
for metric, score in result.diagnostics.items():
    print(f"  {metric}: {score}/100")
```

### Batch Processing Multiple Videos

```python
from pathlib import Path
import pandas as pd

# Process all videos in a folder
video_folder = Path("path/to/videos")
results = []

for video_path in video_folder.glob("*.mp4"):
    print(f"Processing: {video_path.name}")
    result = ingestion.analyze_video(str(video_path))
    results.append({
        "filename": result.filename,
        "duration": result.duration_seconds,
        "pass_probability": result.pass_probability,
        "recommendation": result.recommendation,
        "attention_score": result.diagnostics["attention_score"],
        "brand_recall_score": result.diagnostics["brand_recall_score"],
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("video_analysis_results.csv", index=False)
print(f"\nProcessed {len(results)} videos!")
```

---

## 📁 Project Structure

```
ct-orchestrator/
├── frontend/               # Streamlit application
│   ├── app.py              # Main entry point
│   └── pages/              # Multi-page navigation
├── agents/                 # Agent implementations
├── services/               # Business logic
│   ├── video_analysis.py   # Frame analysis with Ollama
│   ├── video_ingestion.py  # Video processing pipeline
│   ├── performance_modeling.py  # ML prediction models
│   └── ...
├── demo_data/              # Pre-scored sample data
├── models/                 # Pydantic data models
├── workflows/              # LangGraph workflow definitions
├── requirements.txt        # Minimal deps (HF-compatible)
├── requirements-local.txt  # Full deps for local dev
└── .env.example
```

---

## 🛠️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEMO_MODE` | Use pre-scored videos (no Ollama needed) | `true` |
| `ANTHROPIC_API_KEY` | Your Anthropic API key | Required for chat |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_VISION_MODEL` | Vision model for analysis | `llava:13b` |

### Demo Mode vs Local Mode

| Feature | Demo Mode (`true`) | Local Mode (`false`) |
|---------|-------------------|---------------------|
| Pre-scored videos | ✅ | ✅ |
| Live video scoring | ❌ | ✅ |
| Requires Ollama | ❌ | ✅ |
| Insights Chat | Limited | Full |

---

## ❓ Troubleshooting

### "Connection refused" or "Ollama not found"

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve
```

### "Model not found"

```bash
ollama pull llava:13b
# Or for less RAM:
ollama pull llava:7b
```

### Video analysis is slow

- First run downloads the model (~8GB)
- Use `llava:7b` for faster (but less accurate) analysis
- Reduce `FRAMES_PER_VIDEO` in `.env`

### Out of memory

- Use `llava:7b` (~8GB RAM) instead of `llava:13b` (~16GB RAM)
- Close other applications

---

## 📊 Model Performance

With synthetic training data (n≈50):

| Metric | Value |
|--------|-------|
| Accuracy | ~70% |
| Precision | ~72% |
| Recall | ~68% |

Performance improves significantly with real brand lift study data.

---

## 🤝 Contributing

Contributions welcome! Please submit a Pull Request.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

Apache License 2.0 - see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [Anthropic](https://anthropic.com) for Claude API
- [Ollama](https://ollama.ai) for local LLM hosting
- [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration
- [Streamlit](https://streamlit.io) for the frontend

---

## 📧 Contact

**Akshar Gupta** - [LinkedIn](https://linkedin.com/in/akshar-g-4093621b) - [GitHub](https://github.com/akshargupta84)
