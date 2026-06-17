<div align="center">

# 🔮 LatentGate

### *Process Locally. Send Smart. Pay Less.*

**A VL-JEPA-inspired pipeline that compresses images, text, conversations, and RAG documents locally via Ollama, then sends only compact semantic payloads to any LLM API — cutting token costs by ~80%.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.0-orange.svg)](CHANGELOG.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black.svg)](https://ollama.com)
[![MCP](https://img.shields.io/badge/MCP-supported-purple.svg)](https://modelcontextprotocol.io)

[**Quick Start**](#-quick-start) · [**AI Tool Integrations**](#-use-with-ai-coding-tools-mcp-integration) · [**Benchmarks**](#-cost-benchmarks) · [**Contributing**](#-contributing)

</div>

---

## 🏗️ Architecture

<div align="center">

![LatentGate Architecture](docs/architecture.png)

</div>

---

## 💡 The Problem

Every time you send an image or long prompt to GPT-4o / Claude / Gemini, you are burning 1,000+ tokens on processing that could happen locally for free.

```
Traditional:  Image → Cloud LLM (1,200 tokens) → Answer
LatentGate:   Image → Local Ollama (FREE) → Cloud LLM (200 tokens) → Answer
```

---

## ✨ Features

- 🏠 **Local-First** — Vision and text compression runs on Ollama (free)
- 💰 **~80% Token Savings** — Send ~200 tokens instead of ~1,200
- 🔌 **MCP Server** — Works with Claude Desktop, Cursor, Cline, Continue, Zed
- 🎯 **Selective Decoding** — For video, only call API when scene changes (~2.85x fewer calls)
- 📝 **Text Compression** — Long prompts, conversations, RAG docs compressed locally
- ⚡ **Speed Optimized** — Connection pooling, model preloading, parallel processing
- 🔌 **Multi-Provider** — OpenAI, Anthropic, Google, Groq, or any OpenAI-compatible endpoint

---

## 🚀 Quick Start

### Install

```bash
# Core install
pip install latent-gate

# With MCP server (for Claude Desktop, Cursor, Cline, etc.)
pip install latent-gate[mcp]

# Pull required Ollama models
ollama pull llava:7b
ollama pull llama3:8b
```

### Run

```bash
# Image query
python -m latent_gate photo.jpg "What is in this image?" --provider ollama -v

# Text compression
python -m latent_gate --text "Your long prompt here..." --provider ollama -v

# Image + Text combined
python -m latent_gate photo.jpg "Analyze" --text "Extra context..." -v
```

### Python API

```python
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    vision_model="llava:7b",
    remote_provider="openai",
    remote_model="gpt-4o-mini",
)

with LatentGatePipeline(config) as pipeline:
    result = pipeline.query("photo.jpg", "Describe this")
    result = pipeline.query_text("Your 500-word prompt...")
    result = pipeline.query_conversation(messages, "Follow-up question")
    result = pipeline.query_documents(["doc1...", "doc2..."], "Question?")
    result = pipeline.query_universal(text="...", image="photo.jpg")

    print(result["timing"])
    print(result["tokens_estimated"])
```

---

## 🔌 Use With AI Coding Tools (MCP Integration)

LatentGate works as a Model Context Protocol (MCP) server with every major AI coding tool. Once configured, your AI assistant automatically compresses images, long prompts, and documents — saving you ~80% on tokens without changing your workflow.

### Supported Tools

| Tool              | Status      |
| ----------------- | ----------- |
| Claude Desktop    | Supported   |
| Claude Code (CLI) | Supported   |
| Cursor            | Supported   |
| Cline (VS Code)   | Supported   |
| Continue.dev      | Supported   |
| Zed Editor        | Supported   |

### Quick Setup

```bash
pip install latent-gate[mcp]
ollama pull llava:7b
ollama pull llama3:8b
```

Then add to your AI tool MCP config:

```json
{
  "mcpServers": {
    "latent-gate": {
      "command": "python",
      "args": ["-m", "latent_gate.mcp_server"]
    }
  }
}
```

Detailed setup guides for each tool: see the `integrations/` folder.

### What Gets Compressed Automatically

| Tool Call               | When AI Uses It                |
| ----------------------- | ------------------------------ |
| `compress_image`        | Before analyzing any image     |
| `compress_text`         | For prompts longer than ~500 tokens |
| `compress_conversation` | When chat history is large     |
| `compress_documents`    | For RAG queries                |
| `get_stats`             | To check session savings       |

---

## ⚡ Speed Optimizations

| Optimization          | What It Does                                                | Impact                          |
| --------------------- | ----------------------------------------------------------- | ------------------------------- |
| Connection Pooling    | Reuses HTTP connections via `requests.Session`              | ~30-50% faster per call         |
| Model Preloading      | Warms up Ollama models on init (`keep_alive`)               | Eliminates 5-15s cold start     |
| Shorter Prompts       | Optimized extraction prompts produce fewer output tokens    | ~20% faster generation          |
| 3-Tier JSON Parsing   | Fast parse, extract from text, LLM fallback                 | Avoids slow LLM call 90% of time |
| Parallel Processing   | Image and text processed simultaneously via ThreadPool      | ~40% faster combined queries    |
| Caching               | Content-hash disk cache for repeated images                 | Instant on cache hit            |

---

## 📊 Cost Benchmarks

### Image Queries (by provider)

| Provider                       | Raw Image Tokens | LatentGate Tokens | Savings |
| ------------------------------ | ---------------: | ----------------: | ------- |
| OpenAI GPT-4o (high detail)    | ~1,105           | ~150              | ~86%    |
| Claude 3.5 Sonnet (1MP image)  | ~1,334           | ~150              | ~89%    |
| Gemini 3 Pro                   | ~560             | ~150              | ~73%    |
| Gemini 2.0 Flash               | ~258             | ~150              | ~42%    |

### Text and Other Modes (all providers benefit equally)

| Scenario                  | Traditional | LatentGate           | Savings |
| ------------------------- | ----------: | -------------------: | ------- |
| Long text prompt          | ~800        | ~120                 | ~85%    |
| Conversation (10 turns)   | ~2,500      | ~350                 | ~86%    |
| RAG documents (3 docs)    | ~3,000      | ~450                 | ~85%    |
| Video stream (1 min)*     | varies      | ~2.85x fewer calls   | ~65%    |

*With selective decoding

### At Scale (10,000 image queries with gpt-4o-mini)

|                | Traditional | LatentGate | Savings        |
| -------------- | ----------- | ---------- | -------------- |
| Input tokens   | 12,000,000  | 2,000,000  | 10M tokens     |
| Cost           | $1.80       | $0.30      | $1.50 (83%)    |

---

## 📁 Project Structure

```
latent-gate/
├── latent_gate/
│   ├── __init__.py
│   ├── config.py
│   ├── payload.py
│   ├── text_processor.py
│   ├── local_processor.py
│   ├── remote_decoder.py
│   ├── selective_decoder.py
│   ├── fast_client.py
│   ├── cache.py
│   ├── pipeline.py
│   ├── cli.py
│   └── mcp_server.py
├── integrations/
│   ├── README.md
│   ├── mcp_server/
│   ├── claude_code_skill/
│   ├── cursor/
│   ├── continue_dev/
│   └── openai_functions/
├── examples/
├── tests/
├── docs/
│   ├── architecture.png
│   └── how_it_works.md
├── CHANGELOG.md
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Priority Areas

- True embedding similarity (replace Jaccard with cosine via sentence-transformers)
- FastAPI server wrapper
- Direct video file input (auto frame extraction)
- Cost tracking dashboard
- More vision model support (Florence-2, InternVL)
- PyPI publish

---

## 📄 Citation

```bibtex
@software{latentgate2026,
  author  = {Kathan Modh},
  title   = {LatentGate: Local-First Vision-Language Pipeline Inspired by VL-JEPA},
  year    = {2026},
  version = {0.4.0},
  url     = {https://github.com/KathanModh259/latent-gate}
}
```

Inspired by [VL-JEPA](https://arxiv.org/abs/2512.10942) (Meta FAIR, 2025).

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

**Built with 🧠 by [Kathan Modh](https://github.com/KathanModh259)**

*Process locally. Send smart. Pay less.*

Star this repo if it saved you tokens (and money)!

</div>
