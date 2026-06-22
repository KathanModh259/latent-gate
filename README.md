<div align="center">

# LatentGate

### *Process Locally. Send Smart. Pay Less.*

**A VL-JEPA-inspired pipeline that compresses images, text, conversations, and RAG documents locally via Ollama, then sends only compact semantic payloads to any LLM API — cutting token costs by ~80%.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.5.3-orange.svg)](CHANGELOG.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black.svg)](https://ollama.com)
[![MCP](https://img.shields.io/badge/MCP-supported-purple.svg)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-62%20passed-brightgreen.svg)](tests/)

[**Quick Start**](#-quick-start) | [**Python API**](#python-api) | [**REST API**](#rest-api) | [**AI Tool Integrations**](#-ai-coding-tool-integration-mcp) | [**Benchmarks**](#-cost-benchmarks) | [**Contributing**](#-contributing)

</div>

---

## The Problem

Every time you send an image or long prompt to GPT-4o / Claude / Gemini, you burn 1,000+ tokens on processing that could happen locally for free.

```
Traditional:  Image -> Cloud LLM (1,200 tokens) -> Answer
LatentGate:   Image -> Local Ollama (FREE) -> Cloud LLM (200 tokens) -> Answer
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Local-First** | Vision and text compression runs on Ollama (free, no API key needed) |
| **~80% Token Savings** | Send ~200 tokens instead of ~1,200 for image queries |
| **MCP Server** | Works with Claude Desktop, Cursor, Cline, Continue, Zed |
| **Selective Decoding** | For video, only call API when scene changes (~2.85x fewer calls) with cosine similarity |
| **Text Compression** | Long prompts, conversations, RAG docs compressed locally |
| **Speed Optimized** | Connection pooling, model preloading, parallel processing |
| **Multi-Provider** | OpenAI, Anthropic, Google, Ollama, or any OpenAI-compatible endpoint |
| **REST API** | FastAPI server for web application integration |
| **Video Processing** | Direct video file input with automatic frame extraction |
| **Cost Tracking** | Persistent cost tracking with SQLite analytics and exportable reports |
| **Async Support** | Non-blocking async methods for FastAPI, aiohttp, etc. |
| **Streaming Responses** | Stream responses from remote LLMs |
| **Config Persistence** | YAML/TOML config files with environment variable overrides |
| **Structured Logging** | JSON-formatted logging with rotation and correlation IDs |
| **Docker Support** | Dockerfile and docker-compose for easy deployment |
| **Plugin System** | Custom processors for domain-specific compression |
| **Multi-Language** | Support for 30+ languages with automatic detection |

---

## Quick Start

### Install

```bash
# Core install
pip install latent-gate

# With MCP server (for Claude Desktop, Cursor, Cline, etc.)
pip install latent-gate[mcp]

# With API server (for web applications)
pip install latent-gate[api]

# With video processing
pip install latent-gate[video]

# With embedding-based similarity (more accurate selective decoding)
pip install latent-gate[embeddings]

# With all features
pip install latent-gate[all]
```

### Pull Ollama Models

```bash
ollama pull llava:7b      # Vision model (required for image queries)
ollama pull llama3:8b     # Text model (required for text compression & prediction)
```

### CLI Usage

```bash
# Image query
python -m latent_gate photo.jpg "What is in this image?" --provider ollama -v

# Text compression
python -m latent_gate --text "Your long prompt here..." --provider ollama -v

# Text from file
python -m latent_gate --text-file prompt.txt --provider openai -v

# Image + Text combined
python -m latent_gate photo.jpg "Analyze" --text "Extra context..." -v

# Full JSON output
python -m latent_gate photo.jpg "Describe" --json -v

# Start API server
latent-gate-api
```

---

## Python API

### Image Query

```python
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    vision_model="llava:7b",
    predictor_model="llama3:8b",
    remote_provider="openai",
    remote_model="gpt-4o-mini",
)

with LatentGatePipeline(config) as pipeline:
    result = pipeline.query("photo.jpg", "What is in this image?")

    print(result["answer"])
    print(f"Tokens sent: ~{result['tokens_estimated']}")
    print(f"Timing: {result['timing']}")
```

### Text Compression

```python
# Long prompt compression
result = pipeline.query_text("Your 500-word prompt here...", mode="auto")

# Conversation history compression
messages = [
    {"role": "user", "content": "Help me with Kubernetes setup"},
    {"role": "assistant", "content": "Sure! What's your target configuration?"},
    {"role": "user", "content": "3 nodes, t3.large, us-east-1 with autoscaling"},
]
result = pipeline.query_conversation(messages, "Now give me the setup commands")

# RAG document compression
documents = ["doc1 text...", "doc2 text...", "doc3 text..."]
result = pipeline.query_documents(documents, "How do I implement JWT refresh?")

# Universal (auto-detect input type)
result = pipeline.query_universal(text="Explain this code...", image="screenshot.png")
```

### Batch Processing

```python
# Sequential with selective decoding (skips redundant API calls)
results = pipeline.query_batch(image_paths, "Describe each scene")

# Parallel processing
results = pipeline.query_batch(image_paths, "Describe each scene", parallel=True, max_workers=4)

# Text batch
results = pipeline.query_batch_texts(text_list, question="Summarize each")
```

### Streaming

```python
# Stream image query
for token in pipeline.query_stream("photo.jpg", "Describe this"):
    print(token, end="", flush=True)

# Stream text query
for token in pipeline.query_text_stream("Long prompt...", mode="compress"):
    print(token, end="", flush=True)
```

---

## REST API

### Start Server

```bash
# Default (0.0.0.0:8000)
latent-gate-api

# Custom host/port
LATENTGATE_HOST=127.0.0.1 LATENTGATE_PORT=9000 latent-gate-api
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (Ollama connection status) |
| `GET` | `/stats` | Session usage statistics |
| `POST` | `/query/image` | Image query |
| `POST` | `/query/text` | Text compression |
| `POST` | `/query/conversation` | Conversation compression |
| `POST` | `/query/documents` | RAG document compression |
| `POST` | `/query/universal` | Auto-detect input type |
| `POST` | `/query/image/upload` | Upload image for query |

### Example Requests

```python
import requests

# Image query
response = requests.post("http://localhost:8000/query/image", json={
    "image_path": "photo.jpg",
    "question": "What is in this image?"
})

# Text query
response = requests.post("http://localhost:8000/query/text", json={
    "text": "Your long prompt here...",
    "question": "Summarize this",
    "mode": "auto"  # auto | compress | summarize | condense | code
})

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())  # {"status": "healthy", "ollama_connected": true, ...}
```

---

## Async Support

```python
import asyncio
from latent_gate import AsyncLatentGatePipeline, PipelineConfig

async def main():
    async with AsyncLatentGatePipeline() as pipeline:
        # Single queries
        result = await pipeline.query("photo.jpg", "What is this?")
        result = await pipeline.query_text("Long prompt...")

        # Concurrent batch processing
        results = await pipeline.query_many_images(
            ["img1.jpg", "img2.jpg", "img3.jpg"],
            "Describe each image",
            max_concurrent=3,
        )

asyncio.run(main())
```

---

## Video Processing

```python
from latent_gate import VideoProcessor, VideoConfig

video_config = VideoConfig(
    fps=1.0,            # Extract 1 frame per second
    max_frames=100,     # Max frames to process
    quality=95,         # JPEG quality
    resize_width=640,   # Resize frames (saves processing time)
)

with VideoProcessor(config, video_config) as processor:
    result = processor.process_video("video.mp4", "Describe the action")

    print(f"Frames processed: {result['total_frames']}")
    print(f"Unique scenes: {result['statistics']['unique_scenes']}")
    print(f"Skip rate: {result['statistics']['skip_rate']}")
```

---

## Configuration

### Config File

```yaml
# latentgate.yaml
ollama_base_url: http://localhost:11434
vision_model: llava:7b
predictor_model: llama3:8b
remote_provider: openai
remote_model: gpt-4o-mini
selective_decoding: true
similarity_threshold: 0.85
use_embeddings: true
enable_caching: true
temperature: 0.1
request_timeout: 120
```

```python
from latent_gate import get_config, LatentGatePipeline

config = get_config("latentgate.yaml")
with LatentGatePipeline(config) as pipeline:
    result = pipeline.query("photo.jpg", "Describe this")
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `GOOGLE_API_KEY` | Google API key | - |
| `LATENTGATE_REMOTE_PROVIDER` | Override remote provider | `openai` |
| `LATENTGATE_REMOTE_MODEL` | Override remote model | `gpt-4o-mini` |
| `LATENTGATE_VISION_MODEL` | Override vision model | `llava:7b` |
| `LATENTGATE_LOG_LEVEL` | Log level | `INFO` |
| `LATENTGATE_LOG_FILE` | Log file path | - |
| `LATENTGATE_LOG_JSON` | JSON log format | `false` |

### Save Config

```python
from latent_gate import PipelineConfig, save_config

config = PipelineConfig(remote_provider="anthropic", remote_model="claude-sonnet-4-20250514")
save_config(config, "my_config.yaml")
```

---

## Docker

```bash
# Start with Docker Compose (includes Ollama)
docker-compose up -d

# Or build and run manually
docker build -t latent-gate .
docker run -p 8000:8000 latent-gate
```

The docker-compose setup includes:
- **latent-gate** API server (port 8000)
- **Ollama** local LLM server (port 11434)
- **ollama-init** container that auto-pulls required models

---

## AI Coding Tool Integration (MCP)

LatentGate works as a Model Context Protocol (MCP) server with every major AI coding tool. Your AI assistant automatically compresses images, long prompts, and documents — saving ~80% on tokens.

### Supported Tools

| Tool | Status | Setup |
|------|--------|-------|
| VS Code / Copilot | Supported | [Extension](https://marketplace.visualstudio.com/items?itemName=KathanModh259.latent-gate-vscode) |
| Claude Desktop | Supported | MCP Config |
| Claude Code (CLI) | Supported | Skill |
| Cursor | Supported | MCP Config |
| Cline (VS Code) | Supported | MCP Config |
| Continue.dev | Supported | MCP Config |
| Zed Editor | Supported | MCP Config |

### VS Code Extension

```bash
code --install-extension KathanModh.latent-gate
```

Features:
- Right-click any image to compress with LatentGate
- Select text and press `Ctrl+Shift+Alt+C` to compress
- Cost dashboard in activity bar
- Auto-configures MCP for Copilot Chat
- Status bar showing token savings

### MCP Setup

```bash
pip install latent-gate[mcp]
ollama pull llava:7b
ollama pull llama3:8b
```

Add to your tool's MCP config:

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

### MCP Tools

| Tool | When AI Uses It |
|------|-----------------|
| `compress_image` | Before analyzing any image |
| `compress_text` | For prompts longer than ~500 tokens |
| `compress_conversation` | When chat history is large |
| `compress_documents` | For RAG queries |
| `get_stats` | To check session savings |

See `integrations/` folder for detailed setup guides per tool.

---

## Speed Optimizations

| Optimization | What It Does | Impact |
|-------------|--------------|--------|
| Connection Pooling | Reuses HTTP connections via `requests.Session` | ~30-50% faster per call |
| Model Preloading | Warms up Ollama models on init (`keep_alive`) | Eliminates 5-15s cold start |
| Shorter Prompts | Optimized extraction prompts produce fewer output tokens | ~20% faster generation |
| 3-Tier JSON Parsing | Fast parse, extract from text, LLM fallback | Avoids slow LLM call 90% of time |
| Parallel Processing | Image and text processed simultaneously via ThreadPool | ~40% faster combined queries |
| Content-Hash Caching | Disk cache for repeated images | Instant on cache hit |
| Selective Decoding | Cosine similarity skips redundant API calls | ~2.85x fewer calls |

---

## Cost Benchmarks

### Image Queries (by provider)

| Provider | Raw Image Tokens | LatentGate Tokens | Savings |
|----------|:----------------:|:-----------------:|:-------:|
| OpenAI GPT-4o (high detail) | ~1,105 | ~150 | ~86% |
| Claude 3.5 Sonnet (1MP image) | ~1,334 | ~150 | ~89% |
| Gemini 2.0 Flash | ~258 | ~150 | ~42% |

### Text and Other Modes

| Scenario | Traditional | LatentGate | Savings |
|----------|:-----------:|:----------:|:-------:|
| Long text prompt | ~800 | ~120 | ~85% |
| Conversation (10 turns) | ~2,500 | ~350 | ~86% |
| RAG documents (3 docs) | ~3,000 | ~450 | ~85% |
| Video stream (1 min)* | varies | ~2.85x fewer calls | ~65% |

*With selective decoding

### At Scale (10,000 image queries with gpt-4o-mini)

| | Traditional | LatentGate | Savings |
|--|:-----------:|:----------:|:-------:|
| Input tokens | 12,000,000 | 2,000,000 | 10M tokens |
| Cost | $1.80 | $0.30 | $1.50 (83%) |

---

## Cost Tracking

```python
from latent_gate import CostTracker

tracker = CostTracker()
tracker.record_usage(
    query_type="image",
    provider="openai",
    model="gpt-4o-mini",
    input_tokens=150,
    output_tokens=200,
    tokens_saved=1000,
    compression_ratio=6.7,
    latency_ms=1500,
)

# Session statistics
stats = tracker.get_session_statistics()
print(f"Total cost: ${stats['total_cost']:.4f}")
print(f"Tokens saved: {stats['total_tokens_saved']}")

# Cost projection
projection = tracker.get_cost_projection(
    daily_queries=1000,
    provider="openai",
    model="gpt-4o-mini"
)
print(f"Monthly savings: ${projection['savings']['monthly']:.2f}")

# Export report
tracker.export_report("usage_report.json", fmt="json")
tracker.export_report("usage_report.csv", fmt="csv")
```

---

## Multi-Language Support

```python
from latent_gate import detect_language, MultiLanguageProcessor

# Detect language
lang = detect_language("Esto es un texto en español")
print(f"Detected: {lang.name} ({lang.confidence:.0%})")

# Process with auto-translation to English
processor = MultiLanguageProcessor()
text, lang_info = processor.process("Texto en español para analizar")
print(f"Language: {lang_info.name}, Translated: {text[:100]}...")
```

---

## Project Structure

```
latent-gate/
├── latent_gate/
│   ├── __init__.py           # Package exports and version
│   ├── config.py             # PipelineConfig dataclass
│   ├── config_loader.py      # YAML/TOML/JSON config loading
│   ├── payload.py            # SemanticPayload (compact representation)
│   ├── text_processor.py     # TextPayload + TextProcessor (local compression)
│   ├── local_processor.py    # X-Encoder + Predictor (Ollama vision pipeline)
│   ├── remote_decoder.py     # Y-Decoder (OpenAI, Anthropic, Google, Ollama)
│   ├── selective_decoder.py  # Cosine/Jaccard similarity for skip decisions
│   ├── fast_client.py        # Connection pooling + model preloading
│   ├── cache.py              # Content-hash disk cache
│   ├── pipeline.py           # LatentGatePipeline (main orchestrator)
│   ├── async_pipeline.py     # AsyncLatentGatePipeline
│   ├── video_processor.py    # Video frame extraction + batch processing
│   ├── cost_tracker.py       # SQLite-based cost analytics
│   ├── mcp_server.py         # MCP server (Model Context Protocol)
│   ├── api_server.py         # FastAPI REST server
│   ├── cli.py                # Command-line interface
│   ├── logging_config.py     # Structured logging with rotation
│   ├── plugin_system.py      # Custom processor plugins
│   └── multilang.py          # Multi-language detection and translation
├── integrations/
│   ├── mcp_server/           # Standalone MCP server
│   ├── claude_code_skill/    # Claude Code skill + scripts
│   ├── cursor/               # Cursor MCP config
│   ├── continue_dev/         # Continue.dev config
│   └── openai_functions/     # OpenAI/Anthropic function schemas
├── examples/
│   ├── basic_usage.py
│   ├── text_compression.py
│   ├── advanced_features.py
│   ├── video_streaming.py
│   └── ...
├── tests/                    # 62 tests (unit + integration)
├── vscode-extension/         # VS Code extension source
├── docs/
├── .github/workflows/        # CI + publish workflows
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Setup

```bash
git clone https://github.com/KathanModh259/latent-gate.git
cd latent-gate
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\Activate.ps1      # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest tests/ -v
```

### Priority Areas

- Additional vision model support (Florence-2, InternVL, Qwen-VL)
- Custom similarity plugins for domain-specific use cases
- WebSocket support for real-time streaming
- Advanced cost analytics and optimization suggestions
- Plugin development for specialized industries
- Test coverage improvements
- Documentation and examples

---

## Citation

```bibtex
@software{latentgate2026,
  author  = {Kathan Modh},
  title   = {LatentGate: Local-First Vision-Language Pipeline Inspired by VL-JEPA},
  year    = {2026},
  version = {0.5.3},
  url     = {https://github.com/KathanModh259/latent-gate}
}
```

Inspired by [VL-JEPA](https://arxiv.org/abs/2512.10942) (Meta FAIR, 2025).

---

## License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

**Built by [Kathan Modh](https://github.com/KathanModh259)**

*Process locally. Send smart. Pay less.*

</div>
