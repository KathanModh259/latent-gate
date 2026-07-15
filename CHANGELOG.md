# Changelog

All notable changes to LatentGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.4] - 2026-07-13

### Added

- **Model routing layer** — `text_fast_model` (phi3:mini), `text_smart_model` (qwen2:7b), `embedding_model` (nomic-embed-text); `get_model_for_task()` and `get_fallback_chain()` for intelligent model selection
- **Complexity detection** — `_is_complex(text)` routes long/code-heavy prompts to smart model automatically
- **Fallback compression** — Algorithmic text compression when Ollama is unavailable (`_fallback_compress`)
- **VS Code extension Python bundling** — `latent_gate/` source included in VSIX for standalone worker operation
- **CI/CD** — Frontend build, VS Code extension packaging, Docker image publishing
- **Deployment configs** — `vercel.json` for website, Helm chart, Homebrew/Winget formulas

### Changed

- **Config** — `predictor_model` deprecated; backward-compat maps to `text_fast_model`
- **Docker** — `ollama-init` now pulls all 5 models (llava, phi3, qwen2, nomic-embed, llama3)
- **Security** — CSP headers tightened (removed `unsafe-eval`); XSS fixes in web UI
- **UI** — Loading spinner, progress bar, copy-to-clipboard button, input validation (50k char limit)
- **Packaging** — Cleaned up repo structure (removed docs/examples/marketing from main branch)

### Fixed

- `python -m build` compatibility with setuptools 83+ (license expression vs classifier conflict)
- VS Code extension worker path resolution (now uses extension directory, not parent)

## [1.2.3] - 2026-07-13

### Fixed

- VS Code extension Python worker crash — `latent_gate/` now bundled in VSIX
- Worker PYTHONPATH resolution for production installs

## [1.2.2] - 2026-07-13

### Changed

- Model routing layer integrated across all processors
- Environment variables, docker-compose, and Dockerfile updated for new models
- All builds (website, PyPI, VSIX) verified passing

## [1.2.1] - 2026-06-24

### Changed

- **VS Code extension redesign** — Dashboard now has inline text input with Ctrl+Enter to compress, live loading states, and proper status display
- **VS Code tools tab** — Cleaner layout with shortcut rows, tool descriptions with savings percentages
- **CLI speed** — `--compress-only` bypasses full pipeline, uses TextProcessor directly (~3x faster)
- **Text compression** — Removed input truncation limits, supports any prompt length
- **Ollama optimization** — Added `keep_alive` and `num_ctx` for faster inference

### Fixed

- VS Code webview broken `$(icon)` syntax (replaced with Unicode)
- Dashboard stats displaying wrong values (apiCalls instead of tokensSaved)
- CLI PowerShell angle bracket handling
- `compress_prompt` output truncation at 200 tokens

## [1.2.0] - 2026-06-24

### Fixed

- **Config save/load** — `_config_to_dict()` now includes `offline_first`, `offline_model`, `adaptive_compression`, `target_token_budget` (were silently dropped)
- **Adaptive compression** — `complexity`/`max_tokens` now actually passed to `TextProcessor.compress()` instead of being computed and discarded
- **Root logger hijack** — `pipeline.py` no longer calls `logging.basicConfig()`; uses library-scoped logger instead
- **Resource leaks** — Removed unused `ThreadPoolExecutor` from `LocalProcessor` and `AsyncLatentGatePipeline`
- **Dead code** — Removed unused `_model = None` from `selective_decoder.py`

### Changed

- **Refactored `remote_decoder.py`** — Extracted `OpenAICompatibleDecoder` base class; 9 decoder classes now share streaming/SSE logic (742 → 360 lines, -51%)
- **`TextProcessor` uses `FastClient`** — Now benefits from connection pooling and `keep_alive` instead of raw `requests.post()`
- **Consolidated examples** — Merged 5 redundant example files into single `providers.py`
- **Removed redundant files** — `requirements.txt`, `requirements-dev.txt`, `publish.py`, `MANIFEST.in`, duplicate MCP server
- **Updated `.gitignore`** — Added `.latentgate_costs.db`, `*.db` patterns

## [1.1.0] - 2026-06-23

### Added

- **LangChain Integration** — Use LatentGate as a LangChain LLM and Tool for chains, agents, and pipelines
- **LlamaIndex Integration** — Use LatentGate as a LlamaIndex QueryEngine and Retriever
- **Groq Provider** — Fast inference via Groq API (llama-3.1-8b-instant, mixtral-8x7b)
- **DeepSeek Provider** — DeepSeek Chat and Coder models
- **Together AI Provider** — Together AI inference (Llama-3, Mixtral)
- **Azure OpenAI Provider** — Azure OpenAI deployments
- **AWS Bedrock Provider** — AWS Bedrock models via boto3
- **Offline-First Mode** — Full pipeline without cloud API (local Ollama answering)
- **Adaptive Compression** — Dynamically adjust compression based on query complexity
- **Semantic Deduplication** — Skip similar queries in batch processing
- **Streaming Cost Estimation** — `estimate_cost()` method for real-time cost projections
- **GitHub Pages Site** — Landing page with features, providers, and community links
- **Codecov Integration** — Code coverage reporting in CI
- **Community Section** — Discord, GitHub Discussions, Twitter links in README
- **Helm Chart** — Kubernetes deployment with Ollama sidecar, autoscaling, and secrets management
- **Homebrew Formula** — macOS installation via Homebrew
- **Winget Manifest** — Windows installation via Windows Package Manager
- **Awesome Lists** — Submission guide for awesome-ollama, awesome-mcp, awesome-local-ai, awesome-python, awesome-langchain, awesome-llamaindex
- **Product Hunt Guide** — Launch preparation and strategy guide

### Changed

- Updated multi-provider support in README and features table
- Added new optional dependencies: `langchain`, `bedrock`
- Updated cost tracker with pricing for Groq, DeepSeek, Together, Azure, Bedrock
- Added provider documentation to integrations README

## [1.0.0] - 2026-06-22

### Fixed

- **`query_universal` parallel text compression** — Fixed missing `question` parameter in parallel text processing path
- **Parallel batch result ordering** — Fixed sorting bug where error results would cluster at index 0
- **`query_image_upload` missing null check** — Added pipeline null check to prevent `UnboundLocalError`
- **`api_server.py` missing fastapi import guard** — Graceful error message when fastapi not installed
- **`api_server.py` dead code removal** — Removed unused `PipelineConfig` instantiation
- **`cost_tracker.py` SQLite column indices** — Fixed all 15+ off-by-one indices that caused `TypeError` on `get_statistics()`
- **`logging_config.py` variable shadowing** — Fixed local `logger` shadowing module-level `logger`
- **`cost_tracker.py` and `config_loader.py` parameter shadowing** — Renamed `format` to `fmt` to avoid built-in shadowing
- **`Dockerfile` missing API dependencies** — Changed to install `[api]` extras so API server works in Docker
- **`docker-compose.dev.yml` broken build target** — Fixed to use full build with API dependencies
- **CI workflow missing optional dependencies** — Changed to install `[all]` extras for proper testing
- **`MANIFEST.in` missing `pyproject.toml`** — Added for correct wheel builds
- **README navigation links** — Fixed broken anchor links with extra `-` prefix
- **README VS Code extension ID** — Fixed publisher name mismatch
- **README Video Processing example** — Added missing `PipelineConfig` setup

### Changed

- Bumped all version references from 0.5.x to 1.0.0
- Updated README with PowerShell/CMD env var syntax
- Updated README with production-ready documentation

## [0.5.3] - 2026-06-22

### Fixed

- **`query_universal` parallel text compression** — Fixed missing `question` parameter in parallel text processing path, which prevented context-aware compression (condense mode)
- **Parallel batch result ordering** — Fixed sorting bug where error results would all cluster at index 0 instead of preserving original order
- **`query_image_upload` missing null check** — Added pipeline null check in the image upload endpoint to prevent `UnboundLocalError`
- **Dead code removal** — Removed unused `PipelineConfig` instantiation in `/query/image` endpoint
- **Variable shadowing in `logging_config`** — Fixed local `logger` variable shadowing module-level `logger`
- **Parameter shadowing in `cost_tracker` and `config_loader`** — Renamed `format` parameter to `fmt` to avoid shadowing built-in

## [0.5.0] - 2026-06-22

### Added

- **VSCode Extension** — Full-featured extension with commands, webview panels, and MCP auto-configuration
- **True Embedding Similarity** — Replaced Jaccard similarity with cosine similarity using sentence-transformers for more accurate scene change detection
- **FastAPI Server Wrapper** — REST API server for web application integration
- **Direct Video File Input** — Auto frame extraction from video files with configurable FPS
- **Cost Tracking Dashboard** — Persistent cost tracking with SQLite analytics and exportable reports
- **Async Support** — Async versions of core pipeline methods for non-blocking operations
- **Batch Processing Optimization** — Parallelized batch image processing
- **Streaming Responses** — Support for streaming from remote LLMs
- **Configuration Persistence** — YAML/TOML config files with environment variable overrides
- **Structured Logging** — JSON-formatted logging with log rotation
- **Docker Support** — Dockerfile and docker-compose for easy deployment
- **Plugin System** — Custom processor plugins for domain-specific compression
- **Multi-language Support** — Non-English text compression support

### Changed

- Upgraded selective decoding to use cosine similarity when sentence-transformers is available
- Added `use_embeddings` config option for selective decoding
- Updated README with new features and expanded architecture documentation

## [0.4.0] - 2026-06-17

### Added

- **MCP Server** — Universal Model Context Protocol server that works with Claude Desktop, Cursor, Cline, Continue.dev, Zed, and Claude Code (CLI)
- **Claude Code Skill** — Drop-in skill with `SKILL.md` plus compression scripts for automatic token reduction
- **Cursor integration** — `.cursorrules` template and MCP config
- **Continue.dev integration** — Custom commands plus MCP setup
- **OpenAI / Anthropic Function Schemas** — Ready-to-use tool definitions for custom agents
- New `integrations/` folder with detailed setup guides for each AI coding tool
- New optional install: `pip install latent-gate[mcp]`
- New `all` install extra: `pip install latent-gate[all]`

### MCP Tools Exposed

- `compress_image` — Image to ~150 token semantic payload
- `compress_text` — Long prompt compression
- `compress_conversation` — Multi-turn chat compression
- `compress_documents` — RAG document condensation
- `get_stats` — Session token savings statistics

### Changed

- Bumped development status to Beta (4)
- Updated README with MCP integration section, AI tool table, and per-provider benchmarks
- Added `mcp` to optional extras in `pyproject.toml`

## [0.3.0] - 2026-06-15

### Added

- Connection pooling via `FastClient` (~30-50% faster HTTP calls)
- Model preloading with `keep_alive` (eliminates 5-15s cold start)
- 3-tier JSON parsing (avoids slow LLM fallback 90%+ of the time)
- Parallel processing for `query_universal()` (~40% faster)
- Timing metrics in result (`timing.local_ms`, `remote_ms`, `total_ms`)
- Context manager support (`with LatentGatePipeline() as pipeline:`)

### Changed

- Optimized extraction prompts (~20% fewer output tokens)
- Shared HTTP client across all components

## [0.2.0] - 2026-06-15

### Added

- **Text Compression** — `query_text()` for compressing long prompts locally
- **Conversation Compression** — `query_conversation()` for chat history
- **RAG Document Compression** — `query_documents()` for retrieved docs
- **Universal Mode** — `query_universal()` auto-detects input type
- CLI flags: `--text`, `--text-file`, `--mode`
- New `TextProcessor` and `TextPayload` classes

## [0.1.0] - 2026-06-15

### Added

- Initial release
- Image compression via Ollama (X-Encoder plus Predictor stages)
- Remote decoders for OpenAI, Anthropic, Google, Ollama
- Selective decoding for video streams (~2.85x fewer API calls)
- Content-hash caching
- CLI plus Python API

[1.0.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v1.0.0
[0.5.3]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.5.3
[0.5.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.5.0
[0.4.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.4.0
[0.3.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.3.0
[0.2.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.2.0
[0.1.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.1.0
