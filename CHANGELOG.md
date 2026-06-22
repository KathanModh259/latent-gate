# Changelog

All notable changes to LatentGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.5.3]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.5.3
[0.5.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.5.0
[0.4.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.4.0
[0.3.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.3.0
[0.2.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.2.0
[0.1.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.1.0
