# Changelog

All notable changes to LatentGate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.4.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.4.0
[0.3.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.3.0
[0.2.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.2.0
[0.1.0]: https://github.com/KathanModh259/latent-gate/releases/tag/v0.1.0
