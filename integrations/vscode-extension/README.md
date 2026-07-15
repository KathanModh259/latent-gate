# LatentGate — Local-First AI Compression for VS Code

Compress images, text, conversations, and documents locally via Ollama before sending to cloud LLMs. Save ~80% on token costs.

## Features

- **Compress selected text** — right-click or `Ctrl+Shift+Alt+C`
- **Compress images** — right-click image files in explorer/editor
- **Compress documents** — full file compression from command palette
- **Cost dashboard** — track token savings per session
- **Ollama health check** — verify your local setup
- **Selective decoding** — skip API calls when content is unchanged
- **Embedding-based Similarity** — more accurate scene detection

## Requirements

- [Ollama](https://ollama.com) running locally
- At least one model pulled (e.g., `ollama pull phi3:mini`)

## Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `latentGate.ollamaBaseUrl` | `http://localhost:11434` | Ollama server URL |
| `latentGate.visionModel` | `llava:7b` | Vision model |
| `latentGate.textFastModel` | `phi3:mini` | Fast text model |
| `latentGate.textSmartModel` | `qwen2:7b` | Smart text model |
| `latentGate.embeddingModel` | `nomic-embed-text` | Embedding model |
| `latentGate.enableCaching` | `true` | Cache repeated images |
| `latentGate.selectiveDecoding` | `true` | Skip unchanged scenes |

## Commands

- `LatentGate: Compress Image`
- `LatentGate: Compress Selected Text`
- `LatentGate: Compress Current Document`
- `LatentGate: Show Cost Dashboard`
- `LatentGate: Show Session Stats`
- `LatentGate: Open Settings`
- `LatentGate: Check Ollama Health`
- `LatentGate: Configure MCP Server`
