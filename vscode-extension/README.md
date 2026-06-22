# LatentGate — VSCode Extension

Local-first AI compression for VSCode. Compress images, text, conversations, and documents locally via Ollama before sending to cloud LLMs. Save ~80% on token costs.

## Features

- **Compress Images** — Right-click any image → LatentGate compresses it to ~150 tokens
- **Compress Text** — Select text → Ctrl+Shift+Alt+C to compress
- **Cost Dashboard** — Activity bar panel showing token savings and stats
- **MCP Integration** — Auto-configures MCP server for Copilot Chat
- **Status Bar** — Live token savings counter

## Prerequisites

1. **Python 3.10+** with `latent-gate` installed:
   ```bash
   pip install latent-gate[all]
   ```

2. **Ollama** running with required models:
   ```bash
   ollama pull llava:7b
   ollama pull llama3:8b
   ```

## Commands

| Command | Description | Shortcut |
|---------|-------------|----------|
| `LatentGate: Compress Image` | Compress an image file | Right-click menu |
| `LatentGate: Compress Selected Text` | Compress selected text | `Ctrl+Shift+Alt+C` |
| `LatentGate: Compress Document` | Compress entire document | Command Palette |
| `LatentGate: Show Cost Dashboard` | Open cost dashboard | `Ctrl+Shift+Alt+D` |
| `LatentGate: Show Session Stats` | Show token savings | Command Palette |
| `LatentGate: Check Ollama Health` | Verify Ollama is running | Command Palette |
| `LatentGate: Configure MCP Server` | Setup MCP for Copilot | Command Palette |

## Configuration

Open VSCode Settings → search `latentGate` to configure:

- `latentGate.visionModel` — Vision model (default: `llava:7b`)
- `latentGate.predictorModel` — Text model (default: `llama3:8b`)
- `latentGate.remoteProvider` — Remote LLM provider
- `latentGate.remoteModel` — Remote model name
- `latentGate.selectiveDecoding` — Skip API calls for unchanged scenes

## MCP Server

The extension auto-configures the MCP server for VSCode Copilot Chat. This enables Copilot to use LatentGate's compression tools automatically.

### Manual Configuration

Add to your `settings.json`:

```json
{
  "copilot.chat.mcpServers": {
    "latent-gate": {
      "command": "python",
      "args": ["-m", "latent_gate.mcp_server"]
    }
  }
}
```

## How It Works

1. **Local Processing** — Images/text are processed locally via Ollama (free)
2. **Compression** — Extracts essential semantics (~150 tokens vs ~1200 raw)
3. **Remote Decoding** — Sends compact payload to cloud LLM
4. **Savings** — ~80% reduction in token costs

## Requirements

- Python 3.10+
- Ollama (for local processing)
- VSCode 1.85+

## Links

- [GitHub](https://github.com/KathanModh259/latent-gate)
- [PyPI](https://pypi.org/project/latent-gate/)
- [Documentation](https://github.com/KathanModh259/latent-gate/blob/main/README.md)

## License

MIT License
