# 🔌 LatentGate MCP Server

Universal Model Context Protocol server — works with **every modern AI tool**:

- ✅ Claude Desktop
- ✅ Cursor
- ✅ Cline (VS Code extension)
- ✅ Continue.dev
- ✅ Zed Editor
- ✅ Any custom MCP client

## What It Does

Exposes LatentGate's compression as tools that AI agents can call automatically:

| Tool | When AI Uses It |
|---|---|
| `compress_image` | Before analyzing any image (saves ~85%) |
| `compress_text` | When prompt is long (>500 tokens) |
| `compress_conversation` | When chat history gets large |
| `compress_documents` | For RAG queries with multiple docs |
| `get_stats` | To check cumulative savings |

## Install

```bash
pip install latent-gate mcp
ollama pull llava:7b
ollama pull llama3:8b
```

## Configure

### Claude Desktop

Edit `~/.claude/claude_desktop_config.json` (or on Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

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

Restart Claude Desktop. You should see a 🔌 icon — LatentGate tools are now available.

### Cursor

Add to Cursor settings → MCP:
```json
{
  "latent-gate": {
    "command": "python",
    "args": ["-m", "latent_gate.mcp_server"]
  }
}
```

### Cline (VS Code)

In Cline settings → MCP Servers → Add:
- **Name**: latent-gate
- **Command**: `python -m latent_gate.mcp_server`

## How It Works

```
User: "Analyze this screenshot of my code"
  ↓
Claude/Cursor: [detects image, calls compress_image MCP tool]
  ↓
LatentGate: processes locally via Ollama → returns compact payload
  ↓
Claude/Cursor: reasons about compact payload, generates answer
  ↓
Result: Same quality answer, ~85% fewer tokens consumed
```

## Why This Matters

- **Claude Code** users hit token limits constantly — saving 80% means 5x more work per session
- **Cursor** users in long debugging sessions burn through context fast
- **RAG apps** spend most tokens on document context — compression keeps it lean

## Verify It Works

After configuring, ask your AI tool:
> "What MCP tools do you have available?"

You should see the 5 LatentGate tools listed.

## Troubleshooting

| Problem | Fix |
|---|---|
| Tools not appearing | Restart your AI tool completely |
| "Ollama not running" | Run `ollama serve` in a terminal |
| Slow first call | Models are warming up (~10s), subsequent calls are fast |
| Import error | `pip install latent-gate mcp` |
