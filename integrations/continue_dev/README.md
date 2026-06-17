# 🟢 LatentGate for Continue.dev

Continue.dev is an open-source AI code assistant. Add LatentGate as
an MCP server to save tokens automatically.

## Setup

### 1. Install
```bash
pip install latent-gate mcp
ollama pull llava:7b
ollama pull llama3:8b
```

### 2. Configure Continue

Merge `config.json` into your existing Continue config:

**Location:**
- Windows: `%USERPROFILE%\.continue\config.json`
- macOS/Linux: `~/.continue/config.json`

### 3. Restart Continue

You'll now have two new custom commands:
- `/compress-image` — compress image before analysis
- `/compress-context` — compress current file/selection

## Usage

Type in Continue chat:
```
/compress-context Refactor this for better performance
```

Continue will:
1. Compress the current file/selection via LatentGate (locally, free)
2. Reason about the compact ~150 token payload
3. Apply your changes with full context preserved
