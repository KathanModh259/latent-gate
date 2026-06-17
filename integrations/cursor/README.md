# 🔵 LatentGate for Cursor

Make Cursor automatically compress images, long context, and RAG documents
locally via Ollama — saving ~80% of your token budget.

## Two Integration Options

### Option 1: MCP Server (Recommended)

The MCP server gives Cursor's AI direct access to LatentGate as tools.

**Configure Cursor:**

1. Open Cursor Settings → Features → MCP
2. Click "Add new MCP server"
3. Paste contents of `mcp_config.json`
4. Save and restart Cursor

### Option 2: .cursorrules

Add the `.cursorrules` file to your project root. Cursor will follow
these rules automatically when working in that project.

```bash
cp .cursorrules /path/to/your/project/.cursorrules
```

## Prerequisites

```bash
pip install latent-gate mcp
ollama pull llava:7b
ollama pull llama3:8b
```

## Verifying It Works

In Cursor's chat, ask:
> "What MCP tools do you have available?"

You should see LatentGate tools listed.

Or just try:
> "Analyze this screenshot" (with an image attached)

Cursor should respond with something like:
> "I'll compress this image locally first using LatentGate to save tokens..."

## Use Cases Where This Shines

| Scenario | Without LatentGate | With LatentGate |
|---|---|---|
| Debugging a UI screenshot | 1,200 tokens | 150 tokens |
| Analyzing a 500-line file | 800 tokens | 120 tokens |
| Working with RAG context | 3,000 tokens | 450 tokens |
| Long debugging session | Hits limit fast | 5x more turns possible |
