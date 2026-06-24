# 🔌 LatentGate Integrations

Make your favorite AI tools automatically save tokens with LatentGate.

## Available Integrations

| Integration | Best For | Setup Time |
|---|---|---|
| [🔌 MCP Server](mcp_server/) | Claude Desktop, Cursor, Cline, Continue, Zed | 2 min |
| [🟣 Claude Code Skill](claude_code_skill/) | Claude Code CLI | 2 min |
| [🔵 Cursor](cursor/) | Cursor IDE (.cursorrules + MCP) | 1 min |
| [🟢 Continue.dev](continue_dev/) | Continue VS Code extension | 2 min |
| [🟠 OpenAI Functions](openai_functions/) | Custom OpenAI/Claude/Gemini agents | 5 min |
| [🦜 LangChain](langchain/) | LangChain chains, agents, and tools | 3 min |
| [🦙 LlamaIndex](llamaindex/) | LlamaIndex query engines and retrievers | 3 min |

## Which One Should I Use?

```
Do you use Claude Code CLI?
  → Use the Claude Code Skill

Do you use Claude Desktop, Cursor, or Cline?
  → Use the MCP Server (works with all of them)

Do you use Continue.dev?
  → Use the Continue config

Are you building your own AI agent?
  → Use the OpenAI Functions schema
```

## Prerequisites (All Integrations)

```bash
# 1. Install LatentGate
pip install latent-gate

# 2. Install Ollama
# See https://ollama.com for your platform

# 3. Pull required models
ollama pull llava:7b      # Vision model (~4.7 GB)
ollama pull llama3:8b     # Text model (~4.7 GB)

# 4. Verify
ollama list
python -c "from latent_gate import LatentGatePipeline; print('Ready!')"
```

## What You Save

Across all integrations, average savings per AI session:

| Workflow | Tokens Saved | $$ Saved (Claude) |
|---|---:|---:|
| 1 hour coding with images | ~6,000 | ~$0.018 |
| 1 day Claude Code usage | ~50,000 | ~$0.15 |
| 1 month heavy usage | ~1,500,000 | ~$4.50 |
| Per organization (10 devs) | ~15,000,000 | ~$45/mo |

Multiply that by every developer at your org. It adds up fast.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  YOUR AI TOOL (Claude Code, Cursor, Cline, etc.)     │
└─────────────────┬────────────────────────────────────┘
                  │ MCP / Skill / Function Call
                  ▼
┌──────────────────────────────────────────────────────┐
│  LatentGate (runs locally on your machine)           │
│   - X-Encoder (LLaVA via Ollama) — vision            │
│   - Predictor — structured extraction                │
│   - Y-Decoder — cloud LLM (or local)                 │
└──────────────────────────────────────────────────────┘
```

## Support

- Issues: https://github.com/KathanModh259/latent-gate/issues
- Discussions: https://github.com/KathanModh259/latent-gate/discussions
