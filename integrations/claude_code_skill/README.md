# 🟣 LatentGate Claude Code Skill

Make Claude Code automatically compress images, long prompts, and RAG documents
locally — saving ~80% of your token budget per session.

## Installation

### 1. Install LatentGate
```bash
pip install latent-gate
ollama pull llava:7b
ollama pull llama3:8b
```

### 2. Install Skill into Claude

Copy this directory into your Claude skills folder:

**macOS / Linux:**
```bash
cp -r . ~/.claude/skills/latent-gate/
```

**Windows:**
```powershell
Copy-Item -Recurse . $env:USERPROFILE\.claude\skills\latent-gate\
```

### 3. Restart Claude Code

That's it — Claude will now automatically use LatentGate when appropriate.

## What Changes For You

**Before:**
```
You: "Analyze this screenshot of my code"
Claude: [Burns ~1,200 tokens reading raw image]
Claude: [Has less budget for the actual response]
```

**After:**
```
You: "Analyze this screenshot of my code"
Claude: "I'll compress this locally first to save tokens..."
Claude: [Uses ~150 tokens for compact payload]
Claude: [Has 1,000+ extra tokens for detailed analysis]
```

## Files

| File | Purpose |
|---|---|
| `SKILL.md` | Instructions Claude follows |
| `scripts/compress_image.py` | Image compression |
| `scripts/compress_text.py` | Text/prompt compression |
| `scripts/compress_docs.py` | RAG document compression |

## Token Savings Per Session

In a typical Claude Code session with 5 images and 3 long files:

| Operation | Without Skill | With Skill |
|---|---:|---:|
| 5 images @ 1200 tokens | 6,000 | 750 |
| 3 long files @ 800 tokens | 2,400 | 360 |
| **Total input tokens** | **8,400** | **1,110** |
| **Tokens saved** | — | **7,290 (87%)** |

That's like getting **5x more work done** in the same session.

## Verifying It's Active

In Claude Code, type:
> "What skills do you have available?"

You should see `latent-gate-compression` in the list.

## Troubleshooting

| Problem | Fix |
|---|---|
| Skill not detected | Restart Claude Code after copying files |
| "Ollama not running" | Run `ollama serve` in a terminal |
| Slow first call | Models warming up; subsequent calls are <1s |
| Want to disable | `rm -rf ~/.claude/skills/latent-gate/` |
