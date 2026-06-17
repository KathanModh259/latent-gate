---
name: latent-gate-compression
description: |
  Compress images, long prompts, conversations, and RAG documents locally
  via Ollama before consuming Claude tokens. Reduces token usage by ~80%
  on images (OpenAI/Claude) and ~85% on text. Use proactively when:
  the user uploads images, processes long context, works with multiple
  RAG sources, or when approaching token limits.
version: 0.3.0
author: Kathan Modh
license: MIT
---

# LatentGate Token Compression Skill

You have access to a local-first compression pipeline that processes
images and text via Ollama (free, runs on user's machine) and returns
compact semantic payloads instead of consuming Claude's tokens on
raw content.

## When To Use This Skill

**ALWAYS use when:**
- User uploads or references an image file (>500 KB or any vision task)
- A document or prompt exceeds 500 tokens
- Working with retrieved RAG context (>3 chunks)
- Conversation history is growing large (>5 turns)
- User mentions hitting token limits or wanting to save costs

**Don't use when:**
- Quick factual questions (<100 tokens)
- User explicitly asks to read raw content
- Ollama is not installed/running on user's machine

## Available Scripts

### 1. Compress Image
```bash
python scripts/compress_image.py <image_path> [question]
```
Returns: JSON with `compact_payload`, `tokens_saved`, `extracted_data`

**Example use:**
```
User: "What's wrong with my UI in this screenshot?"
You: [Run] python scripts/compress_image.py screenshot.png "find UI issues"
You: [Reason about the returned compact payload]
You: [Provide answer using ~150 tokens of context instead of ~1200]
```

### 2. Compress Text
```bash
python scripts/compress_text.py <text_file> [mode]
```
Modes: `auto`, `compress`, `summarize`, `condense`, `code`

**Example use:**
```
User: "Help me refactor this 800-line file"
You: [Run] python scripts/compress_text.py code.py code
You: [Reason about extracted intent + code snippets]
You: [Apply refactoring with full context preserved in compact form]
```

### 3. Compress Documents (RAG)
```bash
python scripts/compress_docs.py <question> <doc1> <doc2> ...
```

**Example use:**
```
User: "Based on these 5 docs, what should I do?"
You: [Run] python scripts/compress_docs.py "what to do" doc1.txt doc2.txt ...
You: [Get only relevant facts, save ~2500 tokens]
```

## Decision Flowchart

```
Is there an image?
├── YES → compress_image.py BEFORE analyzing
└── NO ↓

Is the text > 500 tokens?
├── YES → compress_text.py BEFORE responding
└── NO ↓

Are there multiple documents?
├── YES → compress_docs.py BEFORE synthesizing
└── NO → Proceed normally
```

## How It Saves Tokens

| Scenario | Without Skill | With Skill | Savings |
|---|---:|---:|---|
| Single image | ~1,200 tokens | ~150 tokens | 87% |
| Long prompt | ~800 tokens | ~120 tokens | 85% |
| 10-turn chat | ~2,500 tokens | ~350 tokens | 86% |
| 3 RAG docs | ~3,000 tokens | ~450 tokens | 85% |

## Important Notes

1. **Always inform the user** when you use this skill — they should know
   you compressed their input. Say: "I compressed your [image/prompt]
   locally first to save tokens."

2. **Quality stays the same** — the local Ollama model extracts the
   semantic content; you still do the actual reasoning.

3. **Cache is automatic** — same image processed twice returns instantly
   from cache.

4. **Errors are graceful** — if Ollama isn't running, the script will
   tell you, and you can fall back to normal processing.

## Setup Instructions (Tell User Once)

If the user hasn't set up LatentGate yet:

```bash
# Install
pip install latent-gate

# Pull required Ollama models (one-time, ~10GB)
ollama pull llava:7b
ollama pull llama3:8b

# Verify
python -c "from latent_gate import LatentGatePipeline; print('Ready!')"
```

## Real-World Example Sessions

### Session 1: Image Analysis
```
User: "Why is this button not centered? [uploads screenshot.png]"

You: I'll compress the image locally first to save your token budget.

[Bash: python scripts/compress_image.py screenshot.png "button centering issue"]

[Returns]: {
  "compact_payload": "[Scene: UI screenshot] Objects: button (off-center, 
   left-aligned), container (flex), header | Layout: button is at 
   left edge of container | Text: 'Submit' button visible",
  "tokens_saved": 1050,
  "savings_percent": 87.5
}

You: The button isn't centered because the container is using `flex` 
without `justify-content: center`. Add this CSS:

.container {
  display: flex;
  justify-content: center;
}
```

### Session 2: Long Prompt
```
User: [Pastes 600-line specification document]
      "Implement this"

You: I'll compress this spec locally first to save your token budget.

[Bash: python scripts/compress_text.py spec.txt compress]

[Returns]: {
  "compact_payload": "Intent: Build REST API for inventory management | 
   Entities: PostgreSQL, JWT auth, Redis cache | Constraints: <200ms p99 
   latency, OpenAPI docs required | Data: 50 endpoints, 12 models",
  "original_tokens": 850,
  "compressed_tokens": 110,
  "compression_ratio": "7.7x"
}

You: Based on the spec, I'll build the inventory API with these components:
1. FastAPI for OpenAPI generation
2. SQLAlchemy + PostgreSQL for the 12 data models
[...continues with implementation]
```
