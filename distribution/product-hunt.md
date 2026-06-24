# Product Hunt Launch Guide

## Pre-Launch Checklist

- [ ] Create Product Hunt account
- [ ] Submit product at least 1 week before launch
- [ ] Prepare all assets (see below)
- [ ] Write compelling tagline
- [ ] Prepare maker comment
- [ ] Line up 5-10 supporters for launch day upvotes

## Product Details

**Name**: LatentGate

**Tagline Options**:
1. "Process Locally. Send Smart. Pay Less."
2. "Cut LLM costs by 80% with local vision-language processing"
3. "Your images, compressed locally — 80% fewer tokens to the cloud"
4. "Stop paying to process images in the cloud"

**Description** (260 chars):
> LatentGate is a VL-JEPA-inspired pipeline that compresses images, text, and RAG documents locally via Ollama before sending to any LLM API. Cut token costs by ~80% while keeping your data private. Works with Claude, GPT-4, Gemini, and 9+ providers.

**Description** (Full):
> Every time you send an image or long prompt to GPT-4o / Claude / Gemini, you burn 1,000+ tokens on processing that could happen locally for free.
>
> LatentGate flips this: we process images and text locally using Ollama (free, no API key needed), then send only a compact semantic payload (~200 tokens) to the cloud LLM for the final reasoning step.
>
> **Key Features:**
> - Local-first processing via Ollama
> - ~80% token savings
> - Works with OpenAI, Anthropic, Google, Groq, DeepSeek, Together, Azure, AWS Bedrock
> - MCP server for AI coding tools (Claude Desktop, Cursor, Cline)
> - LangChain and LlamaIndex integrations
> - Video processing with selective decoding
> - REST API and async support
> - Docker and Kubernetes deployment

## Assets to Prepare

### Screenshots (1270x760px)
1. **Hero image**: Pipeline diagram showing Local → Cloud flow
2. **Cost comparison**: Before/after token usage
3. **Code example**: Quick start Python snippet
4. **MCP integration**: Claude Desktop screenshot with LatentGate

### Video (1-2 min)
1. Show the problem (sending large images to GPT-4o)
2. Demonstrate LatentGate in action
3. Show the cost savings
4. Show the MCP integration

### GIFs
1. CLI usage: `python -m latent_gate photo.jpg "What is this?"`
2. Cost tracker output showing savings
3. MCP tool being used in Claude Desktop

## Maker Comment Template

```
Hey Product Hunt! 👋

I built LatentGate because I was tired of paying for image processing in the cloud.

Every time you send an image to GPT-4o, you're burning ~1,000+ tokens. But most of that work (identifying objects, understanding scenes) can happen locally for free.

LatentGate uses Ollama to process images and text locally, then sends only a compact semantic payload to the cloud LLM. Result: ~80% token savings.

It works with:
- OpenAI, Anthropic, Google, Groq, DeepSeek, Together, Azure, AWS Bedrock
- MCP (Claude Desktop, Cursor, Cline)
- LangChain and LlamaIndex
- Video processing with selective decoding

Open source, MIT licensed. Give it a try!

https://github.com/KathanModh259/latent-gate
```

## Launch Day Strategy

1. **Post at 12:01 AM PT** (Product Hunt resets at midnight)
2. **Share on Twitter/X** with #ProductHunt and #AI tags
3. **Post in relevant communities**: Reddit, HN, Discord
4. **Reply to every comment** within 1 hour
5. **Update your own status** with milestones

## Post-Launch

- [ ] Thank supporters
- [ ] Share results on social media
- [ ] Update README with Product Hunt badge
- [ ] Write a "lessons learned" blog post
