"""
LatentGate MCP Server
======================
Universal Model Context Protocol server that exposes LatentGate's
compression capabilities to any MCP-compatible AI tool:

  - Claude Desktop
  - Cursor
  - Cline (VS Code)
  - Continue.dev
  - Zed
  - Any custom MCP client

Install:
    pip install mcp latent-gate

Run:
    python -m latent_gate_mcp.server

Configure in Claude Desktop (~/.claude/claude_desktop_config.json):
{
  "mcpServers": {
    "latent-gate": {
      "command": "python",
      "args": ["-m", "latent_gate_mcp.server"]
    }
  }
}
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types

from latent_gate import LatentGatePipeline, PipelineConfig


logger = logging.getLogger("latent_gate_mcp")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Initialize Pipeline (lazy — only when first tool is called)
# ============================================================================

_pipeline: LatentGatePipeline = None


def get_pipeline() -> LatentGatePipeline:
    """Get or create the LatentGate pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        config = PipelineConfig(
            vision_model="llava:7b",
            predictor_model="llama3:8b",
            remote_provider="ollama",
            remote_model="llama3:8b",
            enable_caching=True,
            log_level="WARNING",
        )
        _pipeline = LatentGatePipeline(config, preload=True)
        logger.info("LatentGate pipeline initialized")
    return _pipeline


# ============================================================================
# MCP Server
# ============================================================================

app = Server("latent-gate")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List all tools exposed by this MCP server."""
    return [
        types.Tool(
            name="compress_image",
            description=(
                "Compress an image to a compact ~150 token semantic payload "
                "instead of the typical 1000-1300 tokens consumed when "
                "sending raw images to LLMs. Use this BEFORE sending any "
                "image to Claude/GPT-4o to save 80%+ on token costs. "
                "Returns structured scene data (objects, actions, layout, "
                "text) that the agent can reason about."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute or relative path to image file (jpg, png, webp)",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional question to focus the extraction on",
                        "default": "",
                    },
                },
                "required": ["image_path"],
            },
        ),
        types.Tool(
            name="compress_text",
            description=(
                "Compress a long text prompt locally before sending to "
                "cloud LLM. Extracts intent, entities, constraints, and "
                "key data points into a compact ~150 token payload. "
                "Use when: text > 500 tokens, hitting context limits, "
                "or want to save API costs. Modes: auto, compress, "
                "summarize, condense, code."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The long text/prompt to compress",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "compress", "summarize", "condense", "code"],
                        "description": "Compression strategy",
                        "default": "auto",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="compress_conversation",
            description=(
                "Compress a multi-turn conversation history into a compact "
                "summary. Use when chat history is growing too large and "
                "consuming context window. Returns key facts, decisions "
                "made, and entities discussed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "List of {role, content} dicts",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "new_question": {
                        "type": "string",
                        "description": "The new question to answer",
                    },
                },
                "required": ["messages", "new_question"],
            },
        ),
        types.Tool(
            name="compress_documents",
            description=(
                "Compress retrieved RAG documents into key facts relevant "
                "to a specific question. Use when you have multiple long "
                "documents and need to extract only what's relevant. "
                "Reduces 3000+ tokens of context to ~450 tokens of facts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of document strings to condense",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question these docs should answer",
                    },
                },
                "required": ["documents", "question"],
            },
        ),
        types.Tool(
            name="get_stats",
            description=(
                "Get cumulative token savings statistics for this session "
                "(total tokens saved, compression ratios, cache hits)."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a tool call."""
    pipeline = get_pipeline()

    try:
        if name == "compress_image":
            result = pipeline.query(
                arguments["image_path"],
                arguments.get("question", "Describe this image"),
            )
            response = {
                "compact_payload": result["compact_prompt"],
                "tokens_estimated": result["tokens_estimated"],
                "tokens_saved": 1200 - result["tokens_estimated"],
                "savings_percent": round((1 - result["tokens_estimated"] / 1200) * 100, 1),
                "extracted_data": result["payload"],
                "answer": result["answer"],
            }

        elif name == "compress_text":
            result = pipeline.query_text(
                arguments["text"],
                mode=arguments.get("mode", "auto"),
            )
            response = {
                "compact_payload": result["compact_prompt"],
                "original_tokens": result.get("original_tokens", 0),
                "compressed_tokens": result["tokens_estimated"],
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "tokens_saved": result.get("tokens_saved", 0),
                "answer": result["answer"],
            }

        elif name == "compress_conversation":
            result = pipeline.query_conversation(
                arguments["messages"],
                arguments["new_question"],
            )
            response = {
                "compact_payload": result["compact_prompt"],
                "original_tokens": result.get("original_tokens", 0),
                "compressed_tokens": result["tokens_estimated"],
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "answer": result["answer"],
            }

        elif name == "compress_documents":
            result = pipeline.query_documents(
                arguments["documents"],
                arguments["question"],
            )
            response = {
                "compact_payload": result["compact_prompt"],
                "original_tokens": result.get("original_tokens", 0),
                "compressed_tokens": result["tokens_estimated"],
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "answer": result["answer"],
            }

        elif name == "get_stats":
            stats = pipeline.selective_decoder.stats
            response = {
                "session_stats": stats,
                "note": "Stats reset when MCP server restarts",
            }

        else:
            response = {"error": f"Unknown tool: {name}"}

        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

    except FileNotFoundError as e:
        return [types.TextContent(type="text", text=f"Error: File not found - {e}")]
    except ConnectionError as e:
        return [types.TextContent(
            type="text",
            text=f"Error: Ollama not running. Start with `ollama serve`. Details: {e}"
        )]
    except Exception as e:
        logger.exception("Tool call failed")
        return [types.TextContent(type="text", text=f"Error: {e}")]


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="latent-gate",
                server_version="0.3.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
