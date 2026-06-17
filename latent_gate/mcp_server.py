"""
LatentGate MCP Server
======================
Universal Model Context Protocol server for LatentGate.

Run via:
    python -m latent_gate.mcp_server

Configure in Claude Desktop / Cursor / Cline / Continue / Zed:
{
  "mcpServers": {
    "latent-gate": {
      "command": "python",
      "args": ["-m", "latent_gate.mcp_server"]
    }
  }
}
"""

import asyncio
import json
import logging

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except ImportError as e:
    raise ImportError(
        "MCP package not installed. Install with: pip install mcp"
    ) from e

from latent_gate import LatentGatePipeline, PipelineConfig


logger = logging.getLogger("latent_gate_mcp")
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Lazy Pipeline Initialization
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
                "Returns structured scene data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file (jpg, png, webp)",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional focus question",
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
                "cloud LLM. Use when text > 500 tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "compress", "summarize", "condense", "code"],
                        "default": "auto",
                    },
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="compress_conversation",
            description=(
                "Compress a multi-turn conversation history into a "
                "compact summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "new_question": {"type": "string"},
                },
                "required": ["messages", "new_question"],
            },
        ),
        types.Tool(
            name="compress_documents",
            description="Compress retrieved RAG documents into key facts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "question": {"type": "string"},
                },
                "required": ["documents", "question"],
            },
        ),
        types.Tool(
            name="get_stats",
            description="Get cumulative token savings statistics.",
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
                "savings_percent": round(
                    (1 - result["tokens_estimated"] / 1200) * 100, 1
                ),
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
            text=(
                f"Error: Ollama not running. Start with 'ollama serve'. "
                f"Details: {e}"
            )
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