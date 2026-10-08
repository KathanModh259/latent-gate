"""
LatentGate MCP Server
======================
Universal Model Context Protocol server for LatentGate.

Run via:
    latent-gate-mcp

Claude Code:
    claude mcp add latent-gate -- uvx --from "latent-gate[mcp]" latent-gate-mcp

Claude Desktop / Cursor / Cline / Continue / Zed:
{
  "mcpServers": {
    "latent-gate": {
      "command": "uvx",
      "args": ["--from", "latent-gate[mcp]", "latent-gate-mcp"]
    }
  }
}

The optimizer tools (read_file_optimized, optimize_text, count_tokens) need
no Ollama and no API key. The compress_* tools use local Ollama models.

Hosted (remote) mode, for a gateway such as mcprush:
    latent-gate mcp --http [--port 8000]
Serves streamable HTTP at /mcp. Tools that take a file path are hidden, since
they would read the server's disk. Set LATENTGATE_API_KEY to require
`Authorization: Bearer <key>` so callers can't bypass the paying gateway.
"""

import argparse
import asyncio
import hmac
import json
import logging
import os

try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except ImportError as e:
    raise ImportError("MCP package not installed. Install with: pip install mcp") from e

from pathlib import Path
from typing import Optional

from latent_gate import LatentGatePipeline
from latent_gate.optimizer import TokenOptimizer, count_tokens, token_counter_name
from latent_gate.remote_decoder import RemoteDecodeError

logger = logging.getLogger("latent_gate_mcp")
# stdout carries the MCP protocol; logs go to stderr (basicConfig's default stream)
logging.basicConfig(level=logging.INFO)

MAX_FILE_BYTES = 20 * 1024 * 1024  # read_file_optimized refuses larger files
_LEVELS = ["lossless", "balanced", "aggressive"]

# Set by --http. Path-taking tools read whatever disk the server runs on, which
# remotely means *our* disk, so they are never exposed there. get_stats reports
# process-wide counters, which on a shared server would mix every customer's usage.
_REMOTE = False
_LOCAL_ONLY_TOOLS = {"read_file_optimized", "compress_image", "get_stats"}
# Ollama-backed tools are only listed remotely when the host actually runs Ollama.
_OLLAMA_TOOLS = {"compress_text", "compress_conversation", "compress_documents"}
MAX_REMOTE_INPUT_BYTES = 2 * 1024 * 1024  # per hosted call; plans bill per call, not per byte


def _hidden_remotely() -> set[str]:
    hidden = set(_LOCAL_ONLY_TOOLS)
    if os.environ.get("LATENTGATE_HOSTED_OLLAMA", "").lower() not in ("1", "true", "yes"):
        hidden |= _OLLAMA_TOOLS
    return hidden


# ============================================================================
# Lazy Pipeline Initialization
# ============================================================================

_pipeline: Optional[LatentGatePipeline] = None


def get_pipeline() -> LatentGatePipeline:
    """Get or create the LatentGate pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        from latent_gate.config_loader import get_config

        config = get_config()
        config.log_level = "WARNING"
        _pipeline = LatentGatePipeline(config, preload=True)
        logger.info("LatentGate pipeline initialized")
    return _pipeline


# ============================================================================
# MCP Server
# ============================================================================

app = Server("latent-gate")


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """List the tools exposed in the current mode."""
    tools = _all_tools()
    if _REMOTE:
        hidden = _hidden_remotely()
        tools = [t for t in tools if t.name not in hidden]
    return tools


def _all_tools() -> list[types.Tool]:
    level_schema = {
        "type": "string",
        "enum": _LEVELS,
        "default": "balanced",
        "description": "lossless = meaning unchanged; balanced = also folds log runs and "
        "removes filler; aggressive = also keeps only the most relevant sentences (~50%)",
    }
    return [
        types.Tool(
            name="read_file_optimized",
            description=(
                "Read a local text file (logs, JSON/API dumps, docs, config, source) and "
                "return a token-optimized version instead of the raw content. Use this "
                "INSTEAD of reading large files directly: pretty-printed JSON shrinks "
                "~30% losslessly and repetitive logs ~90% (runs are folded with value "
                "ranges kept). Code, URLs and quoted strings are returned byte-for-byte. "
                "Pass `question` to focus selection on what you need. Works offline, "
                "no Ollama or API key required."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to a text file"},
                    "question": {
                        "type": "string",
                        "default": "",
                        "description": "What you're looking for (focuses sentence selection)",
                    },
                    "level": level_schema,
                    "max_tokens": {
                        "type": "integer",
                        "default": 0,
                        "description": "Token budget for the result (0 = no budget)",
                    },
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="fetch_url_optimized",
            description=(
                "Fetch a public web page, JSON API response, raw log or doc by URL and "
                "return a token-optimized version: HTML is reduced to its visible text, "
                "JSON is compacted losslessly, repetitive lines are folded with value "
                "ranges kept. Deterministic and fact-preserving (no summarizing model), "
                "so numbers, code and URLs survive byte-for-byte. Pass `question` to "
                "focus on what you need. Private/internal addresses are refused."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL"},
                    "question": {
                        "type": "string",
                        "default": "",
                        "description": "What you're looking for (focuses sentence selection)",
                    },
                    "level": level_schema,
                    "max_tokens": {
                        "type": "integer",
                        "default": 0,
                        "description": "Token budget for the result (0 = no budget)",
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="optimize_text",
            description=(
                "Deterministically shrink text before passing it on (e.g. to another tool "
                "or model): lossless JSON/whitespace/duplicate folding, log-run folding, "
                "filler removal, optional question-aware selection under a token budget. "
                "Same input always gives the same output. No Ollama or API key required."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "question": {"type": "string", "default": ""},
                    "level": level_schema,
                    "max_tokens": {"type": "integer", "default": 0},
                },
                "required": ["text"],
            },
        ),
        types.Tool(
            name="count_tokens",
            description="Count tokens in text (tiktoken o200k_base when installed, else an estimate).",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        types.Tool(
            name="compress_image",
            description=(
                "Describe an image with a local vision model (Ollama llava) as a compact "
                "~150-token structured scene payload, instead of sending the raw image "
                "(typically 1000+ tokens). Requires Ollama with a vision model pulled."
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
                "Compress a long prompt with the optimizer plus a fact-checked local LLM "
                "rewrite (Ollama). Prefer optimize_text when Ollama isn't running."
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
            description=("Compress a multi-turn conversation history into a " "compact summary."),
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
    """Execute a tool call. Blocking work always runs off the event loop."""
    try:
        if _REMOTE:
            if name in _hidden_remotely():
                raise ValueError(f"{name} is only available when LatentGate runs locally")
            _check_remote_limits(name, arguments)
        if name in _OPTIMIZER_TOOLS:
            response = await asyncio.to_thread(_OPTIMIZER_TOOLS[name], arguments)
        else:
            # Ollama-backed tools: pipeline creation imports ML libs and warms models
            pipeline = await asyncio.to_thread(get_pipeline)
            response = await asyncio.to_thread(_run_pipeline_tool, pipeline, name, arguments)
        return [types.TextContent(type="text", text=json.dumps(response, indent=2))]

    except FileNotFoundError as e:
        return [types.TextContent(type="text", text=f"Error: File not found - {e}")]
    except ConnectionError as e:
        return [
            types.TextContent(
                type="text",
                text=(
                    "Error: Ollama not running. Start with 'ollama serve', or use the "
                    f"offline tools (read_file_optimized, optimize_text). Details: {e}"
                ),
            )
        ]
    except RemoteDecodeError as e:
        return [types.TextContent(type="text", text=f"Error: LLM provider failed - {e}")]
    except (ValueError, PermissionError, IsADirectoryError) as e:
        return [types.TextContent(type="text", text=f"Error: {e}")]
    except Exception as e:
        logger.exception("Tool call failed")
        return [types.TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


# ---------------------------------------------------------------------------
# Offline optimizer tools (no Ollama, no API key)
# ---------------------------------------------------------------------------


def _optimize_payload(result, extra: dict) -> dict:
    return {
        **extra,
        "optimized": result.text,
        "original_tokens": result.original_tokens,
        "optimized_tokens": result.optimized_tokens,
        "savings_pct": round(result.savings_pct, 1),
        "level": result.level,
        "token_counter": result.counter,
    }


def _optimizer_for(arguments: dict) -> TokenOptimizer:
    level = arguments.get("level") or "balanced"
    if level not in _LEVELS:
        raise ValueError(f"level must be one of {_LEVELS}")
    return TokenOptimizer(level)


def _tool_read_file_optimized(arguments: dict) -> dict:
    path = Path(arguments["path"]).expanduser()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"File is {size / 1048576:.1f}MB; limit is {MAX_FILE_BYTES // 1048576}MB")
    raw = path.read_bytes()
    if b"\0" in raw[:8192]:
        raise ValueError("File looks binary; read_file_optimized only handles text files")
    result = _optimizer_for(arguments).optimize(
        raw.decode("utf-8", errors="replace"),
        question=arguments.get("question", ""),
        max_tokens=int(arguments.get("max_tokens") or 0),
    )
    return _optimize_payload(result, {"path": str(path)})


def _tool_optimize_text(arguments: dict) -> dict:
    result = _optimizer_for(arguments).optimize(
        arguments["text"],
        question=arguments.get("question", ""),
        max_tokens=int(arguments.get("max_tokens") or 0),
    )
    return _optimize_payload(result, {})


def _tool_fetch_url_optimized(arguments: dict) -> dict:
    from latent_gate.web_fetch import fetch_text

    page = fetch_text(arguments["url"])
    result = _optimizer_for(arguments).optimize(
        page["text"],
        question=arguments.get("question", ""),
        max_tokens=int(arguments.get("max_tokens") or 0),
    )
    extra = {"url": page["url"], "content_type": page["content_type"]}
    if page["title"]:
        extra["title"] = page["title"]
    return _optimize_payload(result, extra)


def _tool_count_tokens(arguments: dict) -> dict:
    return {"tokens": count_tokens(arguments["text"]), "token_counter": token_counter_name()}


_OPTIMIZER_TOOLS = {
    "read_file_optimized": _tool_read_file_optimized,
    "optimize_text": _tool_optimize_text,
    "fetch_url_optimized": _tool_fetch_url_optimized,
    "count_tokens": _tool_count_tokens,
}


# ---------------------------------------------------------------------------
# Ollama-backed tools (blocking; called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _counts(result: dict) -> dict:
    return {
        "compact_payload": result["compact_prompt"],
        "original_tokens": result.get("original_tokens", 0),
        "compressed_tokens": result["tokens_estimated"],
        "compression_ratio": result.get("compression_ratio", "1.0x"),
        "tokens_saved": result.get("tokens_saved", 0),
        "answer": result["answer"],
    }


def _run_pipeline_tool(pipeline, name: str, arguments: dict) -> dict:
    if name == "compress_image":
        result = pipeline.query(
            arguments["image_path"],
            arguments.get("question", "Describe this image"),
            compress_only=True,
        )
        baseline = result.get("original_tokens", 1200)
        return {
            "compact_payload": result["compact_prompt"],
            "tokens_estimated": result["tokens_estimated"],
            "tokens_saved": result.get(
                "tokens_saved", max(0, baseline - result["tokens_estimated"])
            ),
            "savings_percent": round((1 - result["tokens_estimated"] / max(baseline, 1)) * 100, 1),
            "extracted_data": result["payload"],
            "answer": result["answer"],
        }
    if name == "compress_text":
        return _counts(
            pipeline.query_text(
                arguments["text"], mode=arguments.get("mode", "auto"), compress_only=True
            )
        )
    if name == "compress_conversation":
        return _counts(
            pipeline.query_conversation(
                arguments["messages"], arguments["new_question"], compress_only=True
            )
        )
    if name == "compress_documents":
        return _counts(
            pipeline.query_documents(
                arguments["documents"], arguments["question"], compress_only=True
            )
        )
    if name == "get_stats":
        return {
            "session_stats": pipeline.selective_decoder.stats,
            "note": "Stats reset when MCP server restarts",
        }
    return {"error": f"Unknown tool: {name}"}


# ============================================================================
# Main
# ============================================================================


def _check_remote_limits(name: str, arguments: dict) -> None:
    """Reject remote calls whose input is too large to serve on one plan call.

    Raise ValueError (its message goes back to the caller) to refuse a call.
    """
    size = len(json.dumps(arguments))  # covers text, documents and messages alike
    if size > MAX_REMOTE_INPUT_BYTES:
        raise ValueError(
            f"Input is {size / 1048576:.1f}MB; hosted limit is "
            f"{MAX_REMOTE_INPUT_BYTES // 1048576}MB per call. Split it, or run LatentGate locally."
        )


def _init_options() -> InitializationOptions:
    return InitializationOptions(
        server_name="latent-gate",
        server_version=__import__("latent_gate").__version__,
        capabilities=app.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )


def serve_http(host: str, port: int) -> None:
    """Run the MCP server over streamable HTTP at /mcp (stateless, JSON responses)."""
    import uvicorn  # ships with the mcp package

    uvicorn.run(build_http_app(), host=host, port=port, log_level="info")


def build_http_app():
    """The Starlette app behind --http: /mcp (Bearer-gated when a key is set) and /health."""
    import contextlib

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # Stateless: every request stands alone, so any instance can serve it and
    # idle instances can be stopped (mcprush hosting does this).
    manager = StreamableHTTPSessionManager(app=app, stateless=True, json_response=True)
    api_key = os.environ.get("LATENTGATE_API_KEY", "")
    if not api_key:
        logger.warning("LATENTGATE_API_KEY not set: /mcp is open to anyone who finds the URL")

    async def mcp_endpoint(scope, receive, send):
        if api_key:
            headers = dict(scope["headers"])
            sent = headers.get(b"authorization", b"").decode("latin-1")
            if not hmac.compare_digest(sent, f"Bearer {api_key}"):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await manager.handle_request(scope, receive, send)

    async def health(_request):
        return JSONResponse({"status": "ok"})

    class _ASGI:  # Route treats a plain callable as a request handler; a class instance as raw ASGI
        def __init__(self, fn):
            self.fn = fn

        async def __call__(self, scope, receive, send):
            await self.fn(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=_ASGI(mcp_endpoint)), Route("/health", endpoint=health)],
        lifespan=lifespan,
    )


async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, _init_options())


def cli_main(argv: Optional[list[str]] = None):
    """Console-script entry point for latent-gate-mcp (stdio, or --http for hosting)."""
    global _REMOTE
    parser = argparse.ArgumentParser(prog="latent-gate mcp")
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP at /mcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)
    if args.http:
        _REMOTE = True
        serve_http(args.host, args.port)
    else:
        asyncio.run(main())


if __name__ == "__main__":
    cli_main()
