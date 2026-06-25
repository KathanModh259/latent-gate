"""
FastAPI Server Wrapper — REST API for LatentGate pipeline.

Provides HTTP endpoints for image/text compression and querying.
Enables integration with web applications, microservices, and other tools.

Features:
  - REST API with OpenAPI documentation
  - Streaming responses for large outputs
  - Health checks and metrics
  - CORS support for web applications
  - Async request handling

Run via:
    latent-gate-api
    python -m latent_gate.api_server

Requires:
    pip install latent-gate[api]
"""

import logging
import os
import time
import tempfile
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import defaultdict

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed.\n"
        "Install with: pip install latent-gate[api]\n"
        "Or directly:  pip install fastapi uvicorn python-multipart"
    ) from e

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline
from latent_gate.remote_decoder import RemoteDecodeError

logger = logging.getLogger("latent_gate.api")

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB max upload
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
REQUEST_TIMEOUT = 300  # seconds
_RATE_LIMIT_CLEANUP_INTERVAL = 300  # seconds between IP cleanup sweeps
_last_rate_limit_cleanup: float = 0.0


# ============================================================================
# Request/Response Models
# ============================================================================


class ImageQueryRequest(BaseModel):
    """Request model for image queries."""

    image_path: str = Field(..., description="Path to the image file")
    question: str = Field(..., description="Question about the image")
    provider: Optional[str] = Field(None, description="Remote LLM provider override")
    model: Optional[str] = Field(None, description="Remote model override")


class TextQueryRequest(BaseModel):
    """Request model for text queries."""

    text: str = Field(..., description="Text to compress and query")
    question: str = Field("", description="Specific question about the text")
    mode: str = Field("auto", description="Compression mode: auto/compress/summarize/condense/code")


class ConversationQueryRequest(BaseModel):
    """Request model for conversation queries."""

    messages: List[dict] = Field(..., description="Conversation messages [{role, content}]")
    new_question: str = Field(..., description="New question to ask")


class DocumentsQueryRequest(BaseModel):
    """Request model for RAG document queries."""

    documents: List[str] = Field(..., description="List of document strings")
    question: str = Field(..., description="Question about the documents")


class UniversalQueryRequest(BaseModel):
    """Request model for universal queries."""

    text: str = Field("", description="Text input")
    image: str = Field("", description="Image path")
    question: str = Field("", description="Question")


class QueryResponse(BaseModel):
    """Response model for all query types."""

    answer: str
    compact_prompt: str
    tokens_estimated: int
    was_cached: bool
    input_type: str
    timing: dict
    original_tokens: Optional[int] = None
    compression_ratio: Optional[str] = None
    tokens_saved: Optional[int] = None


class StatsResponse(BaseModel):
    """Response model for statistics."""

    total_queries: int
    total_tokens_saved: int
    average_compression_ratio: float
    uptime_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    ollama_connected: bool
    models_loaded: bool


# ============================================================================
# Global State
# ============================================================================

pipeline: Optional[LatentGatePipeline] = None
start_time: float = 0.0
query_count: int = 0
total_tokens_saved: int = 0
_request_counts: dict = defaultdict(list)  # IP -> [timestamps] for rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 100  # requests per window


# ============================================================================
# App Factory
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage pipeline lifecycle."""
    global pipeline, start_time

    # Startup
    start_time = time.time()
    config = app.state.config
    pipeline = LatentGatePipeline(config, preload=True)
    logger.info("LatentGate API server started")

    yield

    # Shutdown
    if pipeline:
        pipeline.close()
    logger.info("LatentGate API server stopped")


def _sanitize_error(e: Exception) -> str:
    """Convert exception to user-safe error message."""
    if isinstance(e, FileNotFoundError):
        return "The requested file was not found."
    if isinstance(e, PermissionError):
        return "Permission denied. Check your API key."
    if isinstance(e, TimeoutError):
        return "Request timed out. The server may be overloaded."
    if isinstance(e, ConnectionError):
        return "Unable to connect to the backend service. Please try again."
    if isinstance(e, RemoteDecodeError):
        return f"LLM provider error: {e}"
    if isinstance(e, ValueError):
        return f"Invalid input: {e}"
    if isinstance(e, HTTPException):
        return str(e.detail)
    return "An internal error occurred. Please try again later."


def _check_rate_limit(request: Request) -> None:
    """Simple in-memory rate limiter per IP with periodic cleanup."""
    global _last_rate_limit_cleanup
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Periodic cleanup of stale IPs (every 5 minutes)
    if now - _last_rate_limit_cleanup > _RATE_LIMIT_CLEANUP_INTERVAL:
        stale_threshold = now - RATE_LIMIT_WINDOW * 2
        stale_ips = [
            ip for ip, timestamps in _request_counts.items()
            if not timestamps or timestamps[-1] < stale_threshold
        ]
        for ip in stale_ips:
            del _request_counts[ip]
        _last_rate_limit_cleanup = now

    _request_counts[client_ip] = [
        t for t in _request_counts[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_request_counts[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
        )
    _request_counts[client_ip].append(now)


def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global pipeline, start_time

    app = FastAPI(
        title="LatentGate API",
        description="Local-first vision-language pipeline API. Compress images, text, and documents locally before sending to cloud LLMs.",
        version="1.2.2",
        lifespan=lifespan,
    )

    # CORS — configurable origins in production
    cors_origins_str = os.getenv("LATENTGATE_CORS_ORIGINS", "")
    if cors_origins_str:
        cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    else:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler for unhandled errors
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )

    # Store config for later use
    app.state.config = config or PipelineConfig()

    return app


# ============================================================================
# Default app instance
# ============================================================================

app = create_app()


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ollama_connected = False
    models_loaded = False

    try:
        import asyncio
        import requests

        def _check_ollama():
            resp = requests.get(
                f"{app.state.config.ollama_base_url}/api/tags", timeout=3
            )
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return True, len(models) > 0
            return False, False

        ollama_connected, models_loaded = await asyncio.to_thread(_check_ollama)
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        version="1.2.2",
        ollama_connected=ollama_connected,
        models_loaded=models_loaded,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request):
    """Get usage statistics."""
    global query_count, total_tokens_saved
    _check_rate_limit(request)

    return StatsResponse(
        total_queries=query_count,
        total_tokens_saved=total_tokens_saved,
        average_compression_ratio=0.0,
        uptime_seconds=time.time() - start_time,
    )


@app.post("/query/image", response_model=QueryResponse)
async def query_image(request: ImageQueryRequest, http_request: Request):
    """Process an image and answer a question about it."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query(request.image_path, request.question)

        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=_sanitize_error(e))
    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Image query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


@app.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest, http_request: Request):
    """Compress text and query the remote LLM."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query_text(request.text, request.question, request.mode)

        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Text query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


@app.post("/query/conversation", response_model=QueryResponse)
async def query_conversation(request: ConversationQueryRequest, http_request: Request):
    """Compress conversation history and ask a new question."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query_conversation(request.messages, request.new_question)

        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Conversation query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


@app.post("/query/documents", response_model=QueryResponse)
async def query_documents(request: DocumentsQueryRequest, http_request: Request):
    """Compress RAG documents and answer a question."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query_documents(request.documents, request.question)

        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Documents query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


@app.post("/query/universal", response_model=QueryResponse)
async def query_universal(request: UniversalQueryRequest, http_request: Request):
    """Universal query endpoint - auto-detects input type."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query_universal(
            text=request.text,
            image=request.image,
            question=request.question,
        )

        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=_sanitize_error(e))
    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Universal query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


@app.post("/query/image/upload", response_model=QueryResponse)
async def query_image_upload(
    file: UploadFile = File(...),
    question: str = Form(...),
    http_request: Request = None,
):
    """Upload an image and query it."""
    if http_request:
        _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {ext}. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB.",
        )

    # Save uploaded file temporarily
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = pipeline.query(tmp_path, question)

        global query_count, total_tokens_saved
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]

        return QueryResponse(**result)

    except (ConnectionError, TimeoutError, PermissionError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except RemoteDecodeError as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Image upload query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Failed to clean up temp file: {tmp_path}")


# ============================================================================
# Prompt Compression Endpoint
# ============================================================================


class CompressRequest(BaseModel):
    """Request model for prompt compression."""

    text: str = Field(..., description="Verbose prompt to compress")


class CompressResponse(BaseModel):
    """Response model for prompt compression."""

    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: str
    processing_time_ms: float


@app.post("/compress", response_model=CompressResponse)
async def compress_prompt(request: CompressRequest, http_request: Request):
    """Compress a verbose prompt without calling the cloud LLM."""
    global pipeline
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.compress_prompt(request.text)
        return CompressResponse(**result)
    except (ConnectionError, TimeoutError) as e:
        raise HTTPException(status_code=502, detail=_sanitize_error(e))
    except Exception as e:
        logger.error(f"Prompt compression failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_sanitize_error(e))


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Run the API server."""
    import uvicorn

    # Override from environment
    import os

    host = os.getenv("LATENTGATE_HOST", "0.0.0.0")
    port = int(os.getenv("LATENTGATE_PORT", "8000"))

    logger.info(f"Starting LatentGate API server on {host}:{port}")

    uvicorn.run(
        "latent_gate.api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
