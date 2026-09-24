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

import json
import logging
import os
import time
import tempfile
import asyncio
from typing import Optional, List
from contextlib import asynccontextmanager
from collections import defaultdict

try:
    from fastapi import (
        FastAPI,
        HTTPException,
        APIRouter,
        UploadFile,
        File,
        Form,
        Request,
        Depends,
        Security,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from latent_gate.config_loader import get_config
from latent_gate.pipeline import LatentGatePipeline
from latent_gate.remote_decoder import RemoteDecodeError
from latent_gate.metrics import (
    record_error,
    record_tokens,
    set_pipeline_ready,
    setup_metrics,
    MetricsMiddleware,
)
from latent_gate.security import (
    PathAccessError,
    get_client_ip,
    redact_query_result,
    validate_image_path_access,
    validate_text_length,
    verify_api_key,
)

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

    image_path: str = Field(..., max_length=4096, description="Path to the image file")
    question: str = Field(..., max_length=2000, description="Question about the image")
    provider: Optional[str] = Field(None, description="Remote LLM provider override")
    model: Optional[str] = Field(None, description="Remote model override")


class TextQueryRequest(BaseModel):
    """Request model for text queries."""

    text: str = Field(..., max_length=100000, description="Text to compress and query")
    question: str = Field("", max_length=2000, description="Specific question about the text")
    mode: str = Field("auto", description="Compression mode: auto/compress/summarize/condense/code")


class ConversationQueryRequest(BaseModel):
    """Request model for conversation queries."""

    messages: List[dict] = Field(
        ..., max_length=1000, description="Conversation messages [{role, content}]"
    )
    new_question: str = Field(..., max_length=2000, description="New question to ask")


class DocumentsQueryRequest(BaseModel):
    """Request model for RAG document queries."""

    documents: List[str] = Field(..., max_length=100, description="List of document strings")
    question: str = Field(..., max_length=2000, description="Question about the documents")


class UniversalQueryRequest(BaseModel):
    """Request model for universal queries."""

    text: str = Field("", max_length=100000, description="Text input")
    image: str = Field("", max_length=4096, description="Image path")
    question: str = Field("", max_length=2000, description="Question")


class QueryResponse(BaseModel):
    """Response model for all query types."""

    model_config = {"extra": "ignore"}

    answer: str
    compact_prompt: str
    tokens_estimated: int
    was_cached: bool
    input_type: str
    timing: dict
    original_tokens: Optional[int] = None
    compression_ratio: Optional[str] = None
    tokens_saved: Optional[int] = None
    selective_decoding_stats: Optional[dict] = None
    dedup_stats: Optional[dict] = None
    offline_first: Optional[bool] = None
    payload: Optional[dict] = None


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
_RATE_LIMIT_MAX_IPS = 50_000  # max tracked IPs to prevent memory exhaustion


# ============================================================================
# App Factory
# ============================================================================


async def _rate_limit_cleanup_loop():
    """Background task to reap stale IPs without blocking the event loop."""
    while True:
        try:
            await asyncio.sleep(_RATE_LIMIT_CLEANUP_INTERVAL)
            now = time.time()
            stale_threshold = now - RATE_LIMIT_WINDOW * 2
            stale_ips = [
                ip
                for ip, timestamps in _request_counts.items()
                if not timestamps or timestamps[-1] < stale_threshold
            ]
            for ip in stale_ips:
                _request_counts.pop(ip, None)
        except asyncio.CancelledError:
            break
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage pipeline lifecycle."""
    global pipeline, start_time

    # Startup
    start_time = time.time()
    config = app.state.config

    host = os.getenv("LATENTGATE_HOST", "127.0.0.1")
    if host == "0.0.0.0" and not os.getenv("LATENTGATE_API_KEY"):
        logger.warning(
            "\n" + "!" * 80 + "\n"
            "SECURITY WARNING: Binding to 0.0.0.0 without LATENTGATE_API_KEY set.\n"
            "Your API is completely unauthenticated and accessible to anyone on your network.\n"
            "Set LATENTGATE_API_KEY in your environment to secure the server.\n" + "!" * 80
        )

    app.state.request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    app.state.stats_lock = asyncio.Lock()
    app.state.cleanup_task = asyncio.create_task(_rate_limit_cleanup_loop())
    pipeline = LatentGatePipeline(config, preload=False)
    # Warm models in the background so /health answers immediately (k8s/Docker
    # probes) instead of waiting for multi-GB models to load into GPU memory.
    if os.getenv("LATENTGATE_PRELOAD", "true").lower() in ("true", "1", "yes"):
        app.state.preload_task = asyncio.create_task(
            asyncio.to_thread(pipeline.client.preload_models)
        )
    set_pipeline_ready(True)
    logger.info("LatentGate API server started")

    yield

    # Shutdown
    if getattr(app.state, "cleanup_task", None):
        app.state.cleanup_task.cancel()
    if getattr(app.state, "preload_task", None) and not app.state.preload_task.done():
        app.state.preload_task.cancel()
    if pipeline:
        pipeline.close()
    set_pipeline_ready(False)
    logger.info("LatentGate API server stopped")


def _sanitize_error(e: Exception) -> str:
    """Convert exception to user-safe error message."""
    if isinstance(e, PathAccessError):
        return str(e)
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


def _path_access_http_error(e: PathAccessError) -> HTTPException:
    detail = str(e)
    if "disabled" in detail.lower():
        status = 403
        detail = (
            "Image path queries disabled by default. Set LATENTGATE_ALLOWED_IMAGE_ROOTS "
            "to enable, or use /query/image/upload."
        )
    else:
        status = 403
    return HTTPException(status_code=status, detail=detail)


def _validate_image_path(image_path: str, config: PipelineConfig) -> None:
    try:
        validate_image_path_access(image_path, config)
    except PathAccessError as e:
        raise _path_access_http_error(e) from e


def _check_rate_limit(request: Request) -> None:
    """Simple in-memory rate limiter per IP. Cleanup is handled externally."""
    client_ip = get_client_ip(request)
    now = time.time()

    if len(_request_counts) >= _RATE_LIMIT_MAX_IPS and client_ip not in _request_counts:
        raise HTTPException(
            status_code=429,
            detail="Server is under heavy load. Please try again later.",
        )
    _request_counts[client_ip] = [
        t for t in _request_counts.get(client_ip, []) if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_request_counts[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
        )
    _request_counts[client_ip].append(now)


async def _run_pipeline_call(func, *args, **kwargs):
    """Run blocking pipeline work off the event loop with bounded concurrency."""
    semaphore = app.state.request_semaphore
    async with semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)


async def _record_query(result: dict) -> None:
    """Update process-local stats without racing concurrent requests."""
    global query_count, total_tokens_saved
    async with app.state.stats_lock:
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
    # Also record to Prometheus metrics
    record_tokens(
        input_tokens=result.get("original_tokens", 0) or result.get("tokens_estimated", 0),
        output_tokens=result.get("tokens_estimated", 0),
        saved=result.get("tokens_saved", 0),
    )


# Error-type to HTTP-status mapping for _handle_api_error
_error_type_to_status = [
    (FileNotFoundError, 404),
    (ValueError, 400),
]
_error_502_types = (ConnectionError, TimeoutError, PermissionError, RemoteDecodeError)


def _handle_api_error(e: Exception, context: str = "API call") -> HTTPException:
    """
    Convert an exception to the appropriate HTTPException based on its type.

    Replaces the repetitive try/except/HTTPException pattern in every endpoint
    with a single call. Maps known exception types to the correct HTTP status code.
    """
    # Already an HTTP error (e.g. 403 from path validation) — keep its status
    if isinstance(e, HTTPException):
        return e

    # Exceptions that have specific status codes (client errors — don't track as server errors)
    for exc_type, status in _error_type_to_status:
        if isinstance(e, exc_type):
            return HTTPException(status_code=status, detail=_sanitize_error(e))

    # 502 Bad Gateway for network / upstream failures
    if isinstance(e, _error_502_types):
        record_error("upstream_failure")
        return HTTPException(status_code=502, detail=_sanitize_error(e))

    # 500 for everything else — log + track as server error
    record_error(context.replace(" ", "_"))
    logger.error(f"{context} failed: {e}", exc_info=True)
    return HTTPException(status_code=500, detail=_sanitize_error(e))


def _docs_enabled() -> bool:
    if os.getenv("LATENTGATE_DISABLE_DOCS", "").lower() in ("true", "1", "yes"):
        return False
    if os.getenv("LATENTGATE_API_KEY"):
        return os.getenv("LATENTGATE_ENABLE_DOCS", "").lower() in ("true", "1", "yes")
    return True


security = HTTPBearer(auto_error=False)


def _verify_api_key_dependency(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Dependency for FastAPI routes that need API key auth."""
    expected_key = os.getenv("LATENTGATE_API_KEY")
    if not expected_key:
        return True  # No API key configured, skip auth
    provided = credentials.credentials if credentials else None
    if not verify_api_key(provided, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


def clean_json_quotes(body_str: str) -> str:
    """Escape unescaped double quotes inside JSON string values."""
    import re

    keys_to_clean = ["text", "question", "image_path", "image", "new_question"]
    cleaned = body_str

    for key in keys_to_clean:
        start_pat = rf'"{key}"\s*:\s*"'
        match = re.search(start_pat, cleaned)
        if match:
            start_idx = match.end()
            rest = cleaned[start_idx:]

            # Find the boundary: another key like "other_key":
            boundary_idx = len(rest)
            next_key_match = re.search(r'"\w+"\s*:', rest)
            if next_key_match:
                boundary_idx = next_key_match.start()

            # Find candidates for the closing quote: any quote before the boundary
            # followed by optional whitespace and , or }
            end_candidates = []
            for m in re.finditer(r'"\s*(?:,|\})', rest[:boundary_idx]):
                end_candidates.append(m.start())

            if end_candidates:
                # The correct closing quote is the last candidate before the next key
                end_idx = end_candidates[-1]
                inner_value = rest[:end_idx]

                # Escape any unescaped quotes in the inner value
                escaped_inner = re.sub(r'(?<!\\)"', r"\"", inner_value)

                # Reconstruct
                cleaned = cleaned[:start_idx] + escaped_inner + rest[end_idx:]
    return cleaned


class SafeJSONASGIMiddleware:
    """ASGI Middleware that intercepts and cleans unescaped quotes in JSON request bodies."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT"):
            # Check headers
            headers = dict(scope.get("headers", []))
            content_type = headers.get(b"content-type", b"").decode("utf-8")
            if "application/json" in content_type:
                # Buffer request body
                body_parts = []
                more_body = True
                while more_body:
                    message = await receive()
                    body_parts.append(message.get("body", b""))
                    more_body = message.get("more_body", False)

                body_bytes = b"".join(body_parts)

                # Try to repair unescaped quotes; leave undecodable bodies for
                # FastAPI to reject with a normal 422 instead of crashing here.
                try:
                    body_str = body_bytes.decode("utf-8")
                    json.loads(body_str)
                except UnicodeDecodeError:
                    pass
                except json.JSONDecodeError:
                    body_bytes = clean_json_quotes(body_str).encode("utf-8")

                # Re-create receive channel
                body_offset = 0

                async def new_receive():
                    nonlocal body_offset
                    chunk = body_bytes[body_offset:]
                    body_offset = len(body_bytes)
                    return {
                        "type": "http.request",
                        "body": chunk,
                        "more_body": False,
                    }

                await self.app(scope, new_receive, send)
                return

        await self.app(scope, receive, send)


def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    from latent_gate import __version__

    docs_enabled = _docs_enabled()
    new_app = FastAPI(
        title="LatentGate API",
        description="Local-first vision-language pipeline API. Compress images, text, and documents locally before sending to cloud LLMs.",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    # CORS — configurable origins in production
    cors_origins_str = os.getenv("LATENTGATE_CORS_ORIGINS", "")
    if cors_origins_str:
        cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    else:
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]  # Default to local frontend

    new_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    new_app.add_middleware(SafeJSONASGIMiddleware)

    # Metrics — mount /metrics endpoint (opt-out via LATENTGATE_ENABLE_METRICS=false)
    enable_metrics = os.getenv("LATENTGATE_ENABLE_METRICS", "true").lower() in ("true", "1", "yes")
    if enable_metrics:
        metrics_app = setup_metrics()
        if metrics_app:
            new_app.mount("/metrics", metrics_app)
            new_app.add_middleware(MetricsMiddleware)
            logger.info("Metrics endpoint mounted at /metrics")
        else:
            logger.info(
                "prometheus-client not installed — install with: pip install latent-gate[metrics]"
            )
    else:
        logger.info("Metrics disabled via LATENTGATE_ENABLE_METRICS=false")

    # HTTPException handler — return the original status code
    @new_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    # Global exception handler for unhandled errors
    @new_app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        record_error("unhandled")
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )

    # Store config for later use (env + config file overrides)
    new_app.state.config = config or get_config()

    new_app.include_router(public_router)
    # Apply API key auth to the api_router via a wrapping router
    api_router_with_auth = APIRouter(dependencies=[Depends(_verify_api_key_dependency)])
    api_router_with_auth.include_router(api_router)
    new_app.include_router(api_router_with_auth)
    return new_app


# ============================================================================
# Routers (populated by the endpoint decorators below)
# ============================================================================

public_router = APIRouter()
api_router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================


@public_router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Unauthenticated health check for load balancers and Docker."""
    _check_rate_limit(request)

    ollama_connected = False
    models_loaded = False
    detailed = not os.getenv("LATENTGATE_API_KEY")

    if detailed:
        try:
            import requests

            def _check_ollama():
                resp = requests.get(f"{app.state.config.ollama_base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return True, len(models) > 0
                return False, False

            ollama_connected, models_loaded = await asyncio.to_thread(_check_ollama)
        except Exception as e:
            logger.debug(f"Ollama health check suppressed error: {e}")

    from latent_gate import __version__

    return HealthResponse(
        status="healthy",
        version=__version__ if detailed else "protected",
        ollama_connected=ollama_connected,
        models_loaded=models_loaded,
    )


@api_router.get("/stats", response_model=StatsResponse)
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


@api_router.post("/query/image", response_model=QueryResponse)
async def query_image(request: ImageQueryRequest, http_request: Request):
    """Process an image and answer a question about it."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        _validate_image_path(request.image_path, http_request.app.state.config)
        result = await _run_pipeline_call(pipeline.query, request.image_path, request.question)
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Image query")


@api_router.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest, http_request: Request):
    """Compress text and query the remote LLM."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        validate_text_length(request.text)
        result = await _run_pipeline_call(
            pipeline.query_text,
            request.text,
            request.question,
            request.mode,
        )
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Text query")


@api_router.post("/query/conversation", response_model=QueryResponse)
async def query_conversation(request: ConversationQueryRequest, http_request: Request):
    """Compress conversation history and ask a new question."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await _run_pipeline_call(
            pipeline.query_conversation,
            request.messages,
            request.new_question,
        )
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Conversation query")


@api_router.post("/query/documents", response_model=QueryResponse)
async def query_documents(request: DocumentsQueryRequest, http_request: Request):
    """Compress RAG documents and answer a question."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await _run_pipeline_call(
            pipeline.query_documents,
            request.documents,
            request.question,
        )
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Documents query")


@api_router.post("/query/universal", response_model=QueryResponse)
async def query_universal(request: UniversalQueryRequest, http_request: Request):
    """Universal query endpoint - auto-detects input type."""
    global pipeline, query_count, total_tokens_saved
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        if request.image:
            _validate_image_path(request.image, http_request.app.state.config)
        result = await _run_pipeline_call(
            pipeline.query_universal,
            text=request.text,
            image=request.image,
            question=request.question,
        )
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Universal query")


@api_router.post("/query/image/upload", response_model=QueryResponse)
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

        result = await _run_pipeline_call(pipeline.query, tmp_path, question)
        await _record_query(result)

        return QueryResponse(**redact_query_result(result))

    except Exception as e:
        raise _handle_api_error(e, "Image upload query")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Failed to clean up temp file: {tmp_path}")


# ============================================================================
# Batch Compression Endpoint
# ============================================================================


class BatchCompressRequest(BaseModel):
    """Request model for batch prompt compression."""

    texts: List[str] = Field(
        ..., min_length=1, max_length=100, description="Array of verbose prompts to compress"
    )


class BatchCompressItem(BaseModel):
    """Single item in batch compression response."""

    index: int
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: str
    processing_time_ms: float
    error: Optional[str] = None


class BatchCompressResponse(BaseModel):
    """Response model for batch compression."""

    results: List[BatchCompressItem]
    total_original_tokens: int
    total_compressed_tokens: int
    total_tokens_saved: int
    average_compression_ratio: str
    total_processing_time_ms: float


@api_router.post("/compress/batch", response_model=BatchCompressResponse)
async def compress_batch(request: BatchCompressRequest, http_request: Request):
    """Compress multiple verbose prompts in parallel."""
    global pipeline
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        start_time = time.time()

        async def _compress_single(index: int, text: str) -> BatchCompressItem:
            try:
                result = await _run_pipeline_call(pipeline.compress_prompt, text)
                return BatchCompressItem(
                    index=index,
                    original_prompt=result.get("original_prompt", text),
                    compressed_prompt=result.get("compressed_prompt", ""),
                    original_tokens=result.get("original_tokens", 0),
                    compressed_tokens=result.get("compressed_tokens", 0),
                    tokens_saved=result.get("tokens_saved", 0),
                    compression_ratio=result.get("compression_ratio", "1.0x"),
                    processing_time_ms=result.get("processing_time_ms", 0),
                )
            except Exception as e:
                return BatchCompressItem(
                    index=index,
                    original_prompt=text,
                    compressed_prompt="",
                    original_tokens=0,
                    compressed_tokens=0,
                    tokens_saved=0,
                    compression_ratio="0.0x",
                    processing_time_ms=0,
                    error=str(e),
                )

        tasks = [_compress_single(i, text) for i, text in enumerate(request.texts)]
        results = await asyncio.gather(*tasks)

        total_time = (time.time() - start_time) * 1000
        total_original = sum(r.original_tokens for r in results)
        total_compressed = sum(r.compressed_tokens for r in results)
        total_saved = sum(r.tokens_saved for r in results)
        avg_ratio = total_original / max(total_compressed, 1)

        await _record_query({"tokens_saved": total_saved})

        return BatchCompressResponse(
            results=sorted(results, key=lambda r: r.index),
            total_original_tokens=total_original,
            total_compressed_tokens=total_compressed,
            total_tokens_saved=total_saved,
            average_compression_ratio=f"{avg_ratio:.1f}x",
            total_processing_time_ms=round(total_time, 1),
        )

    except Exception as e:
        raise _handle_api_error(e, "Batch compression")


# ============================================================================
# OpenAI-Compatible API Wrapper
# ============================================================================


class OpenAIChatMessage(BaseModel):
    """OpenAI-compatible chat message format."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(
        "latent-gate", description="Model identifier (ignored, always uses LatentGate compression)"
    )
    messages: List[OpenAIChatMessage] = Field(..., min_length=1, description="Chat messages")
    temperature: Optional[float] = Field(None, ge=0, le=2, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")


class OpenAIUsage(BaseModel):
    """OpenAI-compatible token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""

    index: int = 0
    message: OpenAIChatMessage
    finish_reason: str = "stop"


class OpenAIChatResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = "chatcmpl-latent-gate"
    object: str = "chat.completion"
    created: int = 0
    model: str = "latent-gate"
    choices: List[OpenAIChatChoice]
    usage: OpenAIUsage


@api_router.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(request: OpenAIChatRequest, http_request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Accepts the standard OpenAI request format, runs LatentGate compression
    on the conversation, then returns a response. This lets any tool that
    supports OpenAI's API automatically use LatentGate compression.
    """
    global pipeline
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate before try block so HTTPException bypasses error handling
    user_messages = [m for m in request.messages if m.role == "user"]
    system_message = next((m.content for m in request.messages if m.role == "system"), None)

    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in request")

    try:

        # Combine all user messages for compression
        full_user_text = "\n".join(m.content for m in user_messages)
        last_question = user_messages[-1].content

        # Step 1: Compress locally via LatentGate
        compressed = await _run_pipeline_call(pipeline.compress_prompt, full_user_text)
        compressed_text = compressed.get("compressed_prompt", full_user_text)
        # System instructions are kept verbatim (not compressed) so behavior rules survive
        decoder_input = (
            f"System instructions:\n{system_message}\n\n{compressed_text}"
            if system_message
            else compressed_text
        )

        original_tokens = compressed.get("original_tokens", len(full_user_text.split()))
        compressed_tokens = compressed.get("compressed_tokens", len(compressed_text.split()))

        # Step 2: Forward compressed prompt directly to the remote decoder
        # (Skip query_text to avoid double compression and auto-detect issues)
        try:
            answer, api_usage = await _run_pipeline_call(
                pipeline.remote_decoder.decode, decoder_input, last_question
            )
            completion_tokens = api_usage.get("completion_tokens", 0) if api_usage else 0
            response_content = (
                f"[⚡ LatentGate: ~{original_tokens} → ~{compressed_tokens} tokens | "
                f"Saved ~{compressed.get('tokens_saved', 0)} | "
                f"{compressed.get('compression_ratio', '1.0x')}x compression]\n\n"
                f"{answer}"
            )
            return OpenAIChatResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=f"latent-gate+{pipeline.config.remote_provider}",
                choices=[
                    OpenAIChatChoice(
                        index=0,
                        message=OpenAIChatMessage(
                            role="assistant",
                            content=response_content,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=compressed_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=compressed_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            logger.warning(f"Remote LLM call failed, returning compressed prompt: {e}")

        # Fallback: return compressed prompt directly if remote unavailable
        response_content = (
            f"[⚡ LatentGate: ~{original_tokens} → ~{compressed_tokens} tokens | "
            f"Saved ~{compressed.get('tokens_saved', 0)} | "
            f"{compressed.get('compression_ratio', '1.0x')}x compression]\n\n"
            f"{compressed_text}"
        )

        return OpenAIChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model="latent-gate",
            choices=[
                OpenAIChatChoice(
                    index=0,
                    message=OpenAIChatMessage(
                        role="assistant",
                        content=response_content,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=compressed_tokens,
                completion_tokens=0,
                total_tokens=compressed_tokens,
            ),
        )

    except Exception as e:
        raise _handle_api_error(e, "OpenAI-compatible endpoint")


# ============================================================================
# WebSocket Endpoint
# ============================================================================


@public_router.websocket("/ws/compress")
async def websocket_compress(websocket: WebSocket):
    """
    WebSocket endpoint for real-time compression.

    Send a JSON message with {"text": "..."} and receive streaming
    compression progress updates, then the final result.

    Message flow:
      Client -> {"text": "Your long prompt here..."}
      Server <- {"status": "compressing", "original_tokens": 500}
      Server <- {"status": "progress", "stage": "extracting|processing|done"}
      Server <- {"status": "complete", "result": {...}}
    """
    # Browsers cannot set headers on WebSocket upgrades, so also accept ?api_key=
    expected_key = os.getenv("LATENTGATE_API_KEY")
    if expected_key:
        auth = websocket.headers.get("authorization", "")
        provided = (
            auth[7:]
            if auth.lower().startswith("bearer ")
            else websocket.query_params.get("api_key")
        )
        if not verify_api_key(provided, expected_key):
            await websocket.close(code=1008, reason="Invalid or missing API Key")
            return

    await websocket.accept()
    try:
        while True:
            # Receive text input
            data = await websocket.receive_json()
            text = data.get("text", "") if isinstance(data, dict) else ""
            if not text or not isinstance(text, str):
                await websocket.send_json({"status": "error", "message": "No text provided"})
                continue
            try:
                validate_text_length(text)
            except ValueError as e:
                await websocket.send_json({"status": "error", "message": str(e)})
                continue

            if not pipeline:
                await websocket.send_json(
                    {"status": "error", "message": "Pipeline not initialized"}
                )
                continue

            # Estimate tokens
            est_tokens = max(1, len(text.split()))
            await websocket.send_json(
                {
                    "status": "compressing",
                    "original_tokens": est_tokens,
                    "stage": "extracting",
                }
            )

            try:
                # Off the event loop, sharing the HTTP endpoints' concurrency cap
                result = await _run_pipeline_call(pipeline.compress_prompt, text)

                await websocket.send_json(
                    {
                        "status": "progress",
                        "stage": "done",
                        "compressed_tokens": result.get("compressed_tokens", 0),
                        "tokens_saved": result.get("tokens_saved", 0),
                    }
                )

                # Send final result
                await websocket.send_json(
                    {
                        "status": "complete",
                        "result": {
                            "original_prompt": (
                                result.get("original_prompt", text)[:200] + "..."
                                if len(result.get("original_prompt", text)) > 200
                                else result.get("original_prompt", text)
                            ),
                            "compressed_prompt": result.get("compressed_prompt", ""),
                            "original_tokens": result.get("original_tokens", 0),
                            "compressed_tokens": result.get("compressed_tokens", 0),
                            "tokens_saved": result.get("tokens_saved", 0),
                            "compression_ratio": result.get("compression_ratio", "1.0x"),
                            "processing_time_ms": result.get("processing_time_ms", 0),
                        },
                    }
                )

            except Exception as e:
                await websocket.send_json(
                    {
                        "status": "error",
                        "message": f"Compression failed: {str(e)}",
                    }
                )

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# Prompt Compression Endpoint
# ============================================================================


class CompressRequest(BaseModel):
    """Request model for prompt compression."""

    text: str = Field(..., max_length=100000, description="Verbose prompt to compress")


class CompressResponse(BaseModel):
    """Response model for prompt compression."""

    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: str
    processing_time_ms: float
    method: Optional[str] = Field(None, description="optimizer/<level> or llm/<model>")


@public_router.post(
    "/compress",
    response_model=CompressResponse,
    dependencies=[Depends(_verify_api_key_dependency)],
)
async def compress_prompt(request: CompressRequest, http_request: Request):
    """Compress a verbose prompt without calling the cloud LLM.

    Open when LATENTGATE_API_KEY is unset (used by the website demo);
    requires the bearer key once LATENTGATE_API_KEY is configured.
    """
    global pipeline
    _check_rate_limit(http_request)

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await _run_pipeline_call(pipeline.compress_prompt, request.text)
        return CompressResponse(**result)
    except Exception as e:
        raise _handle_api_error(e, "Prompt compression")


# ============================================================================
# Default app instance
# ============================================================================

# Must be created AFTER every route above is registered: include_router()
# copies routes at call time, so routers populated later would be empty.
app = create_app()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """Run the API server."""
    import uvicorn

    # Override from environment
    import os

    host = os.getenv("LATENTGATE_HOST", "127.0.0.1")
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
