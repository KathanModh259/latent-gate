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
import time
from typing import Optional, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI dependencies not installed.\n"
        "Install with: pip install latent-gate[api]\n"
        "Or directly:  pip install fastapi uvicorn python-multipart"
    ) from e

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline


logger = logging.getLogger("latent_gate.api")


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


# ============================================================================
# App Factory
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage pipeline lifecycle."""
    global pipeline, start_time
    
    # Startup
    start_time = time.time()
    config = PipelineConfig()
    pipeline = LatentGatePipeline(config, preload=True)
    logger.info("LatentGate API server started")
    
    yield
    
    # Shutdown
    if pipeline:
        pipeline.close()
    logger.info("LatentGate API server stopped")


def create_app(config: Optional[PipelineConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global pipeline, start_time
    
    app = FastAPI(
        title="LatentGate API",
        description="Local-first vision-language pipeline API. Compress images, text, and documents locally before sending to cloud LLMs.",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
        import requests
        resp = requests.get(f"{app.state.config.ollama_base_url}/api/tags", timeout=5)
        ollama_connected = resp.status_code == 200
        if ollama_connected:
            models = resp.json().get("models", [])
            models_loaded = len(models) > 0
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        ollama_connected=ollama_connected,
        models_loaded=models_loaded,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get usage statistics."""
    global query_count, total_tokens_saved
    
    return StatsResponse(
        total_queries=query_count,
        total_tokens_saved=total_tokens_saved,
        average_compression_ratio=0.0,  # TODO: Track this
        uptime_seconds=time.time() - start_time,
    )


@app.post("/query/image", response_model=QueryResponse)
async def query_image(request: ImageQueryRequest):
    """Process an image and answer a question about it."""
    global pipeline, query_count, total_tokens_saved
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query(request.image_path, request.question)
        
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
        
        return QueryResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Image query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/text", response_model=QueryResponse)
async def query_text(request: TextQueryRequest):
    """Compress text and query the remote LLM."""
    global pipeline, query_count, total_tokens_saved
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query_text(request.text, request.question, request.mode)
        
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Text query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/conversation", response_model=QueryResponse)
async def query_conversation(request: ConversationQueryRequest):
    """Compress conversation history and ask a new question."""
    global pipeline, query_count, total_tokens_saved
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query_conversation(request.messages, request.new_question)
        
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Conversation query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/documents", response_model=QueryResponse)
async def query_documents(request: DocumentsQueryRequest):
    """Compress RAG documents and answer a question."""
    global pipeline, query_count, total_tokens_saved
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.query_documents(request.documents, request.question)
        
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Documents query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/universal", response_model=QueryResponse)
async def query_universal(request: UniversalQueryRequest):
    """Universal query endpoint - auto-detects input type."""
    global pipeline, query_count, total_tokens_saved
    
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Universal query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/image/upload", response_model=QueryResponse)
async def query_image_upload(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Upload an image and query it."""
    import tempfile
    import os
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = pipeline.query(tmp_path, question)
        
        global query_count, total_tokens_saved
        query_count += 1
        if "tokens_saved" in result:
            total_tokens_saved += result["tokens_saved"]
        
        return QueryResponse(**result)
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    PipelineConfig()
    
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
