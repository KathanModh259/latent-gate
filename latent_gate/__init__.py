"""
LatentGate v1.0.0 — Process Locally. Send Smart. Pay Less.
============================================================
VL-JEPA inspired pipeline that compresses images, text, conversations,
and RAG documents locally via Ollama, then sends compact payloads to
any LLM API.

New in v1.0.0:
  - True embedding similarity (cosine similarity via sentence-transformers)
  - FastAPI server wrapper for web applications
  - Direct video file input with automatic frame extraction
  - Cost tracking dashboard with analytics
  - Async support for non-blocking operations
  - Batch processing optimization
  - Streaming responses
  - Configuration persistence (YAML/TOML)
  - Structured logging
  - Docker support
  - Plugin system for custom processors
  - Multi-language support

Usage:
    from latent_gate import LatentGatePipeline, PipelineConfig

    config = PipelineConfig(
        vision_model="llava:7b",
        remote_provider="openai",
        remote_model="gpt-4o-mini",
    )

    with LatentGatePipeline(config) as pipeline:
        # Image
        result = pipeline.query("photo.jpg", "What is this?")

        # Text compression
        result = pipeline.query_text("Long prompt here...")

        # Conversation history
        result = pipeline.query_conversation(messages, "Follow-up?")

        # RAG documents
        result = pipeline.query_documents(["doc1", "doc2"], "Question?")

        # Universal (auto-detect)
        result = pipeline.query_universal(text="...", image="photo.jpg")
        
    # Video processing
    from latent_gate import VideoProcessor, VideoConfig
    
    video_config = VideoConfig(fps=1.0, max_frames=50)
    with VideoProcessor(config, video_config) as processor:
        result = processor.process_video("video.mp4", "Describe the action")
"""

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.text_processor import TextProcessor, TextPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.fast_client import FastClient
from latent_gate.remote_decoder import (
    RemoteDecoder,
    OpenAIDecoder,
    AnthropicDecoder,
    OllamaRemoteDecoder,
)
from latent_gate.cache import PayloadCache
from latent_gate.pipeline import LatentGatePipeline
from latent_gate.video_processor import VideoProcessor, VideoConfig
from latent_gate.cost_tracker import CostTracker
from latent_gate.async_pipeline import AsyncLatentGatePipeline
from latent_gate.config_loader import load_config, save_config, get_config
from latent_gate.logging_config import setup_logging, setup_from_env
from latent_gate.plugin_system import (
    ProcessorPlugin,
    PreProcessorPlugin,
    PostProcessorPlugin,
    SimilarityPlugin,
    PluginManager,
    get_plugin_manager,
)
from latent_gate.multilang import (
    detect_language,
    detect_text_language,
    is_english,
    get_supported_languages,
    MultiLanguageProcessor,
)

__version__ = "1.0.0"
__author__ = "Kathan Modh"
__license__ = "MIT"
__url__ = "https://github.com/KathanModh259/latent-gate"

__all__ = [
    "LatentGatePipeline",
    "PipelineConfig",
    "SemanticPayload",
    "TextPayload",
    "TextProcessor",
    "LocalProcessor",
    "SelectiveDecoder",
    "FastClient",
    "RemoteDecoder",
    "OpenAIDecoder",
    "AnthropicDecoder",
    "OllamaRemoteDecoder",
    "PayloadCache",
    "VideoProcessor",
    "VideoConfig",
    "CostTracker",
    "AsyncLatentGatePipeline",
    "load_config",
    "save_config",
    "get_config",
    "setup_logging",
    "setup_from_env",
    "ProcessorPlugin",
    "PreProcessorPlugin",
    "PostProcessorPlugin",
    "SimilarityPlugin",
    "PluginManager",
    "get_plugin_manager",
    "detect_language",
    "detect_text_language",
    "is_english",
    "get_supported_languages",
    "MultiLanguageProcessor",
]
