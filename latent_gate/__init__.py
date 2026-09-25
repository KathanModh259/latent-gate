"""
LatentGate v1.3.1 — Process Locally. Send Smart. Pay Less.
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
        vision_model="llava:7b",       # Image compression
        text_fast_model="phi3:mini",   # Fast text compression
        text_smart_model="qwen2:7b",   # Complex text compression
        embedding_model="nomic-embed-text",  # RAG embeddings
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

# Lazy imports — avoid slow module-level imports (e.g. numpy) on package init.
# Each sub-module is imported only when first accessed.
import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from latent_gate.config import PipelineConfig
    from latent_gate.payload import SemanticPayload
    from latent_gate.text_processor import TextProcessor, TextPayload
    from latent_gate.local_processor import LocalProcessor
    from latent_gate.selective_decoder import SelectiveDecoder
    from latent_gate.fast_client import FastClient
    from latent_gate.remote_decoder import (
        RemoteDecoder,
        RemoteDecodeError,
        OpenAIDecoder,
        AnthropicDecoder,
        GoogleDecoder,
        OllamaRemoteDecoder,
        GroqDecoder,
        DeepSeekDecoder,
        TogetherDecoder,
        AzureOpenAIDecoder,
        BedrockDecoder,
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


# Replace eager imports with lazy attribute resolution.
# This makes `import latent_gate` instant while still allowing
# `from latent_gate import LatentGatePipeline` to work normally
# (the import machinery triggers the actual import at that point).
__lazy_modules__ = {
    "PipelineConfig": "latent_gate.config",
    "SemanticPayload": "latent_gate.payload",
    "TextProcessor": "latent_gate.text_processor",
    "TextPayload": "latent_gate.text_processor",
    "LocalProcessor": "latent_gate.local_processor",
    "SelectiveDecoder": "latent_gate.selective_decoder",
    "FastClient": "latent_gate.fast_client",
    "RemoteDecoder": "latent_gate.remote_decoder",
    "RemoteDecodeError": "latent_gate.remote_decoder",
    "OpenAIDecoder": "latent_gate.remote_decoder",
    "AnthropicDecoder": "latent_gate.remote_decoder",
    "GoogleDecoder": "latent_gate.remote_decoder",
    "OllamaRemoteDecoder": "latent_gate.remote_decoder",
    "GroqDecoder": "latent_gate.remote_decoder",
    "DeepSeekDecoder": "latent_gate.remote_decoder",
    "TogetherDecoder": "latent_gate.remote_decoder",
    "AzureOpenAIDecoder": "latent_gate.remote_decoder",
    "BedrockDecoder": "latent_gate.remote_decoder",
    "PayloadCache": "latent_gate.cache",
    "LatentGatePipeline": "latent_gate.pipeline",
    "VideoProcessor": "latent_gate.video_processor",
    "VideoConfig": "latent_gate.video_processor",
    "CostTracker": "latent_gate.cost_tracker",
    "AsyncLatentGatePipeline": "latent_gate.async_pipeline",
    "load_config": "latent_gate.config_loader",
    "save_config": "latent_gate.config_loader",
    "get_config": "latent_gate.config_loader",
    "setup_logging": "latent_gate.logging_config",
    "setup_from_env": "latent_gate.logging_config",
    "ProcessorPlugin": "latent_gate.plugin_system",
    "PreProcessorPlugin": "latent_gate.plugin_system",
    "PostProcessorPlugin": "latent_gate.plugin_system",
    "SimilarityPlugin": "latent_gate.plugin_system",
    "PluginManager": "latent_gate.plugin_system",
    "get_plugin_manager": "latent_gate.plugin_system",
    "detect_language": "latent_gate.multilang",
    "detect_text_language": "latent_gate.multilang",
    "is_english": "latent_gate.multilang",
    "get_supported_languages": "latent_gate.multilang",
    "MultiLanguageProcessor": "latent_gate.multilang",
    "setup_metrics": "latent_gate.metrics",
    "MetricsMiddleware": "latent_gate.metrics",
    "TokenOptimizer": "latent_gate.optimizer",
    "OptimizationResult": "latent_gate.optimizer",
    "optimize": "latent_gate.optimizer",
    "count_tokens": "latent_gate.optimizer",
}


class _LazyPackageModule(sys.modules[__name__].__class__):
    """Module class that lazy-imports submodule attributes on demand."""

    def __getattr__(self, name):
        if name in __lazy_modules__:
            module_path = __lazy_modules__[name]
            module = importlib.import_module(module_path)
            attr = getattr(module, name)
            # Cache the result so subsequent lookups are instant
            setattr(self, name, attr)
            return attr
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _LazyPackageModule

__version__ = "1.3.1"
__author__ = "Kathan Modh"
__license__ = "Proprietary"
__url__ = "https://github.com/KathanModh259/latent-gate"

__all__ = [
    "LatentGatePipeline",
    "TokenOptimizer",
    "OptimizationResult",
    "optimize",
    "count_tokens",
    "PipelineConfig",
    "SemanticPayload",
    "TextPayload",
    "TextProcessor",
    "LocalProcessor",
    "SelectiveDecoder",
    "FastClient",
    "RemoteDecoder",
    "RemoteDecodeError",
    "OpenAIDecoder",
    "AnthropicDecoder",
    "GoogleDecoder",
    "OllamaRemoteDecoder",
    "GroqDecoder",
    "DeepSeekDecoder",
    "TogetherDecoder",
    "AzureOpenAIDecoder",
    "BedrockDecoder",
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
