"""
LatentGate — Process Locally. Send Smart. Pay Less.
====================================================
A VL-JEPA inspired pipeline that compresses images AND text locally
(FREE via Ollama), then sends only compact semantic payloads to cloud LLMs.

Usage:
    from latent_gate import LatentGatePipeline, PipelineConfig

    pipeline = LatentGatePipeline(PipelineConfig(remote_provider="ollama"))

    # Image mode
    result = pipeline.query("image.jpg", "What is this?")

    # Text mode (NEW)
    result = pipeline.query_text("Your long 500-word prompt here...")

    # Conversation mode (NEW)
    result = pipeline.query_conversation(messages, "Follow-up question")

    # RAG Document mode (NEW)
    result = pipeline.query_documents(["doc1...", "doc2..."], "Question?")

    # Universal (auto-detect)
    result = pipeline.query_universal(text="...", image="photo.jpg")
"""

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.text_processor import TextProcessor, TextPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import (
    RemoteDecoder,
    OpenAIDecoder,
    AnthropicDecoder,
    OllamaRemoteDecoder,
)
from latent_gate.cache import PayloadCache
from latent_gate.pipeline import LatentGatePipeline

__version__ = "0.2.0"
__author__ = "Kathan Modh"

__all__ = [
    "LatentGatePipeline",
    "PipelineConfig",
    "SemanticPayload",
    "TextPayload",
    "TextProcessor",
    "LocalProcessor",
    "SelectiveDecoder",
    "RemoteDecoder",
    "OpenAIDecoder",
    "AnthropicDecoder",
    "OllamaRemoteDecoder",
    "PayloadCache",
]
