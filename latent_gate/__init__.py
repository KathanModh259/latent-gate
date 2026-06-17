"""
LatentGate v0.4.0 — Process Locally. Send Smart. Pay Less.
============================================================
VL-JEPA inspired pipeline that compresses images, text, conversations,
and RAG documents locally via Ollama, then sends compact payloads to
any LLM API.

New in v0.4.0:
  - MCP Server (Claude Desktop, Cursor, Cline, Continue, Zed)
  - Claude Code Skill
  - OpenAI/Anthropic function schemas

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

__version__ = "0.4.0"
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
]
