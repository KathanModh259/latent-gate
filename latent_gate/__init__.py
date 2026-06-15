"""
LatentGate — Process Locally. Send Smart. Pay Less.
====================================================
A VL-JEPA inspired pipeline that does heavy vision-language processing
locally (FREE via Ollama), then sends only compact semantic payloads
to cloud LLMs — cutting API token costs by ~80%.

Usage:
    from latent_gate import LatentGatePipeline, PipelineConfig

    config = PipelineConfig(
        vision_model="llava:7b",
        remote_provider="openai",
        remote_model="gpt-4o-mini",
    )
    pipeline = LatentGatePipeline(config)
    result = pipeline.query("image.jpg", "What is in this image?")
"""

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
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

__version__ = "0.1.0"
__author__ = "Kathan Modh"

__all__ = [
    "LatentGatePipeline",
    "PipelineConfig",
    "SemanticPayload",
    "LocalProcessor",
    "SelectiveDecoder",
    "RemoteDecoder",
    "OpenAIDecoder",
    "AnthropicDecoder",
    "OllamaRemoteDecoder",
    "PayloadCache",
]
