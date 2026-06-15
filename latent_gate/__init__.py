"""
LatentGate v0.3.0 — Process Locally. Send Smart. Pay Less.
============================================================
Speed-optimized VL-JEPA inspired pipeline with connection pooling,
model preloading, parallel processing, and text compression.
"""

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.text_processor import TextProcessor, TextPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.fast_client import FastClient
from latent_gate.remote_decoder import (
    RemoteDecoder, OpenAIDecoder, AnthropicDecoder, OllamaRemoteDecoder,
)
from latent_gate.cache import PayloadCache
from latent_gate.pipeline import LatentGatePipeline

__version__ = "0.3.0"
__author__ = "Kathan Modh"

__all__ = [
    "LatentGatePipeline", "PipelineConfig", "SemanticPayload",
    "TextPayload", "TextProcessor", "LocalProcessor", "SelectiveDecoder",
    "FastClient", "RemoteDecoder", "OpenAIDecoder", "AnthropicDecoder",
    "OllamaRemoteDecoder", "PayloadCache",
]
