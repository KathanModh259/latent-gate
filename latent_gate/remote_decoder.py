"""
Remote Decoder (Y-Decoder) — Cloud LLM API integration.
Uses FastClient for connection pooling and session reuse.
"""

import os
import logging
from abc import ABC, abstractmethod

from latent_gate.config import PipelineConfig
from latent_gate.fast_client import FastClient


logger = logging.getLogger("latent_gate.remote")


SYSTEM_PROMPT = (
    "You receive pre-analyzed visual scene data in a structured compact format. "
    "This data was extracted by a local vision model from an image or video frame. "
    "Use this structured data to answer the user's question accurately and helpfully. "
    "Do not mention that you received pre-processed data — respond naturally as if "
    "you analyzed the image yourself."
)


class RemoteDecoder(ABC):
    """Abstract base class for cloud LLM API integration."""

    @abstractmethod
    def decode(self, compact_input: str, user_query: str) -> str:
        pass


class OpenAIDecoder(RemoteDecoder):
    """Y-Decoder using OpenAI-compatible API."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = config.remote_base_url or "https://api.openai.com/v1"

    def decode(self, compact_input: str, user_query: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.remote_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"},
            ],
            "max_tokens": 500,
            "temperature": 0.3,
        }
        logger.info(f"OpenAI request to {self.config.remote_model}")
        data = self.client.remote_post(url, headers, payload)
        return data["choices"][0]["message"]["content"]


class AnthropicDecoder(RemoteDecoder):
    """Y-Decoder using Anthropic API (Claude)."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def decode(self, compact_input: str, user_query: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.remote_model,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"},
            ],
        }
        logger.info(f"Anthropic request to {self.config.remote_model}")
        data = self.client.remote_post(url, headers, payload)
        return data["content"][0]["text"]


class GoogleDecoder(RemoteDecoder):
    """Y-Decoder using Google Gemini API."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("GOOGLE_API_KEY", "")

    def decode(self, compact_input: str, user_query: str) -> str:
        model = self.config.remote_model or "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
        }
        logger.info(f"Google request to {model}")
        data = self.client.remote_post(url, {}, payload)
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaRemoteDecoder(RemoteDecoder):
    """Y-Decoder using Ollama (fully local, zero cost)."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)

    def decode(self, compact_input: str, user_query: str) -> str:
        logger.info(f"Ollama remote request to {self.config.remote_model}")
        return self.client.ollama_generate(
            model=self.config.remote_model,
            prompt=f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}\n\nAnswer:",
            max_tokens=500,
        )


def create_decoder(config: PipelineConfig, client: FastClient = None) -> RemoteDecoder:
    """Factory function: create the appropriate decoder."""
    provider = config.remote_provider.lower()
    decoder_map = {
        "openai": OpenAIDecoder,
        "anthropic": AnthropicDecoder,
        "google": GoogleDecoder,
        "ollama": OllamaRemoteDecoder,
    }
    decoder_class = decoder_map.get(provider, OpenAIDecoder)
    logger.info(f"Created {decoder_class.__name__} for '{provider}'")
    return decoder_class(config, client)
