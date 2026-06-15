"""
Remote Decoder (Y-Decoder) — Cloud LLM API integration.

This is the only PAID component of the pipeline. It receives the
compact SemanticPayload (~150-200 tokens) instead of the raw image
(~1000+ tokens), then generates the final answer.

Supports: OpenAI, Anthropic, Google, Ollama (local), or any custom endpoint.
"""

import os
import logging
from abc import ABC, abstractmethod

import requests

from latent_gate.config import PipelineConfig


logger = logging.getLogger("latent_gate.remote")


# System prompt for the cloud LLM — tells it to work with pre-analyzed data
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
        """
        Send compact semantic payload + user query to cloud LLM.

        Args:
            compact_input: The compact prompt from SemanticPayload.to_compact_prompt()
            user_query: The user's original question about the image.

        Returns:
            The LLM's text response.
        """
        pass


class OpenAIDecoder(RemoteDecoder):
    """Y-Decoder using OpenAI-compatible API (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
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
                {
                    "role": "user",
                    "content": (
                        f"[SCENE DATA]: {compact_input}\n\n"
                        f"[QUESTION]: {user_query}"
                    ),
                },
            ],
            "max_tokens": 500,
            "temperature": 0.3,
        }

        logger.info(f"OpenAI request to {self.config.remote_model}")
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=self.config.request_timeout
        )
        resp.raise_for_status()

        answer = resp.json()["choices"][0]["message"]["content"]
        logger.info(f"OpenAI response: {len(answer)} chars")
        return answer


class AnthropicDecoder(RemoteDecoder):
    """Y-Decoder using Anthropic API (Claude)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
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
                {
                    "role": "user",
                    "content": (
                        f"[SCENE DATA]: {compact_input}\n\n"
                        f"[QUESTION]: {user_query}"
                    ),
                }
            ],
        }

        logger.info(f"Anthropic request to {self.config.remote_model}")
        resp = requests.post(
            url, headers=headers, json=payload,
            timeout=self.config.request_timeout
        )
        resp.raise_for_status()

        answer = resp.json()["content"][0]["text"]
        logger.info(f"Anthropic response: {len(answer)} chars")
        return answer


class GoogleDecoder(RemoteDecoder):
    """Y-Decoder using Google Gemini API."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.api_key = config.remote_api_key or os.getenv("GOOGLE_API_KEY", "")

    def decode(self, compact_input: str, user_query: str) -> str:
        model = self.config.remote_model or "gemini-2.0-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                f"{SYSTEM_PROMPT}\n\n"
                                f"[SCENE DATA]: {compact_input}\n\n"
                                f"[QUESTION]: {user_query}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
            },
        }

        logger.info(f"Google request to {model}")
        resp = requests.post(url, json=payload, timeout=self.config.request_timeout)
        resp.raise_for_status()

        answer = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        logger.info(f"Google response: {len(answer)} chars")
        return answer


class OllamaRemoteDecoder(RemoteDecoder):
    """
    Y-Decoder using another Ollama model (fully local, zero cost).

    Use this when you want the entire pipeline to be free.
    The 'remote' model can be a different (possibly larger) model
    than the predictor.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def decode(self, compact_input: str, user_query: str) -> str:
        url = f"{self.config.ollama_base_url}/api/generate"

        payload = {
            "model": self.config.remote_model,
            "prompt": (
                f"{SYSTEM_PROMPT}\n\n"
                f"[SCENE DATA]: {compact_input}\n\n"
                f"[QUESTION]: {user_query}\n\n"
                f"Answer:"
            ),
            "stream": False,
            "options": {"temperature": 0.3},
        }

        logger.info(f"Ollama remote request to {self.config.remote_model}")
        resp = requests.post(url, json=payload, timeout=self.config.request_timeout)
        resp.raise_for_status()

        answer = resp.json().get("response", "")
        logger.info(f"Ollama remote response: {len(answer)} chars")
        return answer


def create_decoder(config: PipelineConfig) -> RemoteDecoder:
    """Factory function: create the appropriate decoder based on config."""
    provider = config.remote_provider.lower()

    decoder_map = {
        "openai": OpenAIDecoder,
        "anthropic": AnthropicDecoder,
        "google": GoogleDecoder,
        "ollama": OllamaRemoteDecoder,
    }

    decoder_class = decoder_map.get(provider, OpenAIDecoder)
    logger.info(f"Created {decoder_class.__name__} for provider '{provider}'")
    return decoder_class(config)
