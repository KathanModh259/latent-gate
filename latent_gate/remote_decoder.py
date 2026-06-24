"""
Remote Decoder (Y-Decoder) — Cloud LLM API integration.

Uses FastClient for connection pooling and session reuse.
Supports both standard and streaming responses.

All OpenAI-compatible providers (OpenAI, Groq, DeepSeek, Together, Azure)
share a single base class to eliminate code duplication.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Iterator

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


def _build_messages(compact_input: str, user_query: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"},
    ]


def _stream_sse(response) -> Iterator[str]:
    """Parse Server-Sent Events from an OpenAI-compatible streaming response."""
    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    content = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


class RemoteDecoder(ABC):
    """Abstract base class for cloud LLM API integration."""

    @abstractmethod
    def decode(self, compact_input: str, user_query: str) -> str:
        pass

    @abstractmethod
    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        pass


class OpenAICompatibleDecoder(RemoteDecoder):
    """
    Base class for all OpenAI-compatible APIs.

    Subclasses only need to set base_url, api_key, and optionally
    override the headers or default model name.
    """

    provider_name: str = "openai"
    default_base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv(self.api_key_env, "")
        self.base_url = config.remote_base_url or self.default_base_url

    @property
    def model(self) -> str:
        return self.config.remote_model or self.default_model

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, stream: bool = False) -> dict:
        p = {
            "model": self.model,
            "messages": _build_messages("", ""),
            "max_tokens": 500,
            "temperature": 0.3,
        }
        if stream:
            p["stream"] = True
        return p

    def decode(self, compact_input: str, user_query: str) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload()
        payload["messages"] = _build_messages(compact_input, user_query)
        logger.info(f"{self.provider_name} request to {self.model}")
        data = self.client.remote_post(url, self._headers(), payload)
        return data["choices"][0]["message"]["content"]

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(stream=True)
        payload["messages"] = _build_messages(compact_input, user_query)
        logger.info(f"{self.provider_name} streaming request to {self.model}")
        try:
            import requests
            response = requests.post(
                url, headers=self._headers(), json=payload, stream=True, timeout=120
            )
            response.raise_for_status()
            yield from _stream_sse(response)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


class OpenAIDecoder(OpenAICompatibleDecoder):
    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"


class GroqDecoder(OpenAICompatibleDecoder):
    provider_name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    default_model = "llama-3.1-8b-instant"
    api_key_env = "GROQ_API_KEY"


class DeepSeekDecoder(OpenAICompatibleDecoder):
    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
    api_key_env = "DEEPSEEK_API_KEY"


class TogetherDecoder(OpenAICompatibleDecoder):
    provider_name = "together"
    default_base_url = "https://api.together.xyz/v1"
    default_model = "meta-llama/Llama-3-8b-chat-hf"
    api_key_env = "TOGETHER_API_KEY"


class AzureOpenAIDecoder(RemoteDecoder):
    """Azure OpenAI uses api-key header instead of Bearer token."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.base_url = config.remote_base_url or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")

    @property
    def model(self) -> str:
        return self.config.remote_model or "gpt-4o-mini"

    def _url(self) -> str:
        return f"{self.base_url}/openai/deployments/{self.deployment}/chat/completions?api-version=2024-02-01"

    def _headers(self) -> dict:
        return {"api-key": self.api_key, "Content-Type": "application/json"}

    def decode(self, compact_input: str, user_query: str) -> str:
        payload = {
            "messages": _build_messages(compact_input, user_query),
            "max_tokens": 500,
            "temperature": 0.3,
        }
        logger.info(f"Azure OpenAI request to {self.deployment}")
        data = self.client.remote_post(self._url(), self._headers(), payload)
        return data["choices"][0]["message"]["content"]

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        payload = {
            "messages": _build_messages(compact_input, user_query),
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": True,
        }
        logger.info(f"Azure OpenAI streaming request to {self.deployment}")
        try:
            import requests
            response = requests.post(
                self._url(), headers=self._headers(), json=payload, stream=True, timeout=120
            )
            response.raise_for_status()
            yield from _stream_sse(response)
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


class AnthropicDecoder(RemoteDecoder):
    """Anthropic Claude API decoder."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("ANTHROPIC_API_KEY", "")

    @property
    def model(self) -> str:
        return self.config.remote_model or "claude-sonnet-4-20250514"

    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _payload(self, stream: bool = False) -> dict:
        p = {
            "model": self.model,
            "max_tokens": 500,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": ""}],
        }
        if stream:
            p["stream"] = True
        return p

    def decode(self, compact_input: str, user_query: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        payload = self._payload()
        payload["messages"] = [{"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
        logger.info(f"Anthropic request to {self.model}")
        data = self.client.remote_post(url, self._headers(), payload)
        return data["content"][0]["text"]

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = "https://api.anthropic.com/v1/messages"
        payload = self._payload(stream=True)
        payload["messages"] = [{"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
        logger.info(f"Anthropic streaming request to {self.model}")
        try:
            import requests
            response = requests.post(
                url, headers=self._headers(), json=payload, stream=True, timeout=120
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            if event.get("type") == "content_block_delta":
                                if event.get("delta", {}).get("text"):
                                    yield event["delta"]["text"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


class GoogleDecoder(RemoteDecoder):
    """Google Gemini API decoder."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.api_key = config.remote_api_key or os.getenv("GOOGLE_API_KEY", "")

    @property
    def model(self) -> str:
        return self.config.remote_model or "gemini-2.0-flash"

    def _content(self, compact_input: str, user_query: str) -> dict:
        return {
            "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 500},
        }

    def decode(self, compact_input: str, user_query: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        logger.info(f"Google request to {self.model}")
        data = self.client.remote_post(url, {}, self._content(compact_input, user_query))
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?key={self.api_key}"
        logger.info(f"Google streaming request to {self.model}")
        try:
            import requests
            response = requests.post(
                url, json=self._content(compact_input, user_query), stream=True, timeout=120
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if data.get("candidates"):
                            for part in data["candidates"][0].get("content", {}).get("parts", []):
                                if part.get("text"):
                                    yield part["text"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


class OllamaRemoteDecoder(RemoteDecoder):
    """Ollama local decoder (zero cost)."""

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

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        logger.info(f"Ollama streaming request to {self.config.remote_model}")
        try:
            import requests
            url = f"{self.config.ollama_base_url}/api/generate"
            payload = {
                "model": self.config.remote_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}\n\nAnswer:",
                "stream": True,
                "options": {"temperature": 0.3, "num_predict": 500},
            }
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if data.get("response"):
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise


class BedrockDecoder(RemoteDecoder):
    """AWS Bedrock decoder (requires boto3)."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = config.remote_model or "anthropic.claude-3-haiku-20240307-v1:0"

    def _body(self) -> dict:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": ""}],
        }

    def decode(self, compact_input: str, user_query: str) -> str:
        try:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=self.region)
            body = self._body()
            body["messages"] = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
            response = bedrock.invoke_model(
                body=json.dumps(body), modelId=self.model_id,
                contentType="application/json", accept="application/json",
            )
            return json.loads(response["body"].read())["content"][0]["text"]
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        try:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=self.region)
            body = self._body()
            body["messages"] = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
            response = bedrock.invoke_model_with_response_stream(
                body=json.dumps(body), modelId=self.model_id,
                contentType="application/json", accept="application/json",
            )
            for event in response["body"]:
                chunk = json.loads(event["bytes"])
                if chunk.get("type") == "content_block_delta":
                    if chunk.get("delta", {}).get("text"):
                        yield chunk["delta"]["text"]
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")


def create_decoder(config: PipelineConfig, client: FastClient = None) -> RemoteDecoder:
    """Factory function: create the appropriate decoder."""
    provider = config.remote_provider.lower()
    decoder_map = {
        "openai": OpenAIDecoder,
        "anthropic": AnthropicDecoder,
        "google": GoogleDecoder,
        "ollama": OllamaRemoteDecoder,
        "groq": GroqDecoder,
        "deepseek": DeepSeekDecoder,
        "together": TogetherDecoder,
        "azure": AzureOpenAIDecoder,
        "bedrock": BedrockDecoder,
    }
    decoder_class = decoder_map.get(provider, OpenAIDecoder)
    logger.info(f"Created {decoder_class.__name__} for '{provider}'")
    return decoder_class(config, client)
