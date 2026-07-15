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
from typing import Iterator, Tuple

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


class RemoteDecodeError(Exception):
    """Raised when a remote LLM API call fails."""

    def __init__(self, message: str, status_code: int = 0, provider: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


def _extract_content_from_openai(data: dict, provider: str = "openai") -> Tuple[str, dict]:
    """Safely extract content and usage from OpenAI-compatible API response."""
    try:
        choices = data.get("choices")
        if not choices:
            error_msg = data.get("error", {}).get("message", "No choices in response")
            raise RemoteDecodeError(
                f"{provider} returned no choices: {error_msg}",
                status_code=data.get("error", {}).get("code", 0),
                provider=provider,
            )
        first = choices[0]
        message = first.get("message") or first.get("delta", {})
        content = message.get("content")
        if not content:
            finish_reason = first.get("finish_reason", "unknown")
            if finish_reason == "length":
                raise RemoteDecodeError(
                    f"{provider} response truncated (max_tokens too low)",
                    provider=provider,
                )
            elif finish_reason == "content_filter":
                raise RemoteDecodeError(
                    f"{provider} response blocked by content filter",
                    provider=provider,
                )
            raise RemoteDecodeError(
                f"{provider} returned empty content (finish_reason={finish_reason})",
                provider=provider,
            )
        # Extract real token usage from provider response
        raw_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("prompt_tokens", 0),
            "completion_tokens": raw_usage.get("completion_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
            "source": "provider" if raw_usage else "unavailable",
        }
        return content, usage
    except RemoteDecodeError:
        raise
    except Exception as e:
        raise RemoteDecodeError(
            f"Failed to parse {provider} response: {e}",
            provider=provider,
        ) from e


def _extract_content_from_anthropic(data: dict) -> Tuple[str, dict]:
    """Safely extract content and usage from Anthropic API response."""
    try:
        content_blocks = data.get("content")
        if not content_blocks:
            error_type = data.get("error", {}).get("type", "unknown")
            error_msg = data.get("error", {}).get("message", "No content in response")
            raise RemoteDecodeError(
                f"Anthropic returned no content ({error_type}): {error_msg}",
                provider="anthropic",
            )
        texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        result = " ".join(texts).strip()
        if not result:
            raise RemoteDecodeError(
                "Anthropic returned empty text content",
                provider="anthropic",
            )
        # Extract real token usage from Anthropic response
        raw_usage = data.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
            "source": "provider" if raw_usage else "unavailable",
        }
        return result, usage
    except RemoteDecodeError:
        raise
    except Exception as e:
        raise RemoteDecodeError(
            f"Failed to parse Anthropic response: {e}",
            provider="anthropic",
        ) from e


def _extract_content_from_google(data: dict) -> Tuple[str, dict]:
    """Safely extract content and usage from Google Gemini API response."""
    try:
        candidates = data.get("candidates")
        if not candidates:
            raise RemoteDecodeError(
                "Google Gemini returned no candidates",
                provider="google",
            )
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if p.get("text")]
        result = " ".join(texts).strip()
        if not result:
            raise RemoteDecodeError(
                "Google Gemini returned empty text",
                provider="google",
            )
        # Extract real token usage from Google response (usageMetadata)
        raw_usage = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": raw_usage.get("promptTokenCount", 0),
            "completion_tokens": raw_usage.get("candidatesTokenCount", 0),
            "total_tokens": raw_usage.get("totalTokenCount", 0),
            "source": "provider" if raw_usage else "unavailable",
        }
        return result, usage
    except RemoteDecodeError:
        raise
    except Exception as e:
        raise RemoteDecodeError(
            f"Failed to parse Google response: {e}",
            provider="google",
        ) from e


def _build_messages(compact_input: str, user_query: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"},
    ]


def _stream_sse(response) -> Iterator[str]:
    """Parse Server-Sent Events from an OpenAI-compatible streaming response."""
    try:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choices = chunk.get("choices", [])
                        if choices:
                            content = (
                                choices[0]
                                .get("delta", {})
                                .get("content")
                            )
                            if content:
                                yield content
                    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                        continue
    except Exception as e:
        logger.error(f"Stream parsing failed: {e}")
        return


class RemoteDecoder(ABC):
    """Abstract base class for cloud LLM API integration."""

    @abstractmethod
    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        """Decode and return (answer, usage_dict)."""
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

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload()
        payload["messages"] = _build_messages(compact_input, user_query)
        logger.info(f"{self.provider_name} request to {self.model}")
        data = self.client.remote_post(url, self._headers(), payload)
        return _extract_content_from_openai(data, self.provider_name)

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(stream=True)
        payload["messages"] = _build_messages(compact_input, user_query)
        logger.info(f"{self.provider_name} streaming request to {self.model}")
        try:
            response = self.client.post_stream(url, self._headers(), payload)
            yield from _stream_sse(response)
        except RemoteDecodeError:
            raise
        except Exception as e:
            logger.error(f"Streaming failed for {self.provider_name}: {e}")
            raise RemoteDecodeError(
                f"{self.provider_name} streaming failed: {e}",
                provider=self.provider_name,
            ) from e


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

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        payload = {
            "messages": _build_messages(compact_input, user_query),
            "max_tokens": 500,
            "temperature": 0.3,
        }
        logger.info(f"Azure OpenAI request to {self.deployment}")
        data = self.client.remote_post(self._url(), self._headers(), payload)
        return _extract_content_from_openai(data, "azure")

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        payload = {
            "messages": _build_messages(compact_input, user_query),
            "max_tokens": 500,
            "temperature": 0.3,
            "stream": True,
        }
        logger.info(f"Azure OpenAI streaming request to {self.deployment}")
        try:
            response = self.client.post_stream(self._url(), self._headers(), payload)
            yield from _stream_sse(response)
        except RemoteDecodeError:
            raise
        except Exception as e:
            logger.error(f"Streaming failed for azure: {e}")
            raise RemoteDecodeError(
                f"azure streaming failed: {e}",
                provider="azure",
            ) from e


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

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        url = "https://api.anthropic.com/v1/messages"
        payload = self._payload()
        payload["messages"] = [{"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
        logger.info(f"Anthropic request to {self.model}")
        data = self.client.remote_post(url, self._headers(), payload)
        return _extract_content_from_anthropic(data)

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = "https://api.anthropic.com/v1/messages"
        payload = self._payload(stream=True)
        payload["messages"] = [{"role": "user", "content": f"[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
        logger.info(f"Anthropic streaming request to {self.model}")
        try:
            response = self.client.post_stream(url, self._headers(), payload)
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
        except RemoteDecodeError:
            raise
        except Exception as e:
            logger.error(f"Streaming failed for anthropic: {e}")
            raise RemoteDecodeError(
                f"anthropic streaming failed: {e}",
                provider="anthropic",
            ) from e


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

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        logger.info(f"Google request to {self.model}")
        data = self.client.remote_post(url, headers, self._content(compact_input, user_query))
        return _extract_content_from_google(data)

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent"
        headers = {"x-goog-api-key": self.api_key}
        logger.info(f"Google streaming request to {self.model}")
        try:
            response = self.client.post_stream(
                url, headers, self._content(compact_input, user_query)
            )
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
        except RemoteDecodeError:
            raise
        except Exception as e:
            logger.error(f"Streaming failed for google: {e}")
            raise RemoteDecodeError(
                f"google streaming failed: {e}",
                provider="google",
            ) from e


class OllamaRemoteDecoder(RemoteDecoder):
    """Ollama local decoder (zero cost)."""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        logger.info(f"Ollama remote request to {self.config.remote_model}")
        response = self.client.ollama_generate(
            model=self.config.remote_model,
            prompt=f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}\n\nAnswer:",
            max_tokens=500,
        )
        # Ollama returns a string from ollama_generate; usage unavailable via this path
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "source": "unavailable"}
        return response, usage

    def decode_stream(self, compact_input: str, user_query: str) -> Iterator[str]:
        logger.info(f"Ollama streaming request to {self.config.remote_model}")
        try:
            response = self.client.post_stream(
                f"{self.config.ollama_base_url}/api/generate",
                {},
                {
                    "model": self.config.remote_model,
                    "prompt": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}\n\nAnswer:",
                    "stream": True,
                    "options": {"temperature": 0.3, "num_predict": 500},
                },
            )
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if data.get("response"):
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
        except RemoteDecodeError:
            raise
        except Exception as e:
            logger.error(f"Streaming failed for ollama: {e}")
            raise RemoteDecodeError(
                f"ollama streaming failed: {e}",
                provider="ollama",
            ) from e


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

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        try:
            import boto3
            bedrock = boto3.client("bedrock-runtime", region_name=self.region)
            body = self._body()
            body["messages"] = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n[SCENE DATA]: {compact_input}\n\n[QUESTION]: {user_query}"}]
            response = bedrock.invoke_model(
                body=json.dumps(body), modelId=self.model_id,
                contentType="application/json", accept="application/json",
            )
            result = json.loads(response["body"].read())
            content_blocks = result.get("content", [])
            if not content_blocks:
                raise RemoteDecodeError(
                    "Bedrock returned no content blocks",
                    provider="bedrock",
                )
            texts = [b.get("text", "") for b in content_blocks if b.get("text")]
            text = " ".join(texts).strip()
            if not text:
                raise RemoteDecodeError(
                    "Bedrock returned empty text content",
                    provider="bedrock",
                )
            # Extract real token usage from Bedrock Anthropic response
            raw_usage = result.get("usage", {})
            usage = {
                "prompt_tokens": raw_usage.get("input_tokens", 0),
                "completion_tokens": raw_usage.get("output_tokens", 0),
                "total_tokens": raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
                "source": "provider" if raw_usage else "unavailable",
            }
            return text, usage
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")
        except RemoteDecodeError:
            raise
        except Exception as e:
            if "boto3" not in str(type(e).__module__):
                raise RemoteDecodeError(
                    f"Bedrock decode failed: {e}",
                    provider="bedrock",
                ) from e
            raise

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
        except RemoteDecodeError:
            raise
        except Exception as e:
            raise RemoteDecodeError(
                f"Bedrock streaming failed: {e}",
                provider="bedrock",
            ) from e


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
