"""Tests for RemoteDecoder factory and formatting."""

import pytest
from latent_gate.config import PipelineConfig
from latent_gate.remote_decoder import (
    create_decoder,
    OpenAIDecoder,
    AnthropicDecoder,
    GoogleDecoder,
    OllamaRemoteDecoder,
)


class TestDecoderFactory:

    def test_create_openai(self):
        config = PipelineConfig(remote_provider="openai")
        decoder = create_decoder(config)
        assert isinstance(decoder, OpenAIDecoder)

    def test_create_anthropic(self):
        config = PipelineConfig(remote_provider="anthropic")
        decoder = create_decoder(config)
        assert isinstance(decoder, AnthropicDecoder)

    def test_create_google(self):
        config = PipelineConfig(remote_provider="google")
        decoder = create_decoder(config)
        assert isinstance(decoder, GoogleDecoder)

    def test_create_ollama(self):
        config = PipelineConfig(remote_provider="ollama")
        decoder = create_decoder(config)
        assert isinstance(decoder, OllamaRemoteDecoder)

    def test_create_unknown_defaults_to_openai(self):
        config = PipelineConfig(remote_provider="unknown_provider")
        decoder = create_decoder(config)
        assert isinstance(decoder, OpenAIDecoder)

    def test_custom_base_url(self):
        config = PipelineConfig(
            remote_provider="openai",
            remote_base_url="https://custom.endpoint.com/v1",
        )
        decoder = create_decoder(config)
        assert isinstance(decoder, OpenAIDecoder)
        assert decoder.base_url == "https://custom.endpoint.com/v1"
