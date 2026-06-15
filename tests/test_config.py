"""Tests for PipelineConfig."""

import os
import pytest
from latent_gate.config import PipelineConfig


class TestPipelineConfig:

    def test_default_values(self):
        config = PipelineConfig()
        assert config.vision_model == "llava:7b"
        assert config.predictor_model == "llama3:8b"
        assert config.remote_provider == "openai"
        assert config.remote_model == "gpt-4o-mini"
        assert config.enable_caching is True
        assert config.selective_decoding is True
        assert config.similarity_threshold == 0.85
        assert config.temperature == 0.1

    def test_custom_values(self):
        config = PipelineConfig(
            vision_model="bakllava",
            remote_provider="anthropic",
            remote_model="claude-sonnet-4-20250514",
            similarity_threshold=0.9,
        )
        assert config.vision_model == "bakllava"
        assert config.remote_provider == "anthropic"
        assert config.similarity_threshold == 0.9

    def test_api_key_from_env(self):
        os.environ["OPENAI_API_KEY"] = "test-key-123"
        config = PipelineConfig(remote_provider="openai")
        assert config.remote_api_key == "test-key-123"
        del os.environ["OPENAI_API_KEY"]

    def test_validate_missing_api_key(self):
        config = PipelineConfig(remote_provider="openai", remote_api_key="")
        # Clear env var if set
        os.environ.pop("OPENAI_API_KEY", None)
        config = PipelineConfig(remote_provider="openai")
        warnings = config.validate()
        assert len(warnings) >= 1
        assert "api key" in warnings[0].lower() or "API key" in warnings[0]

    def test_validate_ollama_no_key_needed(self):
        config = PipelineConfig(remote_provider="ollama")
        warnings = config.validate()
        # Ollama doesn't need API key
        key_warnings = [w for w in warnings if "api key" in w.lower() or "API key" in w]
        assert len(key_warnings) == 0

    def test_validate_bad_threshold(self):
        config = PipelineConfig(similarity_threshold=1.5)
        warnings = config.validate()
        assert any("threshold" in w.lower() for w in warnings)
