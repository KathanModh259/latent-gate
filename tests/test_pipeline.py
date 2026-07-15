"""Tests for LatentGatePipeline (integration-level, mocked)."""

import pytest
from unittest.mock import patch, MagicMock

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline
from latent_gate.payload import SemanticPayload


class TestPipelineInit:

    def test_default_init(self):
        """Pipeline should initialize without errors."""
        pipeline = LatentGatePipeline(
            PipelineConfig(
                remote_provider="ollama",
                log_level="WARNING",
                enable_caching=False,
            )
        )
        assert pipeline.config.remote_provider == "ollama"
        assert pipeline.local_processor is not None
        assert pipeline.selective_decoder is not None
        assert pipeline.remote_decoder is not None

    def test_reset_selective_decoder(self):
        pipeline = LatentGatePipeline(
            PipelineConfig(remote_provider="ollama", log_level="WARNING")
        )
        # Manually set some state
        pipeline.selective_decoder.call_count = 5
        pipeline.selective_decoder.skip_count = 3

        pipeline.reset_selective_decoder()

        assert pipeline.selective_decoder.call_count == 0
        assert pipeline.selective_decoder.skip_count == 0


class TestPipelineQuery:

    @patch("latent_gate.local_processor.LocalProcessor.process")
    @patch("latent_gate.remote_decoder.OllamaRemoteDecoder.decode")
    def test_query_flow(self, mock_decode, mock_process):
        """Test the full query flow with mocked components."""
        # Mock local processing
        mock_payload = SemanticPayload(
            scene_type="indoor",
            scene_description="A room with furniture",
            objects_detected=["table", "chair"],
        )
        mock_process.return_value = mock_payload

        # Mock remote decoding
        mock_decode.return_value = ("This is a room with a table and chair.", {"completion_tokens": 10})

        # Run pipeline
        pipeline = LatentGatePipeline(
            PipelineConfig(
                remote_provider="ollama",
                selective_decoding=False,
                enable_caching=False,
                log_level="WARNING",
            )
        )
        result = pipeline.query("fake_image.jpg", "What is in this image?")

        assert result["answer"] == "This is a room with a table and chair."
        assert result["was_cached"] is False
        assert "payload" in result
        assert "compact_prompt" in result
        mock_process.assert_called_once_with("fake_image.jpg")
        mock_decode.assert_called_once()
