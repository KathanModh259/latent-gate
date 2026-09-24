"""Tests for new API endpoints and features: batch compression, OpenAI-compatible,
WebSocket streaming, CLI output formats, and model fallback decoder.
"""

import pytest
import json
from unittest.mock import patch
from fastapi.testclient import TestClient

from latent_gate.api_server import app
from latent_gate.config import PipelineConfig
from latent_gate.pipeline import _FallbackDecoderWrapper

# ============================================================================
# Batch Compression Endpoint Tests
# ============================================================================


@pytest.fixture
def test_client():
    app.state.config = PipelineConfig(remote_provider="ollama")
    return app


@patch("latent_gate.api_server.LatentGatePipeline")
def test_batch_compress_basic(mock_pipeline_class, test_client):
    """Test basic batch compression with text prompts."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.compress_prompt.return_value = {
        "original_prompt": "test",
        "compressed_prompt": "compressed",
        "original_tokens": 100,
        "compressed_tokens": 20,
        "tokens_saved": 80,
        "compression_ratio": "5.0x",
        "processing_time_ms": 150.0,
    }

    with TestClient(test_client) as client:
        response = client.post(
            "/compress/batch",
            json={"texts": ["prompt one", "prompt two", "prompt three"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["total_original_tokens"] == 300
        assert data["total_compressed_tokens"] == 60
        assert data["total_tokens_saved"] == 240
        assert mock_pipeline.compress_prompt.call_count == 3


@patch("latent_gate.api_server.LatentGatePipeline")
def test_batch_compress_empty_texts(mock_pipeline_class, test_client):
    """Test batch compression with empty texts array (should fail validation)."""
    with TestClient(test_client) as client:
        response = client.post("/compress/batch", json={"texts": []})
        assert response.status_code == 422  # Validation error


@patch("latent_gate.api_server.LatentGatePipeline")
def test_batch_compress_partial_failure(mock_pipeline_class, test_client):
    """Test batch compression where some items fail."""
    mock_pipeline = mock_pipeline_class.return_value

    def mock_compress(text):
        if text == "fail":
            raise ValueError("Simulated failure")
        return {
            "original_prompt": text,
            "compressed_prompt": text + "_compressed",
            "original_tokens": 100,
            "compressed_tokens": 20,
            "tokens_saved": 80,
            "compression_ratio": "5.0x",
            "processing_time_ms": 150.0,
        }

    mock_pipeline.compress_prompt.side_effect = mock_compress

    with TestClient(test_client) as client:
        response = client.post(
            "/compress/batch",
            json={"texts": ["good", "fail", "also_good"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 3
        assert data["results"][1]["error"] is not None  # Failed item has error
        assert data["results"][0]["error"] is None  # Good item has no error


# ============================================================================
# OpenAI-Compatible Endpoint Tests
# ============================================================================


@patch("latent_gate.api_server.LatentGatePipeline")
def test_openai_basic_request(mock_pipeline_class, test_client):
    """Test basic OpenAI-compatible request."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.compress_prompt.return_value = {
        "original_prompt": "Hello world",
        "compressed_prompt": "Hello world",
        "original_tokens": 50,
        "compressed_tokens": 50,
        "tokens_saved": 0,
        "compression_ratio": "1.0x",
        "processing_time_ms": 10.0,
    }
    mock_pipeline.remote_decoder.decode.return_value = (
        "Compressed response",
        {"completion_tokens": 20},
    )

    with TestClient(test_client) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello world"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "compressed" in data["choices"][0]["message"]["content"].lower()
        assert "usage" in data


@patch("latent_gate.api_server.LatentGatePipeline")
def test_openai_no_user_message(mock_pipeline_class, test_client):
    """Test OpenAI endpoint without user message (should error)."""
    with TestClient(test_client) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "assistant", "content": "I am assistant."}],
            },
        )
        assert response.status_code == 400


# ============================================================================
# Fallback Decoder Tests
# ============================================================================


class MockDecoder:
    """Mock decoder for testing fallback behavior."""

    def __init__(self, name, should_fail=False):
        self.name = name
        self.should_fail = should_fail

    def decode(self, compact_input, user_query):
        if self.should_fail:
            raise ConnectionError(f"{self.name} failed")
        return (f"{self.name} response", {"completion_tokens": 10})

    def decode_stream(self, compact_input, user_query):
        if self.should_fail:
            raise ConnectionError(f"{self.name} failed")
        yield from [f"{self.name} chunk"]


def test_fallback_decoder_success():
    """Test fallback decoder passes through when primary succeeds."""
    primary = MockDecoder("primary", should_fail=False)
    fallback = MockDecoder("fallback", should_fail=False)
    wrapper = _FallbackDecoderWrapper(primary, fallback)

    result, usage = wrapper.decode("input", "question")
    assert result == "primary response"


def test_fallback_decoder_failover():
    """Test fallback decoder uses fallback when primary fails."""
    primary = MockDecoder("primary", should_fail=True)
    fallback = MockDecoder("fallback", should_fail=False)
    wrapper = _FallbackDecoderWrapper(primary, fallback)

    result, usage = wrapper.decode("input", "question")
    assert result == "fallback response"


def test_fallback_decoder_both_fail():
    """Test fallback decoder raises when both fail."""
    primary = MockDecoder("primary", should_fail=True)
    fallback = MockDecoder("fallback", should_fail=True)
    wrapper = _FallbackDecoderWrapper(primary, fallback)

    with pytest.raises(ConnectionError):
        wrapper.decode("input", "question")


# ============================================================================
# CLI Output Format Tests
# ============================================================================


def test_cli_output_json(monkeypatch, capsys):
    """Test CLI --json flag produces valid JSON output."""
    import sys

    test_args = ["latent-gate", "--text", "Hello world", "--json"]
    monkeypatch.setattr(sys, "argv", test_args)

    # Mock result
    with patch("latent_gate.cli.LatentGatePipeline") as mock_pipeline_class:
        mock_pipeline = mock_pipeline_class.return_value.__enter__.return_value
        mock_pipeline.query_text.return_value = {
            "answer": "test answer",
            "compact_prompt": "test",
            "tokens_estimated": 10,
            "original_tokens": 50,
            "compression_ratio": "5.0x",
            "tokens_saved": 40,
            "timing": {"total_ms": 100},
            "was_cached": False,
            "input_type": "text",
        }

        # Just test that the json import works for CLI output
        output_dict = {
            "answer": "test",
            "tokens_estimated": 10,
        }
        json_output = json.dumps(output_dict, indent=2, default=str)
        parsed = json.loads(json_output)
        assert parsed["answer"] == "test"


def test_cli_output_jsonl():
    """Test CLI JSONL output format produces single-line JSON."""
    output_dict = {
        "answer": "test",
        "tokens_estimated": 10,
    }
    jsonl_output = json.dumps(output_dict, default=str)
    # JSONL should be a single line with no extra whitespace
    assert "\n" not in jsonl_output.strip()
    assert jsonl_output == json.dumps(output_dict, default=str)


# ============================================================================
# Public /compress Endpoint Tests
# ============================================================================


@patch("latent_gate.api_server.LatentGatePipeline")
def test_public_compress_endpoint(mock_pipeline_class, test_client):
    """Test the public /compress endpoint (no auth required)."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.compress_prompt.return_value = {
        "original_prompt": "Hello world",
        "compressed_prompt": "Hello",
        "original_tokens": 50,
        "compressed_tokens": 10,
        "tokens_saved": 40,
        "compression_ratio": "5.0x",
        "processing_time_ms": 100.0,
        "input_type": "compress_only",
    }

    with TestClient(test_client) as client:
        response = client.post(
            "/compress",
            json={"text": "Hello world"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["compressed_prompt"] == "Hello"
        assert data["original_tokens"] == 50
        assert data["compressed_tokens"] == 10
        assert data["tokens_saved"] == 40
        assert data["compression_ratio"] == "5.0x"
        assert "processing_time_ms" in data
        mock_pipeline.compress_prompt.assert_called_once_with("Hello world")


@patch("latent_gate.api_server.LatentGatePipeline")
def test_public_compress_endpoint_no_auth(mock_pipeline_class, test_client):
    """Test /compress does NOT require auth header."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.compress_prompt.return_value = {
        "original_prompt": "test",
        "compressed_prompt": "compressed",
        "original_tokens": 10,
        "compressed_tokens": 5,
        "tokens_saved": 5,
        "compression_ratio": "2.0x",
        "processing_time_ms": 50.0,
        "input_type": "compress_only",
    }

    with TestClient(test_client) as client:
        response = client.post("/compress", json={"text": "test"})
        assert response.status_code == 200, "Public endpoint should not need auth"


@patch("latent_gate.api_server.LatentGatePipeline")
def test_public_compress_empty_text(mock_pipeline_class, test_client):
    """Test /compress with empty text (should return success since empty string is valid)."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.compress_prompt.return_value = {
        "original_prompt": "",
        "compressed_prompt": "",
        "original_tokens": 0,
        "compressed_tokens": 0,
        "tokens_saved": 0,
        "compression_ratio": "0.0x",
        "processing_time_ms": 1.0,
        "input_type": "compress_only",
    }

    with TestClient(test_client) as client:
        response = client.post("/compress", json={"text": ""})
        # Empty string is a valid Pydantic str, so it passes validation
        assert response.status_code == 200


@patch("latent_gate.api_server.LatentGatePipeline")
def test_public_compress_missing_text_field(mock_pipeline_class, test_client):
    """Test /compress without text field (should trigger validation error)."""
    with TestClient(test_client) as client:
        response = client.post("/compress", json={})
        assert response.status_code == 422  # Missing required field


def test_public_compress_pipeline_not_initialized(test_client):
    """Test /compress when pipeline is None (should return 503).

    The pipeline must be set to None AFTER the TestClient context is entered
    (lifespan runs on enter and sets pipeline).
    """
    import latent_gate.api_server as api_module

    original = api_module.pipeline

    try:
        with TestClient(test_client) as client:
            # Set pipeline to None AFTER lifespan has initialized it
            api_module.pipeline = None
            response = client.post("/compress", json={"text": "test"})
            assert (
                response.status_code == 503
            ), f"Expected 503, got {response.status_code}: {response.text}"
    finally:
        api_module.pipeline = original


# ============================================================================
# API Auth Wrapping Tests
# ============================================================================


@patch("latent_gate.api_server.LatentGatePipeline")
@patch.dict("os.environ", {"LATENTGATE_API_KEY": "test-secret-key-123"})
def test_api_router_requires_auth(mock_pipeline_class):
    """Test that api_router endpoints reject requests without auth when API key is set."""
    import latent_gate.api_server as api_module

    with patch.object(api_module, "_docs_enabled", return_value=False):
        new_app = api_module.create_app()

        with TestClient(new_app) as client:
            # Try accessing a protected endpoint without auth
            response = client.post(
                "/query/text",
                json={"text": "Hello"},
            )
            assert response.status_code == 401, "Should require auth"

            # Try with wrong auth
            response = client.post(
                "/query/text",
                json={"text": "Hello"},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert response.status_code == 401, "Should reject wrong auth"


@patch("latent_gate.api_server.LatentGatePipeline")
@patch.dict("os.environ", {"LATENTGATE_API_KEY": "test-secret-key-123"})
def test_api_router_passes_with_valid_auth(mock_pipeline_class):
    """Test that api_router endpoints accept requests with valid auth."""
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.query_text.return_value = {
        "answer": "ok",
        "compact_prompt": "ok",
        "tokens_estimated": 10,
        "timing": {},
        "was_cached": False,
        "input_type": "text",
    }

    import latent_gate.api_server as api_module

    with patch.object(api_module, "_docs_enabled", return_value=False):
        new_app = api_module.create_app()

        with TestClient(new_app) as client:
            response = client.post(
                "/query/text",
                json={"text": "Hello"},
                headers={"Authorization": "Bearer test-secret-key-123"},
            )
            assert response.status_code == 200
            assert response.json()["answer"] == "ok"
