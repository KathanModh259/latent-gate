"""Tests for FastAPI server."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from latent_gate.config import PipelineConfig


@pytest.fixture
def test_client():
    from latent_gate.api_server import app

    app.state.config = PipelineConfig(remote_provider="ollama")
    return app


def test_health_check(test_client):
    with TestClient(test_client) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


@patch("latent_gate.api_server.LatentGatePipeline")
def test_compress_text_endpoint(mock_pipeline_class, test_client):
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.query_text.return_value = {
        "answer": "Compressed text",
        "compact_prompt": "compact text",
        "tokens_estimated": 50,
        "original_tokens": 100,
        "compression_ratio": "2.0x",
        "tokens_saved": 50,
        "timing": {"total_ms": 100},
        "was_cached": False,
        "input_type": "text",
    }

    with TestClient(test_client) as client:
        response = client.post("/query/text", json={"text": "This is a long text to compress"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Compressed text"
        assert data["tokens_saved"] == 50
        mock_pipeline.query_text.assert_called_once_with(
            "This is a long text to compress", "", "auto"
        )


def test_rate_limiter(test_client):
    # Patch RATE_LIMIT_MAX to a low value so we don't need 100 requests
    from latent_gate.api_server import _request_counts

    _request_counts.clear()

    with (
        patch("latent_gate.api_server.RATE_LIMIT_MAX", 3),
        patch("latent_gate.api_server.LatentGatePipeline") as mock_pipeline_class,
    ):
        mock_pipeline = mock_pipeline_class.return_value
        mock_pipeline.query_text.return_value = {
            "answer": "ok",
            "compact_prompt": "ok",
            "tokens_estimated": 10,
            "timing": {},
            "was_cached": False,
            "input_type": "text",
        }

        with TestClient(test_client) as client:
            # Send 3 successful requests (within rate limit)
            for _ in range(3):
                res = client.post("/query/text", json={"text": "text"})
                assert res.status_code == 200

            # 4th request should be rate limited
            res_429 = client.post("/query/text", json={"text": "text"})
            assert res_429.status_code == 429


def test_default_app_registers_all_routes():
    """Regression: app must be built after routers are populated (else every route 404s)."""
    from latent_gate.api_server import app

    paths = {r.path for r in app.routes}
    for expected in ("/health", "/compress", "/query/text", "/v1/chat/completions", "/ws/compress"):
        assert expected in paths


@patch("latent_gate.api_server.LatentGatePipeline")
def test_openai_endpoint_forwards_system_prompt(mock_pipeline_class, test_client):
    mock_pipeline = mock_pipeline_class.return_value
    mock_pipeline.config.remote_provider = "ollama"
    mock_pipeline.compress_prompt.return_value = {
        "compressed_prompt": "short",
        "original_tokens": 10,
        "compressed_tokens": 2,
    }
    mock_pipeline.remote_decoder.decode.return_value = ("hi", {"completion_tokens": 1})

    with TestClient(test_client) as client:
        res = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Answer in French."},
                    {"role": "user", "content": "Hello there"},
                ]
            },
        )
    assert res.status_code == 200
    decoder_input = mock_pipeline.remote_decoder.decode.call_args.args[0]
    assert "Answer in French." in decoder_input


@patch("latent_gate.api_server.LatentGatePipeline")
def test_compress_requires_key_when_configured(mock_pipeline_class, test_client, monkeypatch):
    monkeypatch.setenv("LATENTGATE_API_KEY", "secret")
    with TestClient(test_client) as client:
        assert client.post("/compress", json={"text": "x"}).status_code == 401


@patch("latent_gate.api_server.LatentGatePipeline")
def test_websocket_rejects_missing_key(mock_pipeline_class, test_client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("LATENTGATE_API_KEY", "secret")
    with TestClient(test_client) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/compress") as ws:
                ws.receive_json()


def test_non_utf8_json_body_is_rejected_cleanly(test_client):
    with patch("latent_gate.api_server.LatentGatePipeline"):
        with TestClient(test_client) as client:
            res = client.post(
                "/query/text",
                content=b'{"text": "\xff\xfe"}',
                headers={"content-type": "application/json"},
            )
    assert res.status_code in (400, 422)  # client error, not a 500 crash
