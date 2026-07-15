"""Tests for FastAPI server."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from latent_gate.api_server import create_app
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
        "input_type": "text"
    }
    
    with TestClient(test_client) as client:
        response = client.post(
            "/query/text",
            json={"text": "This is a long text to compress"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Compressed text"
        assert data["tokens_saved"] == 50
        mock_pipeline.query_text.assert_called_once_with("This is a long text to compress", "", "auto")

def test_rate_limiter(test_client):
    # We will simulate the rate limit check directly to avoid patching pipeline internals for 100 requests.
    from latent_gate.api_server import _request_counts, RATE_LIMIT_MAX
    _request_counts.clear()
    
    with patch("latent_gate.api_server.LatentGatePipeline") as mock_pipeline_class:
        mock_pipeline = mock_pipeline_class.return_value
        mock_pipeline.query_text.return_value = {
            "answer": "ok", "compact_prompt": "ok", "tokens_estimated": 10, "timing": {},
            "was_cached": False, "input_type": "text"
        }
        
        with TestClient(test_client) as client:
            # Send requests up to RATE_LIMIT_MAX
            for _ in range(RATE_LIMIT_MAX):
                res = client.post("/query/text", json={"text": "text"})
                assert res.status_code == 200
                
            res_429 = client.post("/query/text", json={"text": "text"})
            assert res_429.status_code == 429
