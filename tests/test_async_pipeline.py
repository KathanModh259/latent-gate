"""Tests for AsyncLatentGatePipeline."""

import pytest
from unittest.mock import patch, MagicMock

from latent_gate.async_pipeline import AsyncLatentGatePipeline
from latent_gate.config import PipelineConfig

@pytest.fixture
def async_pipeline():
    config = PipelineConfig(remote_provider="ollama")
    return AsyncLatentGatePipeline(config, preload=False)

@pytest.mark.asyncio
@patch("latent_gate.async_pipeline.LatentGatePipeline.query")
async def test_query_async(mock_query, async_pipeline):
    mock_query.return_value = {"answer": "async answer"}
    
    async with async_pipeline as pipeline:
        result = await pipeline.query("image.jpg", "question")
        
        assert result["answer"] == "async answer"
        mock_query.assert_called_once_with("image.jpg", "question")

@pytest.mark.asyncio
@patch("latent_gate.async_pipeline.LatentGatePipeline.query_text")
async def test_query_text_async(mock_query_text, async_pipeline):
    mock_query_text.return_value = {"answer": "text async answer"}
    
    async with async_pipeline as pipeline:
        result = await pipeline.query_text("some text", question="q", mode="auto")
        
        assert result["answer"] == "text async answer"
        mock_query_text.assert_called_once_with("some text", "q", "auto")

@pytest.mark.asyncio
@patch("latent_gate.async_pipeline.LatentGatePipeline.query_batch")
async def test_query_batch_async(mock_query_batch, async_pipeline):
    mock_query_batch.return_value = [{"answer": "1"}]
    
    async with async_pipeline as pipeline:
        result = await pipeline.query_batch(["img.jpg"], "Describe this image", True)
        
        assert len(result) == 1
        assert result[0]["answer"] == "1"
        mock_query_batch.assert_called_once()
