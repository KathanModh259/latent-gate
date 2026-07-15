"""Tests for MCP server."""

import pytest
from unittest.mock import patch, MagicMock
import json

from latent_gate.mcp_server import call_tool, list_tools

@pytest.mark.asyncio
async def test_list_tools():
    tools = await list_tools()
    assert len(tools) == 5
    names = [t.name for t in tools]
    assert "compress_image" in names
    assert "compress_text" in names
    assert "compress_conversation" in names
    assert "compress_documents" in names
    assert "get_stats" in names

@pytest.mark.asyncio
@patch("latent_gate.mcp_server.get_pipeline")
async def test_call_tool_compress_image(mock_get_pipeline):
    mock_pipeline = MagicMock()
    mock_get_pipeline.return_value = mock_pipeline
    
    mock_pipeline.query.return_value = {
        "compact_prompt": "scene data",
        "tokens_estimated": 150,
        "original_tokens": 1200,
        "answer": "The answer",
        "payload": {"scene_type": "outdoor"}
    }
    
    result = await call_tool("compress_image", {"image_path": "test.jpg"})
    assert len(result) == 1
    
    data = json.loads(result[0].text)
    assert data["answer"] == "The answer"
    assert data["tokens_saved"] == 1050  # 1200 - 150
    assert data["tokens_estimated"] == 150

@pytest.mark.asyncio
@patch("latent_gate.mcp_server.get_pipeline")
async def test_call_tool_unknown(mock_get_pipeline):
    result = await call_tool("unknown_tool", {})
    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown tool" in data["error"]
