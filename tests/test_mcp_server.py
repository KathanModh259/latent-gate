"""Tests for MCP server."""

import pytest
from unittest.mock import patch, MagicMock
import json

from latent_gate.mcp_server import call_tool, list_tools


@pytest.mark.asyncio
async def test_list_tools():
    tools = await list_tools()
    assert len(tools) == 8
    names = [t.name for t in tools]
    for offline in ("read_file_optimized", "optimize_text", "count_tokens"):
        assert offline in names
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
        "payload": {"scene_type": "outdoor"},
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


@pytest.mark.asyncio
@patch("latent_gate.mcp_server.get_pipeline")
async def test_offline_tools_never_build_pipeline(mock_get_pipeline, tmp_path):
    """read_file_optimized / optimize_text must work without Ollama or the ML stack."""
    log = tmp_path / "app.log"
    log.write_text(
        "\n".join(f"2026-09-24 12:00:{i:02d} ERROR worker id={i} timeout" for i in range(50)),
        encoding="utf-8",
    )
    data = json.loads((await call_tool("read_file_optimized", {"path": str(log)}))[0].text)
    assert data["optimized_tokens"] < data["original_tokens"] / 4
    assert "similar lines" in data["optimized"]

    data = json.loads((await call_tool("optimize_text", {"text": "Hi!\n\nFix it."}))[0].text)
    assert data["optimized"] == "Fix it."
    mock_get_pipeline.assert_not_called()


@pytest.mark.asyncio
async def test_read_file_optimized_rejects_binary(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(bytes([0, 1, 2]) + b"binary")
    out = (await call_tool("read_file_optimized", {"path": str(f)}))[0].text
    assert "binary" in out


def test_mcp_module_import_is_fast():
    """Regression: importing the ML stack at module load made startup exceed Claude's 30s MCP timeout."""
    import subprocess
    import sys
    import time

    start = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import latent_gate.mcp_server"], check=True)
    assert time.perf_counter() - start < 15
