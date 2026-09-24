"""Regression tests for bugs fixed before the public release (no Ollama needed)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from latent_gate.config import DEFAULT_REMOTE_MODELS, PipelineConfig
from latent_gate.fast_client import FastClient
from latent_gate.pipeline import LatentGatePipeline


@pytest.fixture
def pipeline():
    pl = LatentGatePipeline(
        PipelineConfig(remote_provider="ollama", log_level="WARNING", enable_caching=False),
        preload=False,
    )
    yield pl
    pl.close()


# --- Provider-aware default model -------------------------------------------


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "groq", "ollama"])
def test_remote_model_defaults_follow_provider(provider):
    assert PipelineConfig(remote_provider=provider).remote_model == DEFAULT_REMOTE_MODELS[provider]


def test_env_provider_gets_its_own_default_model(monkeypatch):
    from latent_gate.config_loader import get_config

    monkeypatch.setenv("LATENTGATE_REMOTE_PROVIDER", "anthropic")
    monkeypatch.delenv("LATENTGATE_REMOTE_MODEL", raising=False)
    monkeypatch.chdir(pytest.importorskip("tempfile").mkdtemp())  # no config file present
    assert get_config().remote_model == DEFAULT_REMOTE_MODELS["anthropic"]


def test_offline_decoder_uses_offline_model():
    pl = LatentGatePipeline(
        PipelineConfig(
            remote_provider="openai",
            offline_first=True,
            offline_model="mistral:7b",
            log_level="WARNING",
            enable_caching=False,
        ),
        preload=False,
    )
    try:
        assert pl._offline_decoder.config.remote_model == "mistral:7b"
    finally:
        pl.close()


# --- Ollama errors engage fallback chains -----------------------------------


def test_missing_ollama_model_maps_to_connection_error():
    client = FastClient(PipelineConfig(remote_provider="ollama"))
    resp = MagicMock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404 model not found")
    with patch.object(client._ollama_session, "post", return_value=resp):
        with pytest.raises(ConnectionError, match="ollama pull phi3:mini"):
            client.ollama_generate(model="phi3:mini", prompt="hi")


def test_preload_skips_quickly_when_ollama_down():
    client = FastClient(PipelineConfig(remote_provider="ollama"))
    with (
        patch.object(client, "list_local_models", return_value=None),
        patch.object(client._ollama_session, "post") as post,
    ):
        assert client.preload_models() == []
        post.assert_not_called()


# --- Dedup cache correctness ------------------------------------------------


def test_dedup_distinguishes_inputs_with_shared_prefix(pipeline):
    header = "x" * 600
    a = pipeline._dedup_key("text/auto", header + " ending A", False)
    b = pipeline._dedup_key("text/auto", header + " ending B", False)
    pipeline._cache_result(a, "q", {"answer": "A"})
    assert pipeline._is_duplicate(b, "q") is None


def test_dedup_separates_compress_only_from_full_query(pipeline):
    pipeline._cache_result(pipeline._dedup_key("text/auto", "t", True), "q", {"answer": ""})
    assert pipeline._is_duplicate(pipeline._dedup_key("text/auto", "t", False), "q") is None


def test_transient_decode_failure_is_not_cached(pipeline):
    key = pipeline._dedup_key("text/auto", "t", False)
    pipeline._cache_result(key, "q", {"answer": "[Remote decode skipped: offline]"})
    assert pipeline._is_duplicate(key, "q") is None


def test_image_cache_keyed_by_content(pipeline, tmp_path):
    img = tmp_path / "frame.png"
    img.write_bytes(b"frame-1")
    first = pipeline._image_fingerprint(str(img))
    img.write_bytes(b"frame-2")  # same path, new content (e.g. video frame dump)
    assert pipeline._image_fingerprint(str(img)) != first


def test_selective_decoding_redecodes_for_new_question(pipeline, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"img")
    payload = MagicMock(estimated_token_count=10)
    payload.to_compact_prompt.return_value = "scene"
    payload.to_dict.return_value = {}
    decoder = MagicMock()
    decoder.decode.side_effect = [("red", None), ("three", None)]
    with (
        patch.object(pipeline.local_processor, "process", return_value=payload),
        patch.object(pipeline, "_get_decoder", return_value=decoder),
        patch.object(pipeline.selective_decoder, "should_decode", return_value=False),
    ):
        pipeline._selective_last_question = None
        assert pipeline.query(str(img), "What color?")["answer"] == "red"
        pipeline._query_cache.clear()
        assert pipeline.query(str(img), "How many people?")["answer"] == "three"


@pytest.mark.parametrize(
    "model,installed,expected",
    [
        ("llava:7b", ["llava:7b"], True),
        ("nomic-embed-text", ["nomic-embed-text:latest"], True),
        ("llama3:latest", ["llama3"], True),
        ("llava:7b", ["llava:13b"], False),
    ],
)
def test_is_model_installed_handles_implicit_latest_tag(model, installed, expected):
    from latent_gate.fast_client import is_model_installed

    assert is_model_installed(model, installed) is expected


def test_fallback_chain_stops_when_ollama_server_down():
    from latent_gate.fast_client import OllamaUnavailableError
    from latent_gate.text_processor import TextProcessor

    tp = TextProcessor(PipelineConfig(remote_provider="ollama"))
    tp.client = MagicMock()
    tp.client.ollama_generate.side_effect = OllamaUnavailableError("down")
    with pytest.raises(OllamaUnavailableError):
        tp._ollama_generate_with_fallback("p", task="text_fast")
    assert tp.client.ollama_generate.call_count == 1  # didn't walk the whole chain


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "groq", "ollama"])
def test_output_token_cap_is_configurable(provider):
    """Regression: every provider hard-coded max_tokens=500, truncating long answers."""
    from latent_gate.remote_decoder import create_decoder

    cfg = PipelineConfig(remote_provider=provider, remote_api_key="k", max_output_tokens=2048)
    client = MagicMock()
    client.remote_post.return_value = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "content": [{"type": "text", "text": "ok"}],
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
    }
    client.ollama_generate.return_value = "ok"
    create_decoder(cfg, client=client).decode("ctx", "q")
    if provider == "ollama":
        sent = client.ollama_generate.call_args.kwargs["max_tokens"]
    else:
        body = client.remote_post.call_args.args[2]
        sent = body.get("max_tokens") or body["generationConfig"]["maxOutputTokens"]
    assert sent == 2048


def test_default_output_cap_is_not_500():
    assert PipelineConfig().max_output_tokens >= 4096
