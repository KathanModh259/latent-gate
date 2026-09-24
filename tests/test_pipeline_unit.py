"""Unit tests for LatentGatePipeline internal methods.

Tests cover internal methods that don't require Ollama:
  - _FallbackDecoderWrapper
  - Semantic deduplication (_query_hash, _is_duplicate, _cache_result)
  - Complexity estimation (_estimate_complexity)
  - Adaptive max tokens (_adaptive_max_tokens)
  - Result builder (_build_result)
  - _reset_if_mode_changed
  - _get_decoder
  - Mode reset logic
"""

import pytest
from unittest.mock import patch, MagicMock

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline, _FallbackDecoderWrapper

# ============================================================================
# _FallbackDecoderWrapper Tests
# ============================================================================


class TestFallbackDecoderWrapper:
    """Tests for the fallback decoder wrapper."""

    def test_primary_success(self):
        """When primary succeeds, fallback is not called."""
        primary = MagicMock()
        primary.decode.return_value = ("primary answer", {"usage": 10})
        fallback = MagicMock()

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        result, usage = wrapper.decode("input", "question")

        assert result == "primary answer"
        primary.decode.assert_called_once_with("input", "question")
        fallback.decode.assert_not_called()

    def test_fallback_on_connection_error(self):
        """When primary raises ConnectionError, fallback is used."""
        primary = MagicMock()
        primary.decode.side_effect = ConnectionError("connection refused")
        fallback = MagicMock()
        fallback.decode.return_value = ("fallback answer", {"usage": 5})

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        result, usage = wrapper.decode("input", "question")

        assert result == "fallback answer"
        primary.decode.assert_called_once()
        fallback.decode.assert_called_once_with("input", "question")

    def test_fallback_on_timeout(self):
        """When primary raises TimeoutError, fallback is used."""
        primary = MagicMock()
        primary.decode.side_effect = TimeoutError("timed out")
        fallback = MagicMock()
        fallback.decode.return_value = ("timeout handling", {})

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        result, usage = wrapper.decode("input", "q")

        assert result == "timeout handling"

    def test_both_fail_raises(self):
        """When both primary and fallback fail, the exception propagates."""
        primary = MagicMock()
        primary.decode.side_effect = ConnectionError("primary fail")
        fallback = MagicMock()
        fallback.decode.side_effect = ConnectionError("fallback fail")

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        with pytest.raises(ConnectionError, match="fallback fail"):
            wrapper.decode("input", "question")

    def test_no_fallback_raises(self):
        """When there's no fallback, primary failure propagates."""
        primary = MagicMock()
        primary.decode.side_effect = ConnectionError("primary fail")

        wrapper = _FallbackDecoderWrapper(primary, fallback=None)
        with pytest.raises(ConnectionError, match="primary fail"):
            wrapper.decode("input", "question")

    def test_stream_primary_success(self):
        """Streaming: primary provides tokens."""
        primary = MagicMock()
        primary.decode_stream.return_value = iter(["token1", "token2"])
        fallback = MagicMock()

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        tokens = list(wrapper.decode_stream("input", "q"))

        assert tokens == ["token1", "token2"]
        primary.decode_stream.assert_called_once()
        fallback.decode_stream.assert_not_called()

    def test_stream_fallback_on_error(self):
        """Streaming: fallback provides tokens when primary fails."""
        primary = MagicMock()
        primary.decode_stream.side_effect = ConnectionError("stream fail")
        fallback = MagicMock()
        fallback.decode_stream.return_value = iter(["fallback_token"])

        wrapper = _FallbackDecoderWrapper(primary, fallback)
        tokens = list(wrapper.decode_stream("input", "q"))

        assert tokens == ["fallback_token"]
        fallback.decode_stream.assert_called_once_with("input", "q")


# ============================================================================
# Pipeline Internal Methods
# ============================================================================


@pytest.fixture
def pipeline():
    """Create a minimal pipeline for testing internal methods."""
    config = PipelineConfig(
        remote_provider="ollama",
        log_level="WARNING",
        enable_caching=False,
        adaptive_compression=True,
        selective_decoding=True,
    )
    pl = LatentGatePipeline(config, preload=False)
    yield pl
    pl.close()


class TestComplexityEstimation:
    """Tests for _estimate_complexity."""

    def test_simple_text_low_complexity(self, pipeline):
        """Short simple text should have low complexity."""
        text = "What is the capital of France?"
        complexity = pipeline._estimate_complexity(text)
        assert 0.0 <= complexity <= 0.6  # Simple question

    def test_long_text_higher_complexity(self, pipeline):
        """Long text should increase complexity."""
        short = "Hello world."
        long_text = " ".join(["word"] * 1000) + "."

        short_complexity = pipeline._estimate_complexity(short)
        long_complexity = pipeline._estimate_complexity(long_text)

        assert long_complexity > short_complexity

    def test_code_text_higher_complexity(self, pipeline):
        """Code-heavy text should add to complexity."""
        normal = "This is a normal question about the weather."
        code = "Please explain this code:\n```\ndef foo():\n    return 42\n```"

        normal_c = pipeline._estimate_complexity(normal)
        code_c = pipeline._estimate_complexity(code)

        assert code_c >= normal_c

    def test_analytical_question_higher(self, pipeline):
        """Analytical questions (how, why, explain) should increase complexity."""
        simple = "What is 2+2?"
        analytical = "How would you design a distributed system and evaluate its performance?"

        assert pipeline._estimate_complexity(analytical) > pipeline._estimate_complexity(simple)

    def test_image_input_type(self, pipeline):
        """Image input type should return a fixed complexity."""
        assert pipeline._estimate_complexity("", input_type="image") == 0.6

    def test_complexity_bounded(self, pipeline):
        """Complexity should never exceed 1.0."""
        very_long = " ".join(["word"] * 5000) + ". Explain " + "how " * 100
        c = pipeline._estimate_complexity(very_long)
        assert c <= 1.0
        assert c >= 0.0

    def test_empty_text(self, pipeline):
        """Empty text should have minimal complexity."""
        c = pipeline._estimate_complexity("")
        assert c >= 0.0


class TestAdaptiveMaxTokens:
    """Tests for _adaptive_max_tokens."""

    def test_low_complexity(self, pipeline):
        """Low complexity should return fewer tokens."""
        assert pipeline._adaptive_max_tokens(0.0) == 200

    def test_high_complexity(self, pipeline):
        """High complexity should return more tokens."""
        assert pipeline._adaptive_max_tokens(1.0) == 500

    def test_medium_complexity(self, pipeline):
        """Medium complexity should return intermediate tokens."""
        assert pipeline._adaptive_max_tokens(0.5) == 350


class TestModeReset:
    """Tests for _reset_if_mode_changed."""

    def test_first_mode_no_reset(self, pipeline):
        """First mode set should not reset (no previous state to reset from)."""
        pipeline.selective_decoder.call_count = 5
        pipeline._reset_if_mode_changed("image")
        assert pipeline.selective_decoder.call_count == 5  # Not reset
        assert pipeline._last_input_type == "image"

    def test_same_mode_no_reset(self, pipeline):
        """Same mode should not reset the decoder."""
        pipeline._last_input_type = "image"
        pipeline.selective_decoder.call_count = 5
        pipeline._reset_if_mode_changed("image")
        assert pipeline.selective_decoder.call_count == 5  # Not reset

    def test_different_mode_resets(self, pipeline):
        """Different mode should reset the decoder."""
        pipeline._last_input_type = "image"
        pipeline.selective_decoder.call_count = 5
        pipeline._reset_if_mode_changed("text")
        assert pipeline.selective_decoder.call_count == 0  # Reset!
        assert pipeline._last_input_type == "text"

    def test_three_mode_changes(self, pipeline):
        """Multiple mode changes should each reset appropriately."""
        pipeline._reset_if_mode_changed("image")
        pipeline.selective_decoder.call_count = 3

        pipeline._reset_if_mode_changed("image")  # Same - no reset
        assert pipeline.selective_decoder.call_count == 3

        pipeline._reset_if_mode_changed("text")  # Different - reset
        assert pipeline.selective_decoder.call_count == 0


class TestDeduplication:
    """Tests for semantic deduplication (internal cache)."""

    def test_initial_cache_empty(self, pipeline):
        """Cache should start empty."""
        assert len(pipeline._query_cache) == 0
        assert pipeline._dedup_hits == 0

    def test_cache_result_then_duplicate(self, pipeline):
        """After caching a result, same query should be duplicate."""
        result = {"answer": "test", "tokens_estimated": 10}
        pipeline._cache_result("hello world", "", result)

        cached = pipeline._is_duplicate("hello world", "")
        assert cached is not None
        assert cached["answer"] == "test"

    def test_cache_different_queries_not_dup(self, pipeline):
        """Different queries should not be duplicates."""
        result1 = {"answer": "first"}
        result2 = {"answer": "second"}

        pipeline._cache_result("query one", "q1", result1)
        pipeline._cache_result("query two", "q2", result2)

        assert pipeline._is_duplicate("query three", "q3") is None

    def test_cache_hit_tracking(self, pipeline):
        """Cache hits should increment dedup_hits."""
        result = {"answer": "test"}
        pipeline._cache_result("text", "q", result)

        assert pipeline._dedup_hits == 0
        pipeline._is_duplicate("text", "q")
        assert pipeline._dedup_hits == 1
        pipeline._is_duplicate("text", "q")
        assert pipeline._dedup_hits == 2

    def test_query_hash_consistent(self, pipeline):
        """Same input should produce the same hash."""
        h1 = pipeline._query_hash("hello world", "question")
        h2 = pipeline._query_hash("hello world", "question")
        assert h1 == h2

    def test_query_hash_different(self, pipeline):
        """Different inputs should produce different hashes."""
        h1 = pipeline._query_hash("hello", "q")
        h2 = pipeline._query_hash("world", "q")
        assert h1 != h2

    def test_cache_bounded(self, pipeline):
        """Cache should not exceed MAX_DEDUP_CACHE_SIZE."""
        from latent_gate.pipeline import _MAX_DEDUP_CACHE_SIZE

        for i in range(_MAX_DEDUP_CACHE_SIZE + 50):
            pipeline._cache_result(f"text_{i}", "", {"answer": str(i)})

        assert len(pipeline._query_cache) <= _MAX_DEDUP_CACHE_SIZE

    def test_cache_lru_eviction(self, pipeline):
        """Least recently used entries should be evicted first."""
        from latent_gate.pipeline import _MAX_DEDUP_CACHE_SIZE

        # Fill the cache
        for i in range(_MAX_DEDUP_CACHE_SIZE):
            pipeline._cache_result(f"text_{i}", "", {"answer": str(i)})

        # Re-cache entry 0 to move it to the end (most recently used)
        pipeline._cache_result("text_0", "", {"answer": str(0)})

        # Add one more to trigger eviction (should evict text_1, not text_0)
        pipeline._cache_result("new_entry", "", {"answer": "new"})

        # text_0 should still be in cache (was most recently accessed)
        assert pipeline._is_duplicate("text_0", "") is not None
        # text_1 should be evicted (was least recently accessed after text_0 was re-cached)
        assert pipeline._is_duplicate("text_1", "") is None

    def test_clear_dedup_cache(self, pipeline):
        """Clear should reset both cache and hits."""
        pipeline._cache_result("text", "", {"answer": "test"})
        pipeline._is_duplicate("text", "")

        assert pipeline._dedup_hits == 1
        pipeline.clear_dedup_cache()

        assert len(pipeline._query_cache) == 0
        assert pipeline._dedup_hits == 0

    def test_get_dedup_stats(self, pipeline):
        """Dedup stats should reflect current state."""
        pipeline._cache_result("text", "", {"answer": "test"})
        pipeline._is_duplicate("text", "")
        pipeline._cache_result("text2", "", {"answer": "test2"})

        stats = pipeline.get_dedup_stats()
        assert stats["cached_results"] == 2
        assert stats["dedup_hits"] == 1
        assert stats["total_queries"] >= 2


class TestResultBuilder:
    """Tests for _build_result internal method."""

    def test_basic_result_structure(self, pipeline):
        """Result should contain all expected fields."""
        pipeline.selective_decoder.call_count = 0
        result = pipeline._build_result(
            answer="test answer",
            compact_prompt="compact",
            tokens_estimated=50,
            was_cached=False,
            payload_dict={"scene_type": "indoor"},
            input_type="image",
            original_tokens=100,
            compression_ratio=2.0,
            local_time_ms=100.0,
            remote_time_ms=200.0,
        )

        assert result["answer"] == "test answer"
        assert result["compact_prompt"] == "compact"
        assert result["tokens_estimated"] == 50
        assert result["was_cached"] is False
        assert result["input_type"] == "image"
        assert result["original_tokens"] == 100
        assert result["compression_ratio"] == "2.0x"
        assert "timing" in result
        assert result["timing"]["local_ms"] == 100.0
        assert result["timing"]["remote_ms"] == 200.0
        assert result["timing"]["total_ms"] == 300.0

    def test_cached_result_no_original_tokens(self, pipeline):
        """Cached results without original_tokens should omit ratio/saved."""
        pipeline.selective_decoder.call_count = 0
        result = pipeline._build_result(
            answer="cached answer",
            compact_prompt="short",
            tokens_estimated=30,
            was_cached=True,
            payload_dict={},
            input_type="text",
        )

        assert result["answer"] == "cached answer"
        assert result["was_cached"] is True
        assert "original_tokens" not in result  # Not set if 0
        assert "compression_ratio" not in result

    def test_low_confidence_warning(self, pipeline):
        """Low extraction confidence should be noted in the result."""
        pipeline.selective_decoder.call_count = 0
        result = pipeline._build_result(
            answer="ans",
            compact_prompt="cp",
            tokens_estimated=10,
            was_cached=False,
            payload_dict={"confidence": 0.45, "scene_type": "outdoor"},
            input_type="image",
        )

        assert result["extraction_confidence"] == 0.45

    def test_result_contains_dedup_stats(self, pipeline):
        """Result should contain dedup statistics."""
        pipeline.selective_decoder.call_count = 0
        result = pipeline._build_result(
            answer="ans",
            compact_prompt="cp",
            tokens_estimated=10,
            was_cached=False,
            payload_dict={},
            input_type="text",
        )

        assert "dedup_stats" in result
        assert result["dedup_stats"]["total_queries"] >= 0

    def test_result_contains_offline_first_flag(self, pipeline):
        """Result should contain offline_first flag."""
        pipeline.selective_decoder.call_count = 0
        result = pipeline._build_result(
            answer="ans",
            compact_prompt="cp",
            tokens_estimated=10,
            was_cached=False,
            payload_dict={},
            input_type="text",
        )

        assert "offline_first" in result
        assert result["offline_first"] is False  # Default config


class TestGetDecoder:
    """Tests for _get_decoder method."""

    def test_offline_first_returns_offline_decoder(self, pipeline):
        """When offline_first is True, the offline decoder should be returned."""
        pipeline.config.offline_first = True
        pipeline._offline_decoder = MagicMock()
        pipeline._offline_decoder.decode.return_value = ("offline", {})

        decoder = pipeline._get_decoder()
        result, usage = decoder.decode("input", "q")
        assert result == "offline"

    def test_fallback_decoder_returns_wrapper(self, pipeline):
        """When fallback decoder exists, a wrapper holding both should be returned."""
        pipeline.config.offline_first = False
        pipeline._fallback_decoder = MagicMock()

        decoder = pipeline._get_decoder()
        assert isinstance(decoder, _FallbackDecoderWrapper)
        assert decoder.primary is pipeline.remote_decoder
        assert decoder.fallback is pipeline._fallback_decoder

    def test_no_fallback_returns_remote_directly(self, pipeline):
        """When no fallback and no offline, return remote_decoder directly."""
        pipeline.config.offline_first = False
        pipeline._fallback_decoder = None

        decoder = pipeline._get_decoder()
        assert decoder is pipeline.remote_decoder


class TestCompressPrompt:
    """Tests for compress_prompt method."""

    def test_compress_prompt_delegates_to_text_processor(self, pipeline):
        """compress_prompt should delegate to text_processor.compress_prompt."""
        with patch.object(pipeline.text_processor, "compress_prompt") as mock_cp:
            mock_cp.return_value = {"compressed_prompt": "short", "original_tokens": 100}
            result = pipeline.compress_prompt("long text here")
            assert result["compressed_prompt"] == "short"
            mock_cp.assert_called_once_with("long text here")


class TestEstimateCost:
    """Tests for estimate_cost method."""

    def test_estimate_cost_structure(self, pipeline):
        """estimate_cost should return expected structure with defaults."""
        result = pipeline.estimate_cost(input_tokens=1000, output_tokens=200)
        assert result["provider"] == "ollama"  # From config
        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 200
        assert "estimated_cost" in result
        assert "savings_per_query" in result
        assert "savings_percentage" in result
