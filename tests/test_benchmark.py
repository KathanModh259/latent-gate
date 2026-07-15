"""Tests for benchmark summary helpers."""

from latent_gate.benchmark import BenchmarkResult, load_cases, summarize_results


def test_load_builtin_cases():
    cases = load_cases()
    assert len(cases) >= 1
    assert all(case.name for case in cases)
    assert all(case.text for case in cases)


def test_summarize_results_successes_and_failures():
    results = [
        BenchmarkResult(
            name="a",
            ok=True,
            original_tokens=100,
            compressed_tokens=25,
            tokens_saved=75,
            latency_ms=100,
        ),
        BenchmarkResult(
            name="b",
            ok=True,
            original_tokens=200,
            compressed_tokens=100,
            tokens_saved=100,
            latency_ms=200,
        ),
        BenchmarkResult(name="c", ok=False, error="boom", latency_ms=50),
    ]

    summary = summarize_results(results)

    assert summary["cases"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["original_tokens"] == 300
    assert summary["compressed_tokens"] == 125
    assert summary["tokens_saved"] == 175
    assert summary["average_compression_ratio"] == 2.4
