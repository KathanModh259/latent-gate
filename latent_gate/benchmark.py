"""Benchmark and evaluation helpers for LatentGate.

The benchmark focuses on product-critical signals: latency, token savings,
compression ratio, and failure rate. It can run built-in cases or JSONL cases.
"""

import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from latent_gate.config import PipelineConfig
from latent_gate.fast_client import FastClient
from latent_gate.text_processor import TextProcessor


@dataclass
class BenchmarkCase:
    name: str
    text: str
    question: str = ""
    mode: str = "auto"


@dataclass
class BenchmarkResult:
    name: str
    ok: bool
    original_tokens: int = 0
    compressed_tokens: int = 0
    tokens_saved: int = 0
    compression_ratio: float = 0.0
    latency_ms: float = 0.0
    mode: str = ""
    error: str = ""


BUILTIN_CASES = [
    BenchmarkCase(
        name="short_instruction",
        text="Explain why local-first AI compression reduces cost and improves privacy.",
    ),
    BenchmarkCase(
        name="long_product_prompt",
        text=(
            "You are helping build a production-ready developer tool. "
            "Review the architecture, identify performance risks, security risks, "
            "developer experience gaps, deployment concerns, observability gaps, "
            "testing gaps, and documentation gaps. Prioritize the work into immediate, "
            "next, and later milestones. Include specific acceptance criteria for each "
            "milestone. Avoid vague advice and focus on implementation steps. "
        )
        * 12,
        question="Create a production readiness plan.",
    ),
    BenchmarkCase(
        name="code_review_prompt",
        mode="code",
        text=(
            "Please review this code for concurrency bugs, security issues, and performance.\n\n"
            "```python\n"
            "cache = {}\n"
            "def get_or_set(key, build):\n"
            "    if key not in cache:\n"
            "        cache[key] = build()\n"
            "    return cache[key]\n"
            "```\n\n"
            "The function is called by a web server with many concurrent requests."
        ),
    ),
]


def load_cases(path: str = "") -> list[BenchmarkCase]:
    """Load JSONL benchmark cases or return built-ins."""
    if not path:
        return list(BUILTIN_CASES)

    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            if "name" not in data or "text" not in data:
                raise ValueError(f"Benchmark case line {line_number} needs name and text")
            cases.append(
                BenchmarkCase(
                    name=str(data["name"]),
                    text=str(data["text"]),
                    question=str(data.get("question", "")),
                    mode=str(data.get("mode", "auto")),
                )
            )
    return cases


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize_results(results: list[BenchmarkResult]) -> dict:
    ok_results = [r for r in results if r.ok]
    latencies = [r.latency_ms for r in ok_results]
    original = sum(r.original_tokens for r in ok_results)
    compressed = sum(r.compressed_tokens for r in ok_results)
    saved = sum(r.tokens_saved for r in ok_results)

    return {
        "cases": len(results),
        "successful": len(ok_results),
        "failed": len(results) - len(ok_results),
        "average_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 50), 1),
        "p95_latency_ms": round(percentile(latencies, 95), 1),
        "original_tokens": original,
        "compressed_tokens": compressed,
        "tokens_saved": saved,
        "savings_percentage": round((saved / max(original, 1)) * 100, 1),
        "average_compression_ratio": round(original / max(compressed, 1), 2),
    }


def run_text_benchmark(
    config: PipelineConfig,
    cases: Iterable[BenchmarkCase],
) -> dict:
    """Run text compression benchmark cases."""
    client = FastClient(config)
    processor = TextProcessor(config, client=client)
    results: list[BenchmarkResult] = []

    try:
        for case in cases:
            start = time.perf_counter()
            try:
                payload = processor.compress(
                    case.text,
                    mode=case.mode,
                    question=case.question,
                )
                payload.to_compact_prompt()
                latency_ms = (time.perf_counter() - start) * 1000
                results.append(
                    BenchmarkResult(
                        name=case.name,
                        ok=True,
                        original_tokens=payload.original_token_count,
                        compressed_tokens=payload.compressed_token_count,
                        tokens_saved=payload.original_token_count - payload.compressed_token_count,
                        compression_ratio=round(payload.compression_ratio, 2),
                        latency_ms=round(latency_ms, 1),
                        mode=case.mode,
                    )
                )
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                results.append(
                    BenchmarkResult(
                        name=case.name,
                        ok=False,
                        latency_ms=round(latency_ms, 1),
                        mode=case.mode,
                        error=str(e),
                    )
                )
    finally:
        client.close()

    return {
        "summary": summarize_results(results),
        "results": [asdict(r) for r in results],
    }


def write_report(report: dict, output_path: str) -> None:
    """Write a JSON benchmark report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

