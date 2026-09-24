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


# ============================================================================
# Deterministic optimizer benchmark (no Ollama, reproducible anywhere)
# ============================================================================


def _realistic_cases() -> list[dict]:
    """Inputs shaped like real developer traffic. Built deterministically."""
    api_json = json.dumps(
        {
            "data": [
                {
                    "id": 1000 + i,
                    "email": f"user{i}@example.com",
                    "active": i % 4 != 0,
                    "plan": ["free", "pro", "team"][i % 3],
                    "last_login": f"2026-09-{(i % 28) + 1:02d}T10:00:00Z",
                }
                for i in range(25)
            ],
            "page": 1,
            "total": 25,
        },
        indent=2,
    )
    logs = "\n".join(
        f"2026-09-24 12:{i // 60:02d}:{i % 60:02d} ERROR [worker-{i % 4}] "
        f"POST /api/orders id={5000 + i} failed: upstream timeout after 30000ms"
        for i in range(60)
    )
    trace = (
        "Traceback (most recent call last):\n"
        '  File "app/orders.py", line 88, in create_order\n'
        "    resp = client.post(url, json=payload, timeout=30)\n"
        "requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='pay.example.com', "
        "port=443): Read timed out. (read timeout=30)"
    )
    spec = (
        "Hi! I hope you're doing well. I'm working on a project and I would really appreciate "
        "your help with it.\n\nAct as an expert backend engineer. Build a REST API in FastAPI.\n"
        "The application must include the following features:\n"
        "- JWT authentication with refresh tokens\n"
        "- Rate limiting at 100 requests per minute per user\n"
        "- PostgreSQL storage using SQLAlchemy 2.0\n"
        "- Pagination on every list endpoint (default page size 20)\n"
        "- OpenAPI docs at /docs\n"
        "- Unit tests with pytest reaching 90% coverage\n\n"
        "Please make sure the code is clean and well structured. Thank you so much in advance!"
    )
    code = (
        "Can you please review this function for concurrency bugs? It is called by many threads.\n\n"
        "```python\n"
        "cache = {}\n\n\n"
        "def get_or_set(key, build):\n"
        "    if key not in cache:          \n"
        "        cache[key] = build()      \n"
        "    return cache[key]\n"
        "```\n\nThanks!"
    )
    chat_paste = (
        "Summarize the decision below for the team.\n\n"
        + (
            "We decided to migrate the billing service from MySQL 5.7 to PostgreSQL 16 in Q4, "
            "keeping the old database read-only for 30 days as a fallback.\n\n"
        )
        * 3
    )
    rag_docs = [
        "Our office is open Monday to Friday from 9am to 6pm. The cafeteria serves lunch "
        "from noon until 2pm. Parking is free in lot B for all employees.",
        "Refund policy: customers may return any product within 30 days of purchase with a "
        "receipt. Refunds are issued to the original payment method within 5 business days. "
        "Opened software is not eligible for a refund.",
        "The annual company picnic takes place every July at Lake Merritt. Families are "
        "welcome and there will be games, food trucks and live music.",
        "Shipping: standard orders ship within 2 business days. Express shipping is available "
        "for an extra $15. International orders may take up to 14 days.",
        "Returned items must be unused and in original packaging. A 10% restocking fee applies "
        "to electronics returned after 14 days.",
    ]
    return [
        {"name": "verbose_spec", "text": spec},
        {
            "name": "api_json_response",
            "text": f"Which users are inactive?\n\n```json\n{api_json}\n```",
        },
        {"name": "logs_and_trace", "text": f"Why do orders fail?\n\n```\n{logs}\n```\n\n{trace}"},
        {"name": "code_review", "text": code},
        {"name": "duplicated_paste", "text": chat_paste},
        {
            "name": "rag_documents",
            "documents": rag_docs,
            "question": "Can I get a refund on a return?",
        },
    ]


def run_optimizer_benchmark(levels=("lossless", "balanced", "aggressive")) -> dict:
    """
    Measure the deterministic optimizer: real tokenizer counts, latency, and
    fact retention (share of numbers/identifiers/URLs/code kept verbatim).
    """
    from latent_gate.optimizer import (
        TokenOptimizer,
        extract_facts,
        missing_facts,
        token_counter_name,
    )

    cases = _realistic_cases()
    report = {"token_counter": token_counter_name(), "levels": {}}
    for level in levels:
        opt = TokenOptimizer(level)
        rows = []
        for case in cases:
            start = time.perf_counter()
            if "documents" in case:
                r = opt.optimize_documents(
                    case["documents"],
                    question=case["question"],
                    target_ratio=0.0 if level == "lossless" else 0.5,
                )
                source = "\n\n".join(case["documents"])
            else:
                r = opt.optimize(case["text"], question=case.get("question", ""))
                source = case["text"]
            latency_ms = (time.perf_counter() - start) * 1000
            facts = extract_facts(source)
            lost = missing_facts(source, r.text)
            rows.append(
                {
                    "name": case["name"],
                    "original_tokens": r.original_tokens,
                    "optimized_tokens": r.optimized_tokens,
                    "savings_pct": round(r.savings_pct, 1),
                    "fact_retention_pct": round(100 * (1 - len(lost) / max(len(facts), 1)), 1),
                    "latency_ms": round(latency_ms, 2),
                }
            )
        original = sum(r["original_tokens"] for r in rows)
        optimized = sum(r["optimized_tokens"] for r in rows)
        report["levels"][level] = {
            "original_tokens": original,
            "optimized_tokens": optimized,
            "savings_pct": round(100 * (original - optimized) / max(original, 1), 1),
            "mean_fact_retention_pct": round(
                statistics.mean(r["fact_retention_pct"] for r in rows), 1
            ),
            "p95_latency_ms": round(percentile([r["latency_ms"] for r in rows], 95), 2),
            "cases": rows,
        }
    return report
