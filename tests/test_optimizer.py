"""Tests for the deterministic TokenOptimizer — each test pins one guarantee."""

import json
from unittest.mock import patch

import pytest

from latent_gate import optimizer as opt_mod
from latent_gate.config import PipelineConfig
from latent_gate.optimizer import (
    TokenOptimizer,
    count_tokens,
    extract_facts,
    missing_facts,
    optimize,
)
from latent_gate.text_processor import TextProcessor

PRETTY_JSON = json.dumps(
    {"users": [{"id": i, "score": 1.0, "big": 1e5, "tags": ["a b", "c"]} for i in range(10)]},
    indent=4,
)
LOGS = "\n".join(
    f"2026-09-24 12:00:{i:02d} ERROR worker-{i % 3} request id={1000 + i} failed: timeout"
    for i in range(30)
)
REQUIREMENTS = "Build it.\n" + "\n".join(
    f"- requirement {i}: must handle case {i}" for i in range(1, 9)
)

SAMPLES = [
    "",
    "hi",
    "What is 2+2?",
    PRETTY_JSON,
    LOGS,
    REQUIREMENTS,
    "Please  fix   this.\n\n\n\nThanks!!",
    "```python\ndef f(x):\n    return x  # please keep  me\n```",
]


# ---------------------------------------------------------------- guarantees


@pytest.mark.parametrize("level", ["lossless", "balanced", "aggressive"])
@pytest.mark.parametrize("text", SAMPLES)
def test_never_increases_tokens(text, level):
    r = TokenOptimizer(level).optimize(text)
    assert r.optimized_tokens <= r.original_tokens
    assert r.optimized_tokens == count_tokens(r.text)


@pytest.mark.parametrize("text", SAMPLES)
def test_deterministic(text):
    assert optimize(text).text == optimize(text).text


def test_code_block_is_restored_byte_for_byte():
    code = "```python\ndef f(x):\n    y  =  x   # please kindly keep  spacing\n    return y\n```"
    r = optimize(f"Could you please review this?\n\n{code}\n\nThanks!", level="aggressive")
    assert code in r.text


def test_urls_and_quotes_are_protected():
    text = 'Fetch https://api.example.com/v1/items?limit=10 and match  "exact   phrase  here".'
    out = optimize(text).text
    assert "https://api.example.com/v1/items?limit=10" in out
    assert '"exact   phrase  here"' in out


def test_json_minify_is_lossless_and_keeps_number_spelling():
    text = f"Data:\n```json\n{PRETTY_JSON}\n```"
    r = optimize(text, level="lossless")
    body = r.text.split("```json\n", 1)[1].rsplit("\n```", 1)[0]
    assert json.loads(body) == json.loads(PRETTY_JSON)
    assert "1.0" in body and "100000.0" in body  # re-serialization would rewrite spellings
    assert '"a b"' in body  # whitespace inside strings survives
    assert r.savings_pct > 30


def test_unfenced_json_in_prose_is_minified():
    r = optimize("Here is the payload:\n" + PRETTY_JSON, level="lossless")
    assert json.loads(r.text.split("\n", 1)[1]) == json.loads(PRETTY_JSON)


def test_log_runs_collapse_but_keep_first_and_last():
    r = optimize(f"Why does this fail?\n\n```\n{LOGS}\n```")
    assert "id=1000" in r.text and "id=1029" in r.text
    assert "28 similar lines" in r.text
    assert r.savings_pct > 70


def test_requirement_lists_are_never_folded():
    """Lines differing only by numbers are folded ONLY for logs, never for prose lists."""
    r = optimize(REQUIREMENTS, level="balanced")
    for i in range(1, 9):
        assert f"requirement {i}:" in r.text


def test_exact_duplicate_lines_keep_count():
    r = optimize("Retrying...\nRetrying...\nRetrying...\nDone", level="lossless")
    assert "Retrying... [×3]" in r.text


def test_duplicate_paragraphs_removed():
    para = "Build a REST API in FastAPI with JWT auth and Postgres storage, rate limit 100/min."
    r = optimize(f"{para}\n\n{para}\n\n{para}", level="lossless")
    assert r.text.count("FastAPI") == 1


def test_lossless_level_keeps_every_fact():
    text = f"Use v2.3.1 of `pkg` at https://x.io.\n\n{PRETTY_JSON}\n\nCall get_user_by_id(42)."
    r = optimize(text, level="lossless")
    assert missing_facts(text, r.text) == []


def test_unterminated_fence_is_protected():
    text = "Explain this:\n```js\nconst  a =  1;   // please"
    out = optimize(text).text
    assert "const  a =  1;   // please" in out


# ---------------------------------------------------------------- selection

LONG_DOC = " ".join(
    [
        "Our company was founded in 1998 in a small garage.",
        "The cafeteria serves lunch from noon until two.",
        "The refund policy allows returns within 30 days of purchase with a receipt.",
        "Employees enjoy a gym membership and free parking.",
        "Refunds are issued to the original payment method within 5 business days.",
        "The annual picnic takes place every July at the lake.",
    ]
    * 3
)


def test_selection_respects_budget_and_keeps_relevant_sentences():
    r = TokenOptimizer("balanced").optimize(
        LONG_DOC, question="What is the refund policy?", max_tokens=60
    )
    assert r.optimized_tokens <= 60
    assert "30 days" in r.text
    assert "picnic" not in r.text
    assert r.units_dropped > 0


def test_aggressive_defaults_to_half_budget():
    r = TokenOptimizer("aggressive").optimize(LONG_DOC)
    assert r.optimized_tokens <= r.original_tokens * 0.5 + 1


def test_selection_never_drops_code_blocks():
    code = "```\nSELECT * FROM refunds WHERE days <= 30;\n```"
    r = TokenOptimizer("balanced").optimize(
        f"{LONG_DOC}\n\n{code}", question="refunds", max_tokens=40
    )
    assert code in r.text


def test_documents_drop_irrelevant_docs_and_keep_labels():
    docs = [
        "The picnic is in July. Bring sunscreen and snacks for everyone attending.",
        "Refunds: returns are accepted within 30 days with a receipt. Refunds take 5 days.",
        "Parking is free for employees in lot B. Visitors must register at reception.",
    ]
    r = TokenOptimizer("balanced").optimize_documents(
        docs, question="How do refunds work?", target_ratio=0.5
    )
    assert "[Doc 2]:" in r.text and "30 days" in r.text and "5 days" in r.text
    assert "picnic" not in r.text
    assert "[Doc 3]" not in r.text  # budget left over is not spent on irrelevant chunks


# ---------------------------------------------------------------- facts & counting


def test_extract_and_missing_facts():
    text = "Set MAX_RETRIES to 5 in config.yaml, see https://docs.x.io, call parseJSON()."
    facts = extract_facts(text)
    assert {"MAX_RETRIES", "5", "config.yaml", "https://docs.x.io", "parseJSON"} <= facts
    assert missing_facts(text, "Set retries in the config.") == sorted(facts)


def test_heuristic_counter_used_without_tiktoken():
    opt_mod._get_encoder.cache_clear()
    try:
        with patch.dict("sys.modules", {"tiktoken": None}):
            opt_mod._get_encoder.cache_clear()
            assert opt_mod.token_counter_name() == "heuristic"
            assert 8 <= count_tokens("This is a test sentence with ten words in it") <= 14
    finally:
        opt_mod._get_encoder.cache_clear()


# ---------------------------------------------------------------- LLM guardrail


@pytest.fixture
def processor():
    return TextProcessor(PipelineConfig(remote_provider="ollama", compression_strategy="auto"))


VERBOSE = (
    "I would like you to write a Python function called parse_rows that reads data.csv, "
    "skips the first 2 rows, and returns at most 100 records as dictionaries. "
) * 3


def test_llm_rewrite_rejected_when_it_drops_a_fact(processor):
    bad = "Write a Python function that reads a CSV and returns records."
    with patch.object(processor, "_ollama_generate_with_fallback", return_value=(bad, "m")):
        out = processor.compress_prompt(VERBOSE)
    assert out["method"].startswith("optimizer/")
    assert "parse_rows" in out["compressed_prompt"]


def test_llm_rewrite_accepted_when_shorter_and_faithful(processor):
    good = (
        "Write Python parse_rows: read data.csv, skip first 2 rows, return <=100 records as dicts."
    )
    with patch.object(processor, "_ollama_generate_with_fallback", return_value=(good, "m")):
        out = processor.compress_prompt(VERBOSE)
    assert out["method"] == "llm/m"
    assert out["compressed_prompt"] == good
    assert out["original_tokens"] == count_tokens(VERBOSE)


def test_deterministic_strategy_never_calls_llm():
    tp = TextProcessor(
        PipelineConfig(remote_provider="ollama", compression_strategy="deterministic")
    )
    with patch.object(tp, "_ollama_generate_with_fallback") as llm:
        tp.compress_prompt(VERBOSE)
        tp.compress(VERBOSE * 5)
    llm.assert_not_called()


# ---------------------------------------------------------------- prompt structure


def test_pleasantry_sentences_removed_but_content_kept():
    text = "Hi there! I hope you're doing well.\nBuild a CLI in Go. Thanks — also add tests.\nThank you so much in advance!"
    out = optimize(text).text
    assert "Hi there" not in out and "doing well" not in out and "in advance" not in out
    assert "Build a CLI in Go." in out
    assert "Thanks — also add tests." in out  # has content, so it's not a pure pleasantry


def test_aggressive_never_drops_the_ask():
    text = (
        "I'm working on a side project and would love some input on it. "
        "It has been a long journey so far and I learned a lot along the way. "
        "Act as an expert engineer. Build a REST API in FastAPI.\n"
        "- JWT authentication\n- Rate limiting at 100 requests per minute\n- PostgreSQL storage"
    )
    out = TokenOptimizer("aggressive").optimize(text).text
    assert "Build a REST API in FastAPI." in out


def test_log_summary_keeps_ranges_of_varying_fields():
    out = optimize(f"```\n{LOGS}\n```").text
    marker = next(line for line in out.splitlines() if "similar lines" in line)
    assert "id=1001" in marker and "1028" in marker
    assert "worker-0" in marker


# ---------------------------------------------------------------- filler phrases

from latent_gate.optimizer import strip_filler_phrases  # noqa: E402


@pytest.mark.parametrize(
    "before,after",
    [
        (
            "Could you please, if at all possible, kindly explain to me in great detail "
            "what the main differences are between TCP and UDP protocols?",
            "Explain in detail the main differences between TCP and UDP protocols?",
        ),
        (
            "I'm working on a project and I would really appreciate your help with it.",
            "I'm working on a project.",
        ),
        ("Please make sure the code is clean.", "Make sure the code is clean."),
        ("- please add unit tests", "- Add unit tests"),
        (
            "I would like you to write a Python script in order to parse logs, thanks!",
            "Write a Python script to parse logs",
        ),
        ("Can you review this? It is slow.", "Review this? It is slow."),
    ],
)
def test_filler_removed(before, after):
    assert strip_filler_phrases(before) == after


@pytest.mark.parametrize(
    "text",
    [
        "Use caching if possible.",  # a real soft preference
        "Compute the derivative with respect to x.",  # precise math phrasing
        "Write an email that says thank you.",  # "thank you" is the content
        "Explain what the problem is.",
        "Only return JSON, exactly 3 keys, step by step.",
        "Copy the .env file to port :8080 now.",
        "Tell me if you can reproduce it.",
        "    indented line keeps its leading spaces",
    ],
)
def test_filler_leaves_meaningful_text_alone(text):
    assert strip_filler_phrases(text) == text


def test_filler_never_touches_protected_spans():
    text = 'Could you please run `please --kindly` and match "could you please"?'
    out = optimize(text).text
    assert "`please --kindly`" in out and '"could you please"' in out
    assert out.startswith("Run ")
