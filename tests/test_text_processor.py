"""Tests for TextProcessor and TextPayload."""

import pytest
from latent_gate.config import PipelineConfig
from latent_gate.text_processor import TextProcessor, TextPayload


class TestTextPayload:

    def test_default_empty(self):
        p = TextPayload()
        assert p.intent == ""
        assert p.key_entities == []
        assert p.compression_ratio == 0.0

    def test_with_data(self):
        p = TextPayload(
            intent="Fix the JWT refresh bug",
            key_entities=["Flask", "PyJWT", "PostgreSQL"],
            constraints=["Must handle race conditions", "Under 2 seconds"],
            question_type="code",
        )
        assert p.intent == "Fix the JWT refresh bug"
        assert len(p.key_entities) == 3

    def test_to_compact_prompt(self):
        p = TextPayload(
            intent="Optimize database query",
            key_entities=["PostgreSQL", "indexing"],
            constraints=["Must be under 100ms"],
            question_type="code",
            tone="technical",
        )
        compact = p.to_compact_prompt()

        assert "[Type: code]" in compact
        assert "Optimize database query" in compact
        assert "PostgreSQL" in compact
        assert "100ms" in compact
        assert p.compressed_token_count > 0

    def test_to_compact_prompt_empty(self):
        p = TextPayload()
        compact = p.to_compact_prompt()
        assert compact == ""
        assert p.compressed_token_count == 10

    def test_to_dict_and_from_dict(self):
        p = TextPayload(
            intent="Test intent",
            key_entities=["entity1"],
            original_token_count=500,
            compressed_token_count=100,
            compression_ratio=5.0,
        )
        d = p.to_dict()
        assert isinstance(d, dict)

        p2 = TextPayload.from_dict(d)
        assert p2.intent == p.intent
        assert p2.compression_ratio == p.compression_ratio

    def test_from_dict_ignores_extra_fields(self):
        d = {"intent": "test", "unknown_field": "ignored"}
        p = TextPayload.from_dict(d)
        assert p.intent == "test"

    def test_compression_ratio_calculation(self):
        p = TextPayload(
            intent="Summarize this document",
            key_entities=["doc1", "doc2"],
            original_token_count=500,
        )
        p.to_compact_prompt()
        assert p.compression_ratio > 1.0  # Should be compressed

    def test_code_snippets_in_compact(self):
        p = TextPayload(
            intent="Fix this code",
            code_snippets=["def foo(): return 42"],
        )
        compact = p.to_compact_prompt()
        assert "def foo()" in compact

    def test_repr(self):
        p = TextPayload(intent="Test intent here", compressed_token_count=50, compression_ratio=3.5)
        r = repr(p)
        assert "Test intent" in r
        assert "50" in r
        assert "3.5" in r


class TestTextProcessorHelpers:

    def setup_method(self):
        config = PipelineConfig(enable_caching=False)
        self.processor = TextProcessor(config)

    def test_estimate_tokens(self):
        text = "This is a test sentence with ten words in it"
        tokens = self.processor._estimate_tokens(text)
        # ~10 words * 1.33 ≈ 13
        assert 10 < tokens < 20

    def test_estimate_tokens_empty(self):
        assert self.processor._estimate_tokens("") == 0

    def test_detect_mode_code(self):
        text = "I have this code:\ndef foo():\n    return 42"
        assert self.processor._detect_mode(text) == "code"

    def test_detect_mode_code_import(self):
        text = "import pandas as pd\nimport numpy as np\ndf = pd.read_csv('data.csv')"
        assert self.processor._detect_mode(text) == "code"

    def test_detect_mode_conversation(self):
        text = "User: Hello\nAssistant: Hi there\nUser: Help me with X"
        assert self.processor._detect_mode(text) == "summarize"

    def test_detect_mode_long_text(self):
        text = " ".join(["word"] * 250)  # 250 words
        assert self.processor._detect_mode(text) == "condense"

    def test_detect_mode_short_text(self):
        text = "What is the capital of France?"
        assert self.processor._detect_mode(text) == "compress"

    def test_detect_mode_explicit(self):
        text = "any text here"
        assert self.processor._detect_mode(text, mode="summarize") == "summarize"
        assert self.processor._detect_mode(text, mode="code") == "code"

    def test_parse_response_valid_json(self):
        raw = '''{"intent": "test intent", "key_entities": ["a", "b"], "constraints": [], "context_summary": "summary", "question_type": "factual", "output_format": "paragraph", "tone": "casual", "data_points": ["42"]}'''
        payload = self.processor._parse_response(raw, "original text " * 50)
        assert payload.intent == "test intent"
        assert payload.key_entities == ["a", "b"]
        assert payload.question_type == "factual"
        assert payload.original_token_count > 0

    def test_parse_response_invalid_json(self):
        raw = "This is not valid JSON at all"
        payload = self.processor._parse_response(raw, "original text")
        # Should fallback gracefully
        assert payload.intent == "Process the following request"
        assert "not valid JSON" in payload.context_summary

    def test_parse_response_with_markdown_fences(self):
        raw = '''```json
{"intent": "test", "key_entities": ["x"]}
```'''
        payload = self.processor._parse_response(raw, "original " * 50)
        assert payload.intent == "test"

    def test_compress_short_text_skips(self):
        """Short text (<100 tokens) should skip compression."""
        payload = self.processor.compress("What is 2+2?")
        assert payload.compression_ratio == 1.0
        assert payload.intent == "What is 2+2?"
