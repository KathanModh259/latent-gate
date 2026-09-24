"""
Text Processor — Local prompt compression & tokenization.

Extends LatentGate beyond vision: compresses long text prompts locally
(FREE via Ollama) before sending to cloud LLMs.

Use cases:
  - Long prompts (500+ tokens → ~100-150 tokens)
  - Multi-turn conversations (growing context → summarized history)
  - RAG / Document Q&A (verbose retrieved chunks → key facts)
  - Code analysis (full files → structured summary)
  - Any scenario where the user's input is verbose

Inspired by VL-JEPA's principle: heavy semantic processing in latent
space locally, lightweight decoding remotely.
"""

import json
import re
import time
from typing import Optional
import logging
from dataclasses import dataclass, field, asdict

from latent_gate.config import PipelineConfig
from latent_gate.fast_client import OllamaUnavailableError
from latent_gate.optimizer import (
    OptimizationResult,
    TokenOptimizer,
    count_tokens,
    missing_facts,
)

logger = logging.getLogger("latent_gate.text")


# ============================================================================
# TEXT PAYLOAD — Compressed representation of text input
# ============================================================================


@dataclass
class TextPayload:
    """
    Compact structured representation of a text prompt.

    Instead of sending the user's full 500+ token prompt to the cloud,
    we extract the essential semantics locally and send ~100-150 tokens.
    """

    # ---- Core Extracted Semantics ----
    intent: str = ""  # What the user wants (1 sentence)
    key_entities: list = field(default_factory=list)  # Important names/values/concepts
    constraints: list = field(default_factory=list)  # Requirements/conditions/rules
    context_summary: str = ""  # Compressed background/context
    question_type: str = ""  # factual/creative/analytical/code/etc.
    output_format: str = ""  # What format the user expects
    tone: str = ""  # formal/casual/technical/etc.
    code_snippets: list = field(default_factory=list)  # Any code blocks (preserved as-is)
    data_points: list = field(default_factory=list)  # Numbers, dates, values mentioned
    # When set, this verbatim-derived text is sent instead of the structured fields
    # (produced by the deterministic TokenOptimizer, so nothing is paraphrased)
    optimized_text: str = ""

    # ---- Metadata ----
    original_token_count: int = 0
    compressed_token_count: int = 0
    compression_ratio: float = 0.0
    processing_time_ms: float = 0.0
    processor_model: str = ""

    def to_compact_prompt(self) -> str:
        """
        Convert to minimal text for the cloud LLM.
        This is what gets sent instead of the full original prompt.
        """
        if self.optimized_text:
            compact = self.optimized_text
        else:
            parts = []
            if self.question_type:
                parts.append(f"[Type: {self.question_type}]")
            if self.intent:
                parts.append(f"Intent: {self.intent}")
            # No caps on list lengths or code size: silently truncating constraints
            # or code changes what the user asked for.
            if self.key_entities:
                parts.append(f"Entities: {', '.join(str(e) for e in self.key_entities)}")
            if self.constraints:
                parts.append(f"Constraints: {'; '.join(str(c) for c in self.constraints)}")
            if self.context_summary:
                parts.append(f"Context: {self.context_summary}")
            if self.output_format:
                parts.append(f"Output format: {self.output_format}")
            if self.tone:
                parts.append(f"Tone: {self.tone}")
            if self.data_points:
                parts.append(f"Data: {', '.join(str(d) for d in self.data_points)}")
            for i, snippet in enumerate(self.code_snippets):
                parts.append(f"Code[{i}]: {snippet}")
            compact = " | ".join(parts)

        # Same tokenizer as original_token_count, so the ratio is a real measurement
        self.compressed_token_count = count_tokens(compact)
        if self.original_token_count > 0:
            self.compression_ratio = self.original_token_count / max(self.compressed_token_count, 1)
        return compact

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TextPayload":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"TextPayload(intent={self.intent[:50]!r}..., "
            f"~{self.compressed_token_count} tokens, "
            f"{self.compression_ratio:.1f}x compression)"
        )


# ============================================================================
# TEXT PROCESSOR — Local prompt compression
# ============================================================================


class TextProcessor:
    """
    Compresses long text prompts locally using Ollama.

    Modes:
      - compress:    Extract intent + entities + constraints from a prompt
      - summarize:   Summarize long context/conversation history
      - condense:    Condense RAG-retrieved documents into key facts
      - code_review: Extract structured info from code-heavy prompts
    """

    # ---- Extraction Prompts ----

    COMPRESS_PROMPT = """You are a prompt compression engine. Extract ONLY the essential semantics from the user's prompt.

USER PROMPT:
{user_text}

Return ONLY valid JSON, no explanation:
{{
  "intent": "what the user wants in 1 clear sentence",
  "key_entities": ["important names", "values", "concepts"],
  "constraints": ["requirement 1", "condition 2"],
  "context_summary": "compressed background info in 1-2 sentences",
  "question_type": "factual/creative/analytical/code/comparison/instruction",
  "output_format": "expected output format (list/paragraph/code/table/etc.)",
  "tone": "formal/casual/technical",
  "data_points": ["any specific numbers, dates, or values mentioned"]
}}"""

    SUMMARIZE_PROMPT = """Compress this conversation history into a minimal summary.
Keep ONLY information needed to continue the conversation.
Remove greetings, repetition, and filler.

CONVERSATION:
{user_text}

Return ONLY valid JSON:
{{
  "intent": "current topic/goal of the conversation",
  "context_summary": "key facts and decisions made so far (2-3 sentences max)",
  "key_entities": ["important names/values discussed"],
  "constraints": ["any requirements or preferences stated"],
  "data_points": ["specific numbers/dates/values mentioned"]
}}"""

    CONDENSE_PROMPT = """Extract ONLY the facts relevant to answering the question from these retrieved documents.
Remove all boilerplate, headers, navigation text, and irrelevant content.

QUESTION: {question}

DOCUMENTS:
{user_text}

Return ONLY valid JSON:
{{
  "intent": "the question being answered",
  "context_summary": "relevant facts extracted from documents (3-4 sentences max)",
  "key_entities": ["key names/values from documents"],
  "data_points": ["specific numbers/dates/statistics found"],
  "constraints": ["any caveats or conditions mentioned in documents"]
}}"""

    CODE_PROMPT = """Analyze this code-related prompt. Preserve code snippets exactly but compress the surrounding text.

PROMPT:
{user_text}

Return ONLY valid JSON:
{{
  "intent": "what the user wants done with the code",
  "context_summary": "compressed explanation around the code",
  "key_entities": ["languages", "frameworks", "libraries mentioned"],
  "constraints": ["requirements for the code"],
  "question_type": "code",
  "code_snippets": ["extract each code block exactly as-is"]
}}"""

    COMPRESS_ONLY_PROMPT = """Shorten this prompt so it keeps every requirement but uses half the words.

Examples of good compression:

Original: "Act as an expert full-stack developer. Create a single-page Google Search clone using HTML, CSS (Tailwind CSS via CDN), and vanilla JavaScript. The application must include the following features: 1. Homepage UI: A top navbar with links for Gmail and Images, an apps grid icon, and a blue Sign in button. 2. A centered Google logo and a rounded search bar with a magnifying glass icon and a hover shadow effect."
Shortened: "Build a Google Search clone in one HTML file with Tailwind CDN and vanilla JS. Include: top navbar (Gmail, Images, apps icon, Sign in), centered logo, rounded search bar with magnifying glass icon, hover shadow effect."

Original: "I need you to please help me write a Python script that can read a CSV file and then calculate the average of each column. The CSV file has headers in the first row. Please make sure to handle missing values gracefully. Also, I'd like the output to be printed in a nice formatted table. Thank you!"
Shortened: "Write a Python script that reads a CSV (headers in first row), calculates column averages, handles missing values gracefully, and prints results as a formatted table."

Now shorten this prompt (output ONLY the shorter version, no explanation):

{user_text}"""

    COMPRESS_ONLY_SHORT = """Condense this prompt to its essential requirements:

{user_text}"""

    def __init__(self, config: PipelineConfig, client=None):
        self.config = config
        self.client = client

    def _ollama_generate(self, prompt: str, max_tokens: int = 0, model: str = "") -> str:
        """Call Ollama for text compression. Uses shared FastClient if available."""
        if max_tokens <= 0:
            max_tokens = self.config.max_local_summary_tokens * 2
        if not model:
            model = self.config.text_fast_model
        if self.client:
            return self.client.ollama_generate(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
            )
        # Fallback: direct requests if no client shared
        import requests as _requests

        url = f"{self.config.ollama_base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
            },
        }
        try:
            resp = _requests.post(url, json=payload, timeout=self.config.request_timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except _requests.exceptions.ConnectionError:
            raise OllamaUnavailableError(
                "Cannot connect to Ollama. Make sure it's running: ollama serve"
            )
        except _requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.config.request_timeout}s. "
                "The model may be loading or the request is too large."
            )
        except _requests.exceptions.HTTPError as e:
            raise ConnectionError(
                f"Ollama HTTP error: {e}. Check if the model '{model}' is available."
            )

    def _estimate_tokens(self, text: str) -> int:
        """Token count (tiktoken o200k_base when installed, else a calibrated estimate)."""
        return count_tokens(text)

    def _detect_mode(self, text: str, mode: str = "auto") -> str:
        """Auto-detect the best compression mode based on content."""
        if mode != "auto":
            return mode

        # Simple heuristics
        code_indicators = ["```", "def ", "function ", "class ", "import ", "const ", "var "]
        if any(indicator in text for indicator in code_indicators):
            return "code"
        if re.search(
            r"\b(from\s+\w+\s+)?import\b|\bexport\b|```|interface\s+\w+|type\s+\w+\s*=", text
        ):
            return "code"

        # Multi-turn conversation pattern
        turn_indicators = ["User:", "Assistant:", "Human:", "AI:", "Q:", "A:"]
        if sum(1 for t in turn_indicators if t in text) >= 2:
            return "summarize"

        # Long text with question
        if len(text.split()) > 200:
            return "condense"

        return "compress"

    def _parse_response(self, raw: str, original_text: str, model_used: str = "") -> TextPayload:
        """Parse LLM output into TextPayload."""
        payload = TextPayload()
        payload.original_token_count = self._estimate_tokens(original_text)

        data = None
        cleaned = raw.strip()

        # Fast path 1: Try parsing direct or with basic cleanup of code fences
        try:
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fast path 2: Try to extract JSON from within mixed text
        if data is None:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except (json.JSONDecodeError, TypeError):
                    pass

        if data is not None:
            try:

                def _ensure_list(val):
                    return val if isinstance(val, list) else [val] if val else []

                payload.intent = str(data.get("intent", ""))
                payload.key_entities = [str(e) for e in _ensure_list(data.get("key_entities"))]
                payload.constraints = [str(e) for e in _ensure_list(data.get("constraints"))]
                payload.context_summary = str(data.get("context_summary", ""))
                payload.question_type = str(data.get("question_type", ""))
                payload.output_format = str(data.get("output_format", ""))
                payload.tone = str(data.get("tone", ""))
                payload.code_snippets = [str(e) for e in _ensure_list(data.get("code_snippets"))]
                payload.data_points = _ensure_list(data.get("data_points"))
            except Exception as e:
                logger.warning(f"Error reading JSON fields ({e}), falling back to raw text")
                data = None

        if data is None:
            logger.warning("JSON parse failed completely, using raw as context_summary")
            payload.intent = "Process the following request"
            payload.context_summary = raw[:500]

        payload.processor_model = model_used or self.config.text_fast_model
        return payload

    def _ollama_generate_with_fallback(
        self, prompt: str, max_tokens: int = 0, task: str = "text_fast"
    ) -> tuple:
        """Try models in fallback chain until one succeeds. Returns (response_text, model_used)."""
        chain = self.config.get_fallback_chain(task)
        last_error = None
        for model in chain:
            try:
                if max_tokens <= 0:
                    max_tokens = self.config.max_local_summary_tokens * 2
                if self.client:
                    resp = self.client.ollama_generate(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                    )
                else:
                    resp = self._ollama_generate(prompt, max_tokens, model)
                logger.info(f"Compression succeeded with {model}")
                return resp, model
            except OllamaUnavailableError:
                raise  # server is down: other models on it can't succeed either
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Model {model} failed ({e}), trying next in chain...")
                last_error = e
                continue
        raise last_error or ConnectionError("All compression models unavailable")

    # ----------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------

    def _optimizer(self) -> TokenOptimizer:
        level = self.config.compression_level
        return TokenOptimizer(
            level=level if level in ("lossless", "balanced", "aggressive") else "balanced"
        )

    @staticmethod
    def _payload_from_optimization(result: OptimizationResult) -> TextPayload:
        return TextPayload(
            optimized_text=result.text,
            original_token_count=result.original_tokens,
            compressed_token_count=result.optimized_tokens,
            compression_ratio=result.compression_ratio,
            processor_model=f"optimizer/{result.level}",
        )

    def _fallback_compress(
        self, text: str, mode: str = "compress", question: str = ""
    ) -> TextPayload:
        """
        Deterministic compression (used when Ollama is unavailable, or with
        compression_strategy="deterministic"). Sends the user's own wording,
        optimized — never an invented summary.
        """
        result = self._optimizer().optimize(
            text, question=question, max_tokens=self.config.target_token_budget
        )
        return self._payload_from_optimization(result)

    def _fallback_rewrite_prompt(self, text: str) -> str:
        """Deterministic prompt shortening (see latent_gate.optimizer)."""
        return self._optimizer().optimize(text, max_tokens=self.config.target_token_budget).text

    def compress(
        self,
        text: str,
        mode: str = "auto",
        question: str = "",
        _optimized: Optional[OptimizationResult] = None,
    ) -> TextPayload:
        """
        Compress a text prompt locally.

        Stage 1 always runs the deterministic TokenOptimizer. With
        compression_strategy="auto", the local LLM then extracts a structured
        payload from the optimized text; that payload is used only if it is
        smaller AND keeps every critical fact (numbers, identifiers, code, URLs).

        Args:
            text:     The full user prompt / text to compress.
            mode:     "auto" | "compress" | "summarize" | "condense" | "code"
            question: The question being answered (focuses selection/condensing).

        Returns:
            TextPayload with compressed representation.
        """
        start = time.time()
        mode = self._detect_mode(text, mode)
        logger.info(f"Text compression mode: {mode}")

        det = _optimized or self._optimizer().optimize(
            text, question=question, max_tokens=self.config.target_token_budget
        )
        det_payload = self._payload_from_optimization(det)

        # Short or deterministic-only: an LLM call (seconds + JSON framing) can't beat this
        if self.config.compression_strategy == "deterministic" or det.optimized_tokens < 100:
            det_payload.processing_time_ms = (time.time() - start) * 1000
            return det_payload

        prompt_map = {
            "compress": self.COMPRESS_PROMPT,
            "summarize": self.SUMMARIZE_PROMPT,
            "condense": self.CONDENSE_PROMPT.replace("{question}", question or "Answer the query"),
            "code": self.CODE_PROMPT,
        }
        prompt = prompt_map.get(mode, self.COMPRESS_PROMPT).replace("{user_text}", det.text)

        task = "text_smart" if self.config._is_complex(det.text) else "text_fast"
        logger.info(f"LLM extraction on {det.optimized_tokens} tokens (task={task})")
        try:
            raw, model_used = self._ollama_generate_with_fallback(
                prompt, max_tokens=self.config.max_local_summary_tokens * 2, task=task
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Ollama unavailable ({e}); using deterministic optimizer result")
            det_payload.processing_time_ms = (time.time() - start) * 1000
            return det_payload

        payload = self._parse_response(raw, text, model_used)
        payload.original_token_count = det.original_tokens
        compact = payload.to_compact_prompt()
        lost = missing_facts(text, compact)
        if lost or payload.compressed_token_count >= det.optimized_tokens:
            logger.info(
                f"LLM extraction rejected ({payload.compressed_token_count} tokens, "
                f"{len(lost)} facts lost e.g. {lost[:3]}); keeping deterministic result"
            )
            payload = det_payload

        payload.processing_time_ms = (time.time() - start) * 1000
        logger.info(
            f"Compressed: {payload.original_token_count} -> {payload.compressed_token_count} "
            f"tokens ({payload.compression_ratio:.1f}x, {payload.processor_model})"
        )
        return payload

    def compress_conversation(self, messages: list) -> TextPayload:
        """
        Compress a multi-turn conversation history.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
                      e.g., [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            TextPayload with compressed conversation context.
        """
        # Format messages into text
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")

        full_text = "\n".join(text_parts)
        return self.compress(full_text, mode="summarize")

    def compress_documents(self, documents: list, question: str) -> TextPayload:
        """
        Compress RAG-retrieved documents into key facts.

        Args:
            documents: List of document strings.
            question:  The user's question these docs should answer.

        Returns:
            TextPayload with condensed document facts.
        """
        full_text = "\n\n".join(f"[Doc {i + 1}]: {doc}" for i, doc in enumerate(documents))
        # Question-aware selection is where extractive compression shines: spend the
        # budget on the most relevant sentences across all documents.
        opt = self._optimizer()
        det = opt.optimize_documents(
            documents,
            question=question,
            max_tokens=self.config.target_token_budget,
            target_ratio=0.0 if opt.level == "lossless" else 0.5,
        )
        return self.compress(full_text, mode="condense", question=question, _optimized=det)

    def compress_prompt(self, text: str) -> dict:
        """
        Compress a verbose prompt into a concise one WITHOUT generating an answer.

        This is different from compress() — it returns the compressed prompt text
        directly, not a structured payload. Use this when you want to save tokens
        by making a prompt shorter before sending it to a cloud LLM.

        Args:
            text: The verbose prompt to compress.

        Returns:
            Dictionary with compressed prompt and stats.
        """
        start = time.time()
        det = self._optimizer().optimize(text, max_tokens=self.config.target_token_budget)
        original_tokens = det.original_tokens
        compressed, method = det.text, f"optimizer/{det.level}"

        if self.config.compression_strategy != "deterministic" and det.optimized_tokens >= 60:
            prompt = self.COMPRESS_ONLY_PROMPT.replace("{user_text}", det.text)
            try:
                resp, model_used = self._ollama_generate_with_fallback(
                    prompt, max_tokens=500, task="text_fast"
                )
                candidate = resp.strip()
                lost = missing_facts(text, candidate)
                if candidate and not lost and count_tokens(candidate) < det.optimized_tokens:
                    compressed, method = candidate, f"llm/{model_used}"
                else:
                    logger.info(
                        f"LLM rewrite rejected ({len(lost)} facts lost e.g. {lost[:3]}); "
                        "keeping deterministic result"
                    )
            except (ConnectionError, TimeoutError):
                logger.info("Ollama unavailable for prompt compression; using optimizer result")

        compressed_tokens = count_tokens(compressed)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Prompt compressed: {original_tokens} -> {compressed_tokens} tokens "
            f"({original_tokens / max(compressed_tokens, 1):.1f}x, {method}, {elapsed_ms:.0f}ms)"
        )

        return {
            "original_prompt": text,
            "compressed_prompt": compressed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": original_tokens - compressed_tokens,
            "compression_ratio": f"{original_tokens / max(compressed_tokens, 1):.1f}x",
            "processing_time_ms": round(elapsed_ms, 1),
            "input_type": "compress_only",
            "method": method,
        }
