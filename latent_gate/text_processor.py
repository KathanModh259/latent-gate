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
import time
import logging
from dataclasses import dataclass, field, asdict

from latent_gate.config import PipelineConfig

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
        parts = []

        if self.question_type:
            parts.append(f"[Type: {self.question_type}]")
        if self.intent:
            parts.append(f"Intent: {self.intent}")
        if self.key_entities:
            parts.append(f"Entities: {', '.join(self.key_entities[:10])}")
        if self.constraints:
            parts.append(f"Constraints: {'; '.join(self.constraints[:5])}")
        if self.context_summary:
            parts.append(f"Context: {self.context_summary}")
        if self.output_format:
            parts.append(f"Output format: {self.output_format}")
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.data_points:
            parts.append(f"Data: {', '.join(str(d) for d in self.data_points[:10])}")
        if self.code_snippets:
            # Code is preserved as-is (can't compress without losing meaning)
            for i, snippet in enumerate(self.code_snippets[:3]):
                parts.append(f"Code[{i}]: {snippet[:300]}")

        compact = " | ".join(parts)
        self.compressed_token_count = len(compact.split()) + 10
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

    COMPRESS_ONLY_PROMPT = """Rewrite the user's prompt to be as concise as possible while preserving the EXACT same intent, requirements, and context.

CRITICAL RULES:
- Output ONLY the rewritten prompt text. Do NOT answer the prompt. Do NOT generate a response.
- Think of this as editing, not generating. You are shortening text, not creating content.
- Keep all specific names, numbers, values, and technical terms
- Remove filler words, greetings, unnecessary explanations, pleasantries
- Remove redundancy and repetition
- Keep code blocks exactly as-is
- The result must be a shorter version that produces the same answer when sent to an AI

ORIGINAL PROMPT:
{user_text}

OUTPUT (shorter version of the same prompt, nothing else):"""

    def __init__(self, config: PipelineConfig, client=None):
        self.config = config
        self.client = client

    def _ollama_generate(self, prompt: str) -> str:
        """Call Ollama for text compression. Uses shared FastClient if available."""
        if self.client:
            return self.client.ollama_generate(
                model=self.config.predictor_model,
                prompt=prompt,
                max_tokens=self.config.max_local_summary_tokens * 4,
            )
        # Fallback: direct requests if no client shared
        import requests as _requests
        url = f"{self.config.ollama_base_url}/api/generate"
        payload = {
            "model": self.config.predictor_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_local_summary_tokens * 4,
            },
        }
        try:
            resp = _requests.post(url, json=payload, timeout=self.config.request_timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except _requests.exceptions.ConnectionError:
            raise ConnectionError("Cannot connect to Ollama. Make sure it's running: ollama serve")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (~0.75 tokens per word for English)."""
        return int(len(text.split()) * 1.33)

    def _detect_mode(self, text: str, mode: str = "auto") -> str:
        """Auto-detect the best compression mode based on content."""
        if mode != "auto":
            return mode

        # Simple heuristics
        code_indicators = ["```", "def ", "function ", "class ", "import ", "const ", "var "]
        if any(indicator in text for indicator in code_indicators):
            return "code"

        # Multi-turn conversation pattern
        turn_indicators = ["User:", "Assistant:", "Human:", "AI:", "Q:", "A:"]
        if sum(1 for t in turn_indicators if t in text) >= 2:
            return "summarize"

        # Long text with question
        if len(text.split()) > 200:
            return "condense"

        return "compress"

    def _parse_response(self, raw: str, original_text: str) -> TextPayload:
        """Parse LLM output into TextPayload."""
        payload = TextPayload()
        payload.original_token_count = self._estimate_tokens(original_text)

        try:
            # Clean markdown fences
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                cleaned = "\n".join(lines)

            data = json.loads(cleaned)

            payload.intent = str(data.get("intent", ""))
            payload.key_entities = list(data.get("key_entities", []))
            payload.constraints = list(data.get("constraints", []))
            payload.context_summary = str(data.get("context_summary", ""))
            payload.question_type = str(data.get("question_type", ""))
            payload.output_format = str(data.get("output_format", ""))
            payload.tone = str(data.get("tone", ""))
            payload.code_snippets = list(data.get("code_snippets", []))
            payload.data_points = list(data.get("data_points", []))

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"JSON parse failed ({e}), using raw as context_summary")
            payload.intent = "Process the following request"
            payload.context_summary = raw[:500]

        payload.processor_model = self.config.predictor_model
        return payload

    # ----------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------

    def compress(
        self,
        text: str,
        mode: str = "auto",
        question: str = "",
    ) -> TextPayload:
        """
        Compress a text prompt locally.

        Args:
            text:     The full user prompt / text to compress.
            mode:     "auto" | "compress" | "summarize" | "condense" | "code"
            question: For 'condense' mode — the question being answered.

        Returns:
            TextPayload with compressed representation.
        """
        start = time.time()

        # Detect mode
        mode = self._detect_mode(text, mode)
        logger.info(f"Text compression mode: {mode}")

        # Short-circuit: if text is already short, don't compress
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens < 100:
            logger.info(f"Text is short ({estimated_tokens} tokens), skipping compression")
            payload = TextPayload(
                intent=text,
                original_token_count=estimated_tokens,
                compressed_token_count=estimated_tokens,
                compression_ratio=1.0,
                processing_time_ms=(time.time() - start) * 1000,
            )
            return payload

        # Select prompt template
        prompt_map = {
            "compress": self.COMPRESS_PROMPT.format(user_text=text[:3000]),
            "summarize": self.SUMMARIZE_PROMPT.format(user_text=text[:3000]),
            "condense": self.CONDENSE_PROMPT.format(
                user_text=text[:3000], question=question or "Answer the query"
            ),
            "code": self.CODE_PROMPT.format(user_text=text[:3000]),
        }
        prompt = prompt_map.get(mode, prompt_map["compress"])

        # Run local LLM
        logger.info(
            f"Compressing {estimated_tokens} tokens locally with {self.config.predictor_model}"
        )
        raw = self._ollama_generate(prompt)

        # Parse into payload
        payload = self._parse_response(raw, text)
        payload.processing_time_ms = (time.time() - start) * 1000

        # Calculate compression stats
        payload.to_compact_prompt()  # Populates compressed_token_count
        logger.info(
            f"Compressed: {payload.original_token_count} → {payload.compressed_token_count} tokens "
            f"({payload.compression_ratio:.1f}x reduction)"
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
        full_text = "\n\n---\n\n".join(f"[Doc {i+1}]: {doc}" for i, doc in enumerate(documents))
        return self.compress(full_text, mode="condense", question=question)

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
        original_tokens = self._estimate_tokens(text)

        prompt = self.COMPRESS_ONLY_PROMPT.format(user_text=text[:4000])
        compressed = self._ollama_generate(prompt).strip()

        compressed_tokens = self._estimate_tokens(compressed)
        elapsed_ms = (time.time() - start) * 1000

        logger.info(
            f"Prompt compressed: {original_tokens} → {compressed_tokens} tokens "
            f"({original_tokens / max(compressed_tokens, 1):.1f}x reduction, {elapsed_ms:.0f}ms)"
        )

        return {
            "original_prompt": text,
            "compressed_prompt": compressed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "tokens_saved": original_tokens - compressed_tokens,
            "compression_ratio": f"{original_tokens / max(compressed_tokens, 1):.1f}x",
            "processing_time_ms": round(elapsed_ms, 1),
        }
