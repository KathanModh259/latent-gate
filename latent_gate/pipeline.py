"""
LatentGatePipeline — Main orchestrator.

Connects all components:
  Image → [LOCAL: X-Encoder → Predictor → SemanticPayload]
        → [Selective Decoding Check]
        → [REMOTE: Y-Decoder (Cloud LLM)] → Final Answer

  Text  → [LOCAL: TextProcessor → TextPayload]
        → [REMOTE: Y-Decoder (Cloud LLM)] → Final Answer

  Both  → [LOCAL: Image + Text compressed together]
        → [REMOTE: Y-Decoder] → Final Answer
"""

import logging
from typing import Optional, Union

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.text_processor import TextProcessor, TextPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import RemoteDecoder, create_decoder


logger = logging.getLogger("latent_gate.pipeline")


class LatentGatePipeline:
    """
    Main pipeline orchestrator inspired by VL-JEPA architecture.

    Supports THREE modes:
      1. query()          — Image + question  (original)
      2. query_text()     — Text prompt compression + cloud reasoning
      3. query_universal() — Any combo of image + text, auto-detected

    Args:
        config: PipelineConfig with all settings.

    Example:
        >>> from latent_gate import LatentGatePipeline, PipelineConfig
        >>> pipeline = LatentGatePipeline(PipelineConfig(remote_provider="ollama"))
        >>>
        >>> # Image mode
        >>> result = pipeline.query("photo.jpg", "What's in this?")
        >>>
        >>> # Text mode
        >>> result = pipeline.query_text("Write a 500-word essay about AI safety...")
        >>>
        >>> # Universal mode (auto-detect)
        >>> result = pipeline.query_universal(text="Long prompt...", image="photo.jpg")
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._setup_logging()

        # Validate config
        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

        # Initialize components
        self.local_processor = LocalProcessor(self.config)
        self.text_processor = TextProcessor(self.config)
        self.selective_decoder = SelectiveDecoder(
            similarity_threshold=self.config.similarity_threshold
        )
        self.remote_decoder: RemoteDecoder = create_decoder(self.config)

        logger.info(
            f"Pipeline initialized: "
            f"vision={self.config.vision_model}, "
            f"predictor={self.config.predictor_model}, "
            f"remote={self.config.remote_provider}/{self.config.remote_model}"
        )

    def _setup_logging(self):
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    # ================================================================
    # MODE 1: Image Query (original)
    # ================================================================

    def query(self, image_path: str, question: str) -> dict:
        """
        Process an image and answer a question about it.

        Args:
            image_path: Path to the image file.
            question: Question about the image.

        Returns:
            Dict with answer, payload, compact_prompt, tokens_estimated, etc.
        """
        # ---- STAGE 1: LOCAL (FREE) ----
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Vision Processing (X-Encoder + Predictor)")

        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()

        logger.info(f"Compact payload: ~{payload.estimated_token_count} tokens")

        # ---- SELECTIVE DECODING CHECK ----
        if self.config.selective_decoding:
            if not self.selective_decoder.should_decode(payload):
                logger.info("Selective Decoding: reusing previous response")
                return self._build_result(
                    self.selective_decoder.previous_response,
                    compact_input, payload.estimated_token_count,
                    was_cached=True, payload_dict=payload.to_dict(),
                    input_type="image",
                )

        # ---- STAGE 2: REMOTE (PAID — minimal tokens) ----
        logger.info("STAGE 2: Remote Decoding (Y-Decoder)")
        answer = self.remote_decoder.decode(compact_input, question)
        self.selective_decoder.update(payload, answer)

        return self._build_result(
            answer, compact_input, payload.estimated_token_count,
            was_cached=False, payload_dict=payload.to_dict(),
            input_type="image",
        )

    # ================================================================
    # MODE 2: Text Query (NEW)
    # ================================================================

    def query_text(
        self,
        text: str,
        question: str = "",
        mode: str = "auto",
    ) -> dict:
        """
        Compress a text prompt locally, then send to cloud LLM.

        Instead of sending 500+ tokens to the API, we compress locally
        to ~100-150 tokens and send only that.

        Args:
            text:     The full user prompt / context / document.
            question: Optional separate question (if text is context/docs).
                      If empty, the intent is extracted from text itself.
            mode:     "auto" | "compress" | "summarize" | "condense" | "code"

        Returns:
            Dict with answer, payload, tokens saved, compression stats, etc.
        """
        # ---- STAGE 1: LOCAL TEXT COMPRESSION (FREE) ----
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Text Compression")

        text_payload = self.text_processor.compress(text, mode=mode, question=question)
        compact_input = text_payload.to_compact_prompt()

        logger.info(
            f"Compressed: {text_payload.original_token_count} → "
            f"{text_payload.compressed_token_count} tokens "
            f"({text_payload.compression_ratio:.1f}x)"
        )

        # ---- STAGE 2: REMOTE (PAID — minimal tokens) ----
        logger.info("STAGE 2: Remote Decoding (Y-Decoder)")

        # Build the query for the cloud LLM
        if question:
            final_question = question
        else:
            final_question = text_payload.intent or "Process the above information and respond."

        answer = self.remote_decoder.decode(compact_input, final_question)

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="text",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
        )

    # ================================================================
    # MODE 2b: Conversation Compression (NEW)
    # ================================================================

    def query_conversation(
        self,
        messages: list,
        new_question: str,
    ) -> dict:
        """
        Compress conversation history + ask a new question.

        Instead of sending the full chat history (which grows every turn),
        compress it locally and send the summary + new question.

        Args:
            messages:     List of {"role": "user/assistant", "content": "..."} dicts.
            new_question: The new question to ask.

        Returns:
            Dict with answer and compression stats.
        """
        logger.info("=" * 50)
        logger.info("STAGE 1: Compressing Conversation History")

        text_payload = self.text_processor.compress_conversation(messages)
        compact_input = text_payload.to_compact_prompt()

        logger.info(
            f"Conversation compressed: {text_payload.original_token_count} → "
            f"{text_payload.compressed_token_count} tokens"
        )

        # ---- STAGE 2: REMOTE ----
        logger.info("STAGE 2: Remote Decoding with new question")
        answer = self.remote_decoder.decode(compact_input, new_question)

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="conversation",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
        )

    # ================================================================
    # MODE 2c: RAG Document Compression (NEW)
    # ================================================================

    def query_documents(
        self,
        documents: list,
        question: str,
    ) -> dict:
        """
        Compress RAG-retrieved documents + answer a question.

        Instead of stuffing all retrieved chunks into the prompt,
        extract only relevant facts locally, then send to cloud.

        Args:
            documents: List of retrieved document strings.
            question:  The user's question.

        Returns:
            Dict with answer and compression stats.
        """
        logger.info("=" * 50)
        logger.info(f"STAGE 1: Condensing {len(documents)} documents locally")

        text_payload = self.text_processor.compress_documents(documents, question)
        compact_input = text_payload.to_compact_prompt()

        logger.info(
            f"Documents condensed: {text_payload.original_token_count} → "
            f"{text_payload.compressed_token_count} tokens"
        )

        # ---- STAGE 2: REMOTE ----
        logger.info("STAGE 2: Remote Decoding")
        answer = self.remote_decoder.decode(compact_input, question)

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="documents",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
        )

    # ================================================================
    # MODE 3: Universal Query (auto-detect input type)
    # ================================================================

    def query_universal(
        self,
        text: str = "",
        image: str = "",
        question: str = "",
    ) -> dict:
        """
        Universal entry point — auto-detects input type and routes accordingly.

        Args:
            text:     Text prompt / context (optional).
            image:    Path to image file (optional).
            question: Explicit question (optional).

        Returns:
            Dict with answer and processing stats.
        """
        has_image = bool(image)
        has_text = bool(text)

        if has_image and has_text:
            # Both: compress text locally + extract image locally, combine
            logger.info("Universal mode: Image + Text")
            image_payload = self.local_processor.process(image)
            text_payload = self.text_processor.compress(text)

            image_compact = image_payload.to_compact_prompt()
            text_compact = text_payload.to_compact_prompt()

            combined = f"[VISUAL]: {image_compact} | [TEXT]: {text_compact}"
            final_question = question or text_payload.intent or "Analyze the image and text together."

            total_compressed = image_payload.estimated_token_count + text_payload.compressed_token_count
            total_original = 1200 + text_payload.original_token_count  # ~1200 for image

            answer = self.remote_decoder.decode(combined, final_question)

            return self._build_result(
                answer, combined, total_compressed,
                was_cached=False,
                payload_dict={
                    "image": image_payload.to_dict(),
                    "text": text_payload.to_dict(),
                },
                input_type="image+text",
                original_tokens=total_original,
                compression_ratio=total_original / max(total_compressed, 1),
            )

        elif has_image:
            return self.query(image, question or "Describe this image.")

        elif has_text:
            return self.query_text(text, question)

        else:
            raise ValueError("Provide at least 'text' or 'image' input.")

    # ================================================================
    # Batch Processing
    # ================================================================

    def query_batch(self, image_paths: list, question: str) -> list:
        """Process multiple images with selective decoding."""
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing frame {i + 1}/{len(image_paths)}: {path}")
            result = self.query(path, question)
            results.append(result)
        logger.info(f"Batch complete — {self.selective_decoder.stats}")
        return results

    def reset_selective_decoder(self):
        """Reset selective decoder state."""
        self.selective_decoder.reset()
        logger.info("Selective decoder reset")

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _build_result(
        self,
        answer: str,
        compact_prompt: str,
        tokens_estimated: int,
        was_cached: bool,
        payload_dict: dict,
        input_type: str = "image",
        original_tokens: int = 0,
        compression_ratio: float = 0.0,
    ) -> dict:
        """Build standardized result dictionary."""
        result = {
            "answer": answer,
            "compact_prompt": compact_prompt,
            "tokens_estimated": tokens_estimated,
            "was_cached": was_cached,
            "payload": payload_dict,
            "input_type": input_type,
            "selective_decoding_stats": self.selective_decoder.stats,
        }
        if original_tokens > 0:
            result["original_tokens"] = original_tokens
            result["compression_ratio"] = f"{compression_ratio:.1f}x"
            result["tokens_saved"] = original_tokens - tokens_estimated
        return result
