"""
LatentGatePipeline — Main orchestrator (v0.3.0 — Speed Optimized)

Speed Optimizations:
  1. Shared FastClient across all components (connection pooling)
  2. Model preloading on init (eliminates cold start)
  3. Parallel image+text processing in universal mode
  4. 3-tier JSON parsing (avoids slow LLM fallback)
  5. keep_alive keeps models in GPU memory between calls
"""

import logging
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.text_processor import TextProcessor, TextPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import RemoteDecoder, create_decoder
from latent_gate.fast_client import FastClient


logger = logging.getLogger("latent_gate.pipeline")


class LatentGatePipeline:
    """
    Main pipeline orchestrator inspired by VL-JEPA architecture.

    v0.3.0 Speed Optimizations:
      - Shared FastClient (single connection pool for everything)
      - Model preloading (warm GPU on init)
      - Parallel processing (image + text at the same time)
      - 3-tier JSON parsing (fast → medium → slow fallback)

    Supports:
      1. query()              — Image + question
      2. query_text()         — Text prompt compression
      3. query_conversation() — Conversation history compression
      4. query_documents()    — RAG document compression
      5. query_universal()    — Auto-detect (image, text, or both)
    """

    def __init__(self, config: Optional[PipelineConfig] = None, preload: bool = True):
        self.config = config or PipelineConfig()
        self._setup_logging()

        # Validate
        for w in self.config.validate():
            logger.warning(f"Config: {w}")

        # SHARED FastClient — single connection pool for everything
        self.client = FastClient(self.config)

        # All components share the same client
        self.local_processor = LocalProcessor(self.config, client=self.client)
        self.text_processor = TextProcessor(self.config)
        self.text_processor.client = self.client  # Share client
        self.selective_decoder = SelectiveDecoder(
            similarity_threshold=self.config.similarity_threshold
        )
        self.remote_decoder: RemoteDecoder = create_decoder(self.config, client=self.client)

        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Preload models into GPU memory (eliminates cold start)
        if preload:
            try:
                self.client.preload_models()
                logger.info("Models preloaded into GPU memory")
            except Exception as e:
                logger.warning(f"Preload skipped: {e}")

        logger.info(
            f"Pipeline ready: vision={self.config.vision_model}, "
            f"remote={self.config.remote_provider}/{self.config.remote_model}"
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    # ================================================================
    # MODE 1: Image Query
    # ================================================================

    def query(self, image_path: str, question: str) -> dict:
        """Process an image and answer a question about it."""
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Vision Processing")
        start = time.time()

        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        logger.info(f"Local: ~{payload.estimated_token_count} tokens, {local_ms:.0f}ms")

        # Selective decoding check
        if self.config.selective_decoding:
            if not self.selective_decoder.should_decode(payload):
                logger.info("Selective: reusing previous")
                return self._build_result(
                    self.selective_decoder.previous_response,
                    compact_input, payload.estimated_token_count,
                    was_cached=True, payload_dict=payload.to_dict(),
                    input_type="image", local_time_ms=local_ms,
                )

        logger.info("STAGE 2: Remote Decoding")
        remote_start = time.time()
        answer = self.remote_decoder.decode(compact_input, question)
        remote_ms = (time.time() - remote_start) * 1000

        self.selective_decoder.update(payload, answer)

        return self._build_result(
            answer, compact_input, payload.estimated_token_count,
            was_cached=False, payload_dict=payload.to_dict(),
            input_type="image", local_time_ms=local_ms, remote_time_ms=remote_ms,
        )

    # ================================================================
    # MODE 2: Text Query
    # ================================================================

    def query_text(self, text: str, question: str = "", mode: str = "auto") -> dict:
        """Compress a text prompt locally, then send to cloud LLM."""
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Text Compression")
        start = time.time()

        text_payload = self.text_processor.compress(text, mode=mode, question=question)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        logger.info(
            f"Compressed: {text_payload.original_token_count} → "
            f"{text_payload.compressed_token_count} tokens ({local_ms:.0f}ms)"
        )

        logger.info("STAGE 2: Remote Decoding")
        remote_start = time.time()
        final_q = question or text_payload.intent or "Process and respond."
        answer = self.remote_decoder.decode(compact_input, final_q)
        remote_ms = (time.time() - remote_start) * 1000

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="text",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms, remote_time_ms=remote_ms,
        )

    # ================================================================
    # MODE 2b: Conversation Compression
    # ================================================================

    def query_conversation(self, messages: list, new_question: str) -> dict:
        """Compress conversation history + ask a new question."""
        start = time.time()
        text_payload = self.text_processor.compress_conversation(messages)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        remote_start = time.time()
        answer = self.remote_decoder.decode(compact_input, new_question)
        remote_ms = (time.time() - remote_start) * 1000

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="conversation",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms, remote_time_ms=remote_ms,
        )

    # ================================================================
    # MODE 2c: RAG Document Compression
    # ================================================================

    def query_documents(self, documents: list, question: str) -> dict:
        """Compress RAG-retrieved documents + answer a question."""
        start = time.time()
        text_payload = self.text_processor.compress_documents(documents, question)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        remote_start = time.time()
        answer = self.remote_decoder.decode(compact_input, question)
        remote_ms = (time.time() - remote_start) * 1000

        return self._build_result(
            answer, compact_input, text_payload.compressed_token_count,
            was_cached=False, payload_dict=text_payload.to_dict(),
            input_type="documents",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms, remote_time_ms=remote_ms,
        )

    # ================================================================
    # MODE 3: Universal (auto-detect) — WITH PARALLEL PROCESSING
    # ================================================================

    def query_universal(self, text: str = "", image: str = "", question: str = "") -> dict:
        """
        Universal entry point — auto-detects input type.
        When both image + text provided, processes them IN PARALLEL.
        """
        has_image = bool(image)
        has_text = bool(text)

        if has_image and has_text:
            # PARALLEL: Process image and text at the same time
            logger.info("Universal: Image + Text (PARALLEL processing)")
            start = time.time()

            future_image = self._executor.submit(self.local_processor.process, image)
            future_text = self._executor.submit(self.text_processor.compress, text)

            image_payload = future_image.result()
            text_payload = future_text.result()
            local_ms = (time.time() - start) * 1000

            image_compact = image_payload.to_compact_prompt()
            text_compact = text_payload.to_compact_prompt()
            combined = f"[VISUAL]: {image_compact} | [TEXT]: {text_compact}"

            total_compressed = image_payload.estimated_token_count + text_payload.compressed_token_count
            total_original = 1200 + text_payload.original_token_count

            final_q = question or text_payload.intent or "Analyze the image and text together."

            remote_start = time.time()
            answer = self.remote_decoder.decode(combined, final_q)
            remote_ms = (time.time() - remote_start) * 1000

            return self._build_result(
                answer, combined, total_compressed,
                was_cached=False,
                payload_dict={"image": image_payload.to_dict(), "text": text_payload.to_dict()},
                input_type="image+text",
                original_tokens=total_original,
                compression_ratio=total_original / max(total_compressed, 1),
                local_time_ms=local_ms, remote_time_ms=remote_ms,
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
            logger.info(f"Frame {i + 1}/{len(image_paths)}: {path}")
            results.append(self.query(path, question))
        logger.info(f"Batch: {self.selective_decoder.stats}")
        return results

    def reset_selective_decoder(self):
        self.selective_decoder.reset()

    def close(self):
        """Clean up resources."""
        self.client.close()
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ================================================================
    # Internal
    # ================================================================

    def _build_result(self, answer, compact_prompt, tokens_estimated,
                      was_cached, payload_dict, input_type="image",
                      original_tokens=0, compression_ratio=0.0,
                      local_time_ms=0.0, remote_time_ms=0.0) -> dict:
        result = {
            "answer": answer,
            "compact_prompt": compact_prompt,
            "tokens_estimated": tokens_estimated,
            "was_cached": was_cached,
            "payload": payload_dict,
            "input_type": input_type,
            "selective_decoding_stats": self.selective_decoder.stats,
            "timing": {
                "local_ms": round(local_time_ms, 1),
                "remote_ms": round(remote_time_ms, 1),
                "total_ms": round(local_time_ms + remote_time_ms, 1),
            },
        }
        if original_tokens > 0:
            result["original_tokens"] = original_tokens
            result["compression_ratio"] = f"{compression_ratio:.1f}x"
            result["tokens_saved"] = original_tokens - tokens_estimated
        return result
