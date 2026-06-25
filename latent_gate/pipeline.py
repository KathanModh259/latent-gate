"""
LatentGatePipeline — Main orchestrator (v1.0.0)

Features:
  1. Shared FastClient across all components (connection pooling)
  2. Model preloading on init (eliminates cold start)
  3. Parallel image+text processing in universal mode
  4. 3-tier JSON parsing (avoids slow LLM fallback)
  5. keep_alive keeps models in GPU memory between calls
  6. Offline-first mode (local Ollama answering when no API key)
  7. Adaptive compression (dynamically adjust based on complexity)
  8. Semantic deduplication (skip similar queries in batches)
"""

import logging
import time
import hashlib
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from latent_gate.config import PipelineConfig
from latent_gate.text_processor import TextProcessor
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import RemoteDecoder, create_decoder
from latent_gate.fast_client import FastClient

logger = logging.getLogger("latent_gate.pipeline")

# Bounded cache to prevent memory leaks in long-running processes
_MAX_DEDUP_CACHE_SIZE = 1000


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
            similarity_threshold=self.config.similarity_threshold,
            use_embeddings=self.config.use_embeddings,
        )
        self.remote_decoder: RemoteDecoder = create_decoder(self.config, client=self.client)

        # Offline-first mode: use local Ollama for answering when enabled
        self._offline_decoder = None
        if self.config.offline_first:
            from latent_gate.remote_decoder import OllamaRemoteDecoder

            self._offline_decoder = OllamaRemoteDecoder(self.config, client=self.client)
            logger.info("Offline-first mode enabled — using local Ollama for answers")

        # Semantic deduplication cache for batch processing (bounded)
        self._query_cache: OrderedDict = OrderedDict()
        self._dedup_hits: int = 0

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
        level = getattr(logging, self.config.log_level, logging.INFO)
        lib_logger = logging.getLogger("latent_gate")
        if not lib_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            lib_logger.addHandler(handler)
        lib_logger.setLevel(level)

    # ================================================================
    # MODE 1: Image Query
    # ================================================================

    def query(self, image_path: str, question: str) -> dict:
        """Process an image and answer a question about it."""
        logger.info("=" * 50)

        # Semantic deduplication check (before expensive processing)
        cached = self._is_duplicate(image_path, question)
        if cached:
            return cached

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
                result = self._build_result(
                    self.selective_decoder.previous_response,
                    compact_input,
                    payload.estimated_token_count,
                    was_cached=True,
                    payload_dict=payload.to_dict(),
                    input_type="image",
                    local_time_ms=local_ms,
                )
                self._cache_result(image_path, question, result)
                return result

        logger.info("STAGE 2: Remote Decoding")
        remote_start = time.time()
        decoder = self._get_decoder()
        answer = decoder.decode(compact_input, question)
        remote_ms = (time.time() - remote_start) * 1000

        self.selective_decoder.update(payload, answer)

        result = self._build_result(
            answer,
            compact_input,
            payload.estimated_token_count,
            was_cached=False,
            payload_dict=payload.to_dict(),
            input_type="image",
            local_time_ms=local_ms,
            remote_time_ms=remote_ms,
        )
        self._cache_result(image_path, question, result)
        return result

    # ================================================================
    # MODE 2: Text Query
    # ================================================================

    def query_text(self, text: str, question: str = "", mode: str = "auto") -> dict:
        """Compress a text prompt locally, then send to cloud LLM."""
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Text Compression")
        start = time.time()

        # Semantic deduplication check
        cached = self._is_duplicate(text, question)
        if cached:
            return cached

        # Adaptive compression: estimate complexity and adjust
        adaptive_mode = mode
        if self.config.adaptive_compression:
            complexity = self._estimate_complexity(text)
            max_tokens = self._adaptive_max_tokens(complexity)
            logger.info(
                f"Adaptive compression: complexity={complexity:.2f}, max_tokens={max_tokens}"
            )
            if mode == "auto":
                if complexity < 0.3:
                    adaptive_mode = "compress"
                elif complexity < 0.6:
                    adaptive_mode = "summarize"
                else:
                    adaptive_mode = "condense"

        text_payload = self.text_processor.compress(text, mode=adaptive_mode, question=question)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        logger.info(
            f"Compressed: {text_payload.original_token_count} → "
            f"{text_payload.compressed_token_count} tokens ({local_ms:.0f}ms)"
        )

        logger.info("STAGE 2: Remote Decoding")
        remote_start = time.time()
        final_q = question or text_payload.intent or "Process and respond."
        decoder = self._get_decoder()
        answer = decoder.decode(compact_input, final_q)
        remote_ms = (time.time() - remote_start) * 1000

        result = self._build_result(
            answer,
            compact_input,
            text_payload.compressed_token_count,
            was_cached=False,
            payload_dict=text_payload.to_dict(),
            input_type="text",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms,
            remote_time_ms=remote_ms,
        )
        self._cache_result(text, question, result)
        return result

    # ================================================================
    # MODE 2b: Conversation Compression
    # ================================================================

    def query_conversation(self, messages: list, new_question: str) -> dict:
        """Compress conversation history + ask a new question."""
        start = time.time()

        # Semantic deduplication check
        msg_text = str(messages)[:500]
        cached = self._is_duplicate(msg_text, new_question)
        if cached:
            return cached

        text_payload = self.text_processor.compress_conversation(messages)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        remote_start = time.time()
        decoder = self._get_decoder()
        answer = decoder.decode(compact_input, new_question)
        remote_ms = (time.time() - remote_start) * 1000

        result = self._build_result(
            answer,
            compact_input,
            text_payload.compressed_token_count,
            was_cached=False,
            payload_dict=text_payload.to_dict(),
            input_type="conversation",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms,
            remote_time_ms=remote_ms,
        )
        self._cache_result(msg_text, new_question, result)
        return result

    # ================================================================
    # MODE 2c: RAG Document Compression
    # ================================================================

    def query_documents(self, documents: list, question: str) -> dict:
        """Compress RAG-retrieved documents + answer a question."""
        start = time.time()

        # Semantic deduplication check
        doc_hash = "|".join(d[:100] for d in documents[:5])
        cached = self._is_duplicate(doc_hash, question)
        if cached:
            return cached

        text_payload = self.text_processor.compress_documents(documents, question)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        remote_start = time.time()
        decoder = self._get_decoder()
        answer = decoder.decode(compact_input, question)
        remote_ms = (time.time() - remote_start) * 1000

        result = self._build_result(
            answer,
            compact_input,
            text_payload.compressed_token_count,
            was_cached=False,
            payload_dict=text_payload.to_dict(),
            input_type="documents",
            original_tokens=text_payload.original_token_count,
            compression_ratio=text_payload.compression_ratio,
            local_time_ms=local_ms,
            remote_time_ms=remote_ms,
        )
        self._cache_result(doc_hash, question, result)
        return result

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
            future_text = self._executor.submit(
                self.text_processor.compress, text, "auto", question
            )

            image_payload = future_image.result()
            text_payload = future_text.result()
            local_ms = (time.time() - start) * 1000

            image_compact = image_payload.to_compact_prompt()
            text_compact = text_payload.to_compact_prompt()
            combined = f"[VISUAL]: {image_compact} | [TEXT]: {text_compact}"

            total_compressed = (
                image_payload.estimated_token_count + text_payload.compressed_token_count
            )
            total_original = image_payload.estimated_token_count + text_payload.original_token_count

            final_q = question or text_payload.intent or "Analyze the image and text together."

            remote_start = time.time()
            decoder = self._get_decoder()
            answer = decoder.decode(combined, final_q)
            remote_ms = (time.time() - remote_start) * 1000

            return self._build_result(
                answer,
                combined,
                total_compressed,
                was_cached=False,
                payload_dict={"image": image_payload.to_dict(), "text": text_payload.to_dict()},
                input_type="image+text",
                original_tokens=total_original,
                compression_ratio=total_original / max(total_compressed, 1),
                local_time_ms=local_ms,
                remote_time_ms=remote_ms,
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

    def query_batch(
        self, image_paths: list, question: str, parallel: bool = False, max_workers: int = 3
    ) -> list:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths.
            question: Question to ask about each image.
            parallel: If True, process images in parallel (no selective decoding).
                     If False, process sequentially with selective decoding.
            max_workers: Maximum parallel workers (only used when parallel=True).

        Returns:
            List of result dictionaries.
        """
        if not parallel:
            # Sequential processing with selective decoding
            results = []
            for i, path in enumerate(image_paths):
                logger.info(f"Frame {i + 1}/{len(image_paths)}: {path}")
                results.append(self.query(path, question))
            logger.info(f"Batch: {self.selective_decoder.stats}")
            return results

        # Parallel processing without selective decoding
        logger.info(f"Parallel batch: {len(image_paths)} images with {max_workers} workers")

        def _process_single(path: str) -> dict:
            # Create a temporary pipeline for thread safety
            from latent_gate.fast_client import FastClient

            client = FastClient(self.config)
            local_processor = LocalProcessor(self.config, client=client)
            remote_decoder = create_decoder(self.config, client=client)

            try:
                start = time.time()
                payload = local_processor.process(path)
                compact_input = payload.to_compact_prompt()
                local_ms = (time.time() - start) * 1000

                remote_start = time.time()
                answer = remote_decoder.decode(compact_input, question)
                remote_ms = (time.time() - remote_start) * 1000

                return {
                    "answer": answer,
                    "compact_prompt": compact_input,
                    "tokens_estimated": payload.estimated_token_count,
                    "was_cached": False,
                    "payload": payload.to_dict(),
                    "input_type": "image",
                    "timing": {
                        "local_ms": round(local_ms, 1),
                        "remote_ms": round(remote_ms, 1),
                        "total_ms": round(local_ms + remote_ms, 1),
                    },
                }
            finally:
                client.close()

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single, path): path for path in image_paths}
            results = []

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results.append(
                        {
                            "answer": f"Error: {str(e)}",
                            "compact_prompt": "",
                            "tokens_estimated": 0,
                            "was_cached": False,
                            "payload": {},
                            "input_type": "image",
                            "timing": {"local_ms": 0, "remote_ms": 0, "total_ms": 0},
                            "error": str(e),
                        }
                    )

        # Sort results by original order
        path_to_index = {path: i for i, path in enumerate(image_paths)}
        results.sort(
            key=lambda r: path_to_index.get(
                r.get("payload", {}).get("source_image", ""),
                len(image_paths) if image_paths else 0,
            )
        )

        logger.info(f"Parallel batch complete: {len(results)} images processed")
        return results

    def query_batch_texts(
        self, texts: list, question: str = "", parallel: bool = True, max_workers: int = 3
    ) -> list:
        """
        Process multiple texts in parallel.

        Args:
            texts: List of text strings.
            question: Question to ask about each text.
            parallel: If True, process in parallel.
            max_workers: Maximum parallel workers.

        Returns:
            List of result dictionaries.
        """
        if not parallel:
            results = []
            for i, text in enumerate(texts):
                logger.info(f"Text {i + 1}/{len(texts)}")
                results.append(self.query_text(text, question))
            return results

        logger.info(f"Parallel text batch: {len(texts)} texts with {max_workers} workers")

        def _process_single(text: str) -> dict:
            return self.query_text(text, question)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single, text): i for i, text in enumerate(texts)}
            results = [None] * len(texts)

            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to process text {index}: {e}")
                    results[index] = {
                        "answer": f"Error: {str(e)}",
                        "compact_prompt": "",
                        "tokens_estimated": 0,
                        "was_cached": False,
                        "payload": {},
                        "input_type": "text",
                        "timing": {"local_ms": 0, "remote_ms": 0, "total_ms": 0},
                        "error": str(e),
                    }

        logger.info(f"Parallel text batch complete: {len(results)} texts processed")
        return results

    # ================================================================
    # Streaming Methods
    # ================================================================

    def query_stream(self, image_path: str, question: str):
        """
        Process an image and stream the response.

        Args:
            image_path: Path to the image file.
            question: Question about the image.

        Yields:
            Response tokens as they arrive.
        """
        logger.info("Streaming image query")

        # Local processing
        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()

        # Stream remote decoding
        decoder = self._get_decoder()
        yield from decoder.decode_stream(compact_input, question)

    def query_text_stream(self, text: str, question: str = "", mode: str = "auto"):
        """
        Compress text and stream the response.

        Args:
            text: Text to compress and query.
            question: Specific question about the text.
            mode: Compression mode.

        Yields:
            Response tokens as they arrive.
        """
        logger.info("Streaming text query")

        # Local processing
        text_payload = self.text_processor.compress(text, mode=mode, question=question)
        compact_input = text_payload.to_compact_prompt()

        final_q = question or text_payload.intent or "Process and respond."

        # Stream remote decoding
        decoder = self._get_decoder()
        yield from decoder.decode_stream(compact_input, final_q)

    def query_universal_stream(self, text: str = "", image: str = "", question: str = ""):
        """
        Universal query with streaming response.

        Args:
            text: Text input.
            image: Image path.
            question: Question.

        Yields:
            Response tokens as they arrive.
        """
        has_image = bool(image)
        has_text = bool(text)

        if has_image and has_text:
            # Process both locally
            image_payload = self.local_processor.process(image)
            text_payload = self.text_processor.compress(text, "auto", question)

            image_compact = image_payload.to_compact_prompt()
            text_compact = text_payload.to_compact_prompt()
            combined = f"[VISUAL]: {image_compact} | [TEXT]: {text_compact}"

            final_q = question or text_payload.intent or "Analyze the image and text together."

            # Stream remote decoding
            decoder = self._get_decoder()
            yield from decoder.decode_stream(combined, final_q)

        elif has_image:
            yield from self.query_stream(image, question or "Describe this image.")
        elif has_text:
            yield from self.query_text_stream(text, question)
        else:
            raise ValueError("Provide at least 'text' or 'image' input.")

    def reset_selective_decoder(self):
        self.selective_decoder.reset()

    def compress_prompt(self, text: str) -> dict:
        """
        Compress a verbose prompt into a concise one WITHOUT generating an answer.

        Use this when you want to save tokens by making a prompt shorter
        before sending it to a cloud LLM.

        Args:
            text: The verbose prompt to compress.

        Returns:
            Dictionary with compressed prompt and stats.
        """
        return self.text_processor.compress_prompt(text)

    def estimate_cost(
        self, provider: str = "", model: str = "", input_tokens: int = 0, output_tokens: int = 0
    ) -> dict:
        """
        Estimate cost for a query without executing it.

        Returns:
            Dictionary with cost estimates and savings.
        """
        from latent_gate.cost_tracker import CostTracker

        provider = provider or self.config.remote_provider
        model = model or self.config.remote_model

        tracker = CostTracker()
        projection = tracker.get_cost_projection(
            daily_queries=1,
            avg_input_tokens=input_tokens,
            avg_output_tokens=output_tokens,
            provider=provider,
            model=model,
        )

        return {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": projection["without_compression"]["cost_per_query"],
            "compressed_cost": projection["with_compression"]["cost_per_query"],
            "savings_per_query": projection["savings"]["per_query"],
            "savings_percentage": projection["savings"]["percentage"],
        }

    def close(self):
        """Clean up resources."""
        self.client.close()
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ================================================================
    # Offline-First Mode
    # ================================================================

    def _decode_offline(self, compact_input: str, question: str) -> str:
        """
        Decode using local Ollama when offline-first is enabled.
        Falls back to remote decoder if no local decoder is available.
        """
        if self._offline_decoder:
            logger.info("Offline-first: using local Ollama for final answer")
            return self._offline_decoder.decode(compact_input, question)
        return self.remote_decoder.decode(compact_input, question)

    def _get_decoder(self):
        """Get the appropriate decoder (offline or remote)."""
        if self.config.offline_first and self._offline_decoder:
            return self._offline_decoder
        return self.remote_decoder

    # ================================================================
    # Adaptive Compression
    # ================================================================

    def _estimate_complexity(self, text: str, input_type: str = "text") -> float:
        """
        Estimate query complexity on a 0.0-1.0 scale.
        Higher complexity = more tokens needed for good response.
        """
        if input_type == "image":
            return 0.6

        words = len(text.split())
        sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
        avg_sentence_length = words / sentences

        complexity = 0.0
        complexity += min(words / 500, 0.4)
        complexity += min(avg_sentence_length / 30, 0.2)

        question_words = [
            "how",
            "why",
            "explain",
            "compare",
            "analyze",
            "evaluate",
            "design",
            "create",
        ]
        if any(w in text.lower() for w in question_words):
            complexity += 0.3

        if any(f in text for f in ["```", "def ", "class "]):
            complexity += 0.1

        return min(complexity, 1.0)

    def _adaptive_max_tokens(self, complexity: float) -> int:
        """Calculate max tokens based on query complexity."""
        base = 200
        return int(base + (complexity * 300))

    # ================================================================
    # Semantic Deduplication
    # ================================================================

    def _query_hash(self, text: str, question: str = "") -> str:
        """Generate a hash for deduplication."""
        content = f"{text[:500]}|{question[:200]}".lower().strip()
        return hashlib.md5(content.encode()).hexdigest()

    def _is_duplicate(self, text: str, question: str = "") -> Optional[dict]:
        """Check if a query has been seen before."""
        qhash = self._query_hash(text, question)
        if qhash in self._query_cache:
            self._dedup_hits += 1
            logger.info(f"Semantic dedup: reusing cached result for hash {qhash[:8]}...")
            return self._query_cache[qhash]
        return None

    def _cache_result(self, text: str, question: str, result: dict):
        """Cache a query result for deduplication (bounded to prevent memory leaks)."""
        qhash = self._query_hash(text, question)
        if qhash in self._query_cache:
            self._query_cache.move_to_end(qhash)
        self._query_cache[qhash] = result
        # Evict oldest entries when cache exceeds max size
        while len(self._query_cache) > _MAX_DEDUP_CACHE_SIZE:
            self._query_cache.popitem(last=False)

    def get_dedup_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            "total_queries": len(self._query_cache) + self._dedup_hits,
            "cached_results": len(self._query_cache),
            "dedup_hits": self._dedup_hits,
            "savings_percentage": round(
                (self._dedup_hits / max(len(self._query_cache) + self._dedup_hits, 1)) * 100, 1
            ),
        }

    def clear_dedup_cache(self):
        """Clear the deduplication cache."""
        self._query_cache.clear()
        self._dedup_hits = 0

    # ================================================================
    # Internal
    # ================================================================

    def _build_result(
        self,
        answer,
        compact_prompt,
        tokens_estimated,
        was_cached,
        payload_dict,
        input_type="image",
        original_tokens=0,
        compression_ratio=0.0,
        local_time_ms=0.0,
        remote_time_ms=0.0,
    ) -> dict:
        result = {
            "answer": answer,
            "compact_prompt": compact_prompt,
            "tokens_estimated": tokens_estimated,
            "was_cached": was_cached,
            "payload": payload_dict,
            "input_type": input_type,
            "selective_decoding_stats": self.selective_decoder.stats,
            "offline_first": self.config.offline_first,
            "dedup_stats": self.get_dedup_stats(),
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
