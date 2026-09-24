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
  9. Model fallback chain (auto-failover to Ollama when primary fails)
  10. Selective decoder reset between input types
"""

import dataclasses
import logging
import time
import threading
import hashlib
import json
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from latent_gate.config import PipelineConfig
from latent_gate.text_processor import TextProcessor
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import RemoteDecoder, create_decoder, get_fallback_decoder
from latent_gate.fast_client import FastClient
from latent_gate.cost_tracker import CostTracker

logger = logging.getLogger("latent_gate.pipeline")

_MAX_DEDUP_CACHE_SIZE = 1000


class _FallbackDecoderWrapper:
    """
    Wraps a primary and fallback decoder. Tries primary first; on failure,
    catches the error, logs a warning, and retries with the fallback.
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    def decode(self, compact_input: str, user_query: str) -> Tuple[str, dict]:
        try:
            return self.primary.decode(compact_input, user_query)
        except (ConnectionError, TimeoutError, PermissionError, Exception) as e:
            logger.warning(f"Primary decoder failed ({e}), falling back to Ollama")
            if self.fallback:
                return self.fallback.decode(compact_input, user_query)
            raise

    def decode_stream(self, compact_input: str, user_query: str):
        try:
            yield from self.primary.decode_stream(compact_input, user_query)
        except (ConnectionError, TimeoutError, PermissionError, Exception) as e:
            logger.warning(f"Primary streaming failed ({e}), falling back to Ollama")
            if self.fallback:
                yield from self.fallback.decode_stream(compact_input, user_query)
            else:
                raise


class LatentGatePipeline:
    """
    Main pipeline orchestrator inspired by VL-JEPA architecture.

    v0.3.0 Speed Optimizations:
      - Shared FastClient (single connection pool for everything)
      - Model preloading (warm GPU on init)
      - Parallel processing (image + text at the same time)
      - 3-tier JSON parsing (fast -> medium -> slow fallback)

    v1.2.4+:
      - Model fallback chain (auto-failover to Ollama)
      - Selective decoder reset between input types
      - Batch compression support

    Supports:
      1. query()              - Image + question
      2. query_text()         - Text prompt compression
      3. query_conversation() - Conversation history compression
      4. query_documents()    - RAG document compression
      5. query_universal()    - Auto-detect (image, text, or both)
    """

    def __init__(self, config: Optional[PipelineConfig] = None, preload: bool = True):
        from latent_gate.config_loader import get_config

        self.config = config or get_config()
        self._setup_logging()

        for w in self.config.validate():
            logger.warning(f"Config: {w}")

        self.client = FastClient(self.config)

        self.local_processor = LocalProcessor(self.config, client=self.client)
        self.text_processor = TextProcessor(self.config)
        self.text_processor.client = self.client
        self.selective_decoder = SelectiveDecoder(
            similarity_threshold=self.config.similarity_threshold,
            use_embeddings=self.config.use_embeddings,
        )
        self.remote_decoder: RemoteDecoder = create_decoder(self.config, client=self.client)

        self._offline_decoder = None
        if self.config.offline_first:
            from latent_gate.remote_decoder import OllamaRemoteDecoder

            # Local answering must use the local offline_model, not the cloud remote_model
            offline_config = dataclasses.replace(
                self.config, remote_provider="ollama", remote_model=self.config.offline_model
            )
            self._offline_decoder = OllamaRemoteDecoder(offline_config, client=self.client)
            logger.info("Offline-first mode enabled")

        self._fallback_decoder = None
        if self.config.remote_provider.lower() != "ollama":
            self._fallback_decoder = get_fallback_decoder(self.config, client=self.client)
            if self._fallback_decoder:
                logger.info(f"Fallback decoder ready: {self.config.remote_provider} -> ollama")

        self._last_input_type: str = ""
        self._selective_last_question: Optional[str] = None

        self._query_cache: OrderedDict = OrderedDict()
        self._dedup_hits: int = 0
        self._state_lock = threading.RLock()

        self.tracker = None
        if self.config.track_costs:
            self.tracker = CostTracker(db_path=self.config.cost_db_path)
            logger.info(f"Cost tracking enabled (DB: {self.config.cost_db_path})")

        self._executor = ThreadPoolExecutor(max_workers=3)

        if preload:
            try:
                self.client.preload_models()
                logger.info("Models preloaded into GPU memory")
            except Exception as e:
                logger.warning(f"Preload skipped: {e}")

        logger.info(
            f"Pipeline ready: vision={self.config.vision_model}, "
            f"text_fast={self.config.text_fast_model}, "
            f"text_smart={self.config.text_smart_model}, "
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

    def _reset_if_mode_changed(self, new_input_type: str) -> None:
        """Reset selective decoder state when the input type changes."""
        if self._last_input_type and self._last_input_type != new_input_type:
            logger.info(
                f"Input type changed ({self._last_input_type} -> {new_input_type}), resetting selective decoder"
            )
            self.selective_decoder.reset()
        self._last_input_type = new_input_type

    # ================================================================
    # MODE 1: Image Query
    # ================================================================

    def query(self, image_path: str, question: str, compress_only: bool = False) -> dict:
        """Process an image and answer a question about it."""
        logger.info("=" * 50)
        self._reset_if_mode_changed("image")

        image_key = self._dedup_key("image", self._image_fingerprint(image_path), compress_only)
        cached = self._is_duplicate(image_key, question)
        if cached:
            return cached

        logger.info("STAGE 1: Local Vision Processing")
        start = time.time()

        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        logger.info(f"Local: ~{payload.estimated_token_count} tokens, {local_ms:.0f}ms")

        if self.config.selective_decoding and not compress_only:
            with self._state_lock:
                should_decode = self.selective_decoder.should_decode(payload)
                previous_response = self.selective_decoder.previous_response
                # A similar scene only answers the SAME question; a new question must decode
                if question != self._selective_last_question:
                    should_decode = True
            if not should_decode:
                logger.info("Selective: reusing previous")
                result = self._build_result(
                    previous_response,
                    compact_input,
                    payload.estimated_token_count,
                    was_cached=True,
                    payload_dict=payload.to_dict(),
                    input_type="image",
                    local_time_ms=local_ms,
                )
                self._cache_result(image_key, question, result)
                return result

        if compress_only:
            answer, api_usage, remote_ms = "", None, 0.0
        else:
            logger.info("STAGE 2: Remote Decoding")
            remote_start = time.time()
            decoder = self._get_decoder()
            answer, api_usage = decoder.decode(compact_input, question)
            remote_ms = (time.time() - remote_start) * 1000
            with self._state_lock:
                self.selective_decoder.update(payload, answer)
                self._selective_last_question = question

        result = self._build_result(
            answer,
            compact_input,
            payload.estimated_token_count,
            was_cached=False,
            payload_dict=payload.to_dict(),
            input_type="image",
            local_time_ms=local_ms,
            remote_time_ms=remote_ms,
            api_usage=api_usage,
        )
        self._cache_result(image_key, question, result)
        return result

    # ================================================================
    # MODE 2: Text Query
    # ================================================================

    def query_text(
        self, text: str, question: str = "", mode: str = "auto", compress_only: bool = False
    ) -> dict:
        """Compress a text prompt locally, then optionally send to cloud LLM."""
        logger.info("=" * 50)
        self._reset_if_mode_changed("text")
        logger.info("STAGE 1: Local Text Compression")
        start = time.time()

        text_key = self._dedup_key(f"text/{mode}", text, compress_only)
        cached = self._is_duplicate(text_key, question)
        if cached:
            return cached

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
            f"Compressed: {text_payload.original_token_count} -> {text_payload.compressed_token_count} tokens ({local_ms:.0f}ms)"
        )

        is_compress_only = compress_only or (
            mode == "compress" and (not question or "compress" in question.lower())
        )

        if is_compress_only:
            answer, api_usage, remote_ms = "", None, 0.0
        else:
            logger.info("STAGE 2: Remote Decoding")
            remote_start = time.time()
            final_q = question or text_payload.intent or ""
            decoder = self._get_decoder()
            try:
                answer, api_usage = decoder.decode(compact_input, final_q)
            except (ConnectionError, PermissionError) as e:
                logger.warning(f"Remote decode unavailable ({e}), returning compressed only")
                answer = f"[Remote decode skipped: {e}]"
                api_usage = None
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
            api_usage=api_usage,
        )
        self._cache_result(text_key, question, result)
        return result

    # ================================================================
    # MODE 2b: Conversation Compression
    # ================================================================

    def query_conversation(
        self, messages: list, new_question: str, compress_only: bool = False
    ) -> dict:
        """Compress conversation history + ask a new question."""
        self._reset_if_mode_changed("conversation")
        start = time.time()

        msg_text = self._dedup_key(
            "conversation", json.dumps(messages, sort_keys=True, default=str), compress_only
        )
        cached = self._is_duplicate(msg_text, new_question)
        if cached:
            return cached

        text_payload = self.text_processor.compress_conversation(messages)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        if compress_only:
            answer, api_usage, remote_ms = "", None, 0.0
        else:
            remote_start = time.time()
            decoder = self._get_decoder()
            answer, api_usage = decoder.decode(compact_input, new_question)
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
            api_usage=api_usage,
        )
        self._cache_result(msg_text, new_question, result)
        return result

    # ================================================================
    # MODE 2c: RAG Document Compression
    # ================================================================

    def query_documents(self, documents: list, question: str, compress_only: bool = False) -> dict:
        """Compress RAG-retrieved documents + answer a question."""
        self._reset_if_mode_changed("documents")
        start = time.time()

        doc_hash = self._dedup_key("documents", "".join(documents), compress_only)
        cached = self._is_duplicate(doc_hash, question)
        if cached:
            return cached

        text_payload = self.text_processor.compress_documents(documents, question)
        compact_input = text_payload.to_compact_prompt()
        local_ms = (time.time() - start) * 1000

        if compress_only:
            answer, api_usage, remote_ms = "", None, 0.0
        else:
            remote_start = time.time()
            decoder = self._get_decoder()
            answer, api_usage = decoder.decode(compact_input, question)
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
            api_usage=api_usage,
        )
        self._cache_result(doc_hash, question, result)
        return result

    # ================================================================
    # MODE 3: Universal (auto-detect) — WITH PARALLEL PROCESSING
    # ================================================================

    def query_universal(
        self, text: str = "", image: str = "", question: str = "", compress_only: bool = False
    ) -> dict:
        """Universal entry point - auto-detects input type."""
        has_image = bool(image)
        has_text = bool(text)
        if has_image and has_text:
            self._reset_if_mode_changed("image+text")
        elif has_image:
            self._reset_if_mode_changed("image")
        elif has_text:
            self._reset_if_mode_changed("text")

        if has_image and has_text:
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

            if compress_only:
                answer, api_usage, remote_ms = "", None, 0.0
            else:
                remote_start = time.time()
                decoder = self._get_decoder()
                answer, api_usage = decoder.decode(combined, final_q)
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
                api_usage=api_usage,
            )

        elif has_image:
            return self.query(
                image, question or "Describe this image.", compress_only=compress_only
            )
        elif has_text:
            return self.query_text(text, question, compress_only=compress_only)
        else:
            raise ValueError("Provide at least 'text' or 'image' input.")

    # ================================================================
    # Batch Processing
    # ================================================================

    def query_batch(
        self, image_paths: list, question: str, parallel: bool = False, max_workers: int = 3
    ) -> list:
        if not parallel:
            results = []
            for i, path in enumerate(image_paths):
                logger.info(f"Frame {i + 1}/{len(image_paths)}: {path}")
                results.append(self.query(path, question))
            logger.info(f"Batch: {self.selective_decoder.stats}")
            return results

        logger.info(f"Parallel batch: {len(image_paths)} images with {max_workers} workers")

        def _process_single(path: str) -> dict:
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
                answer, api_usage = remote_decoder.decode(compact_input, question)
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

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single, path): path for path in image_paths}
            results = []
            for future in as_completed(futures):
                path = futures[future]
                try:
                    results.append(future.result())
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

        path_to_index = {path: i for i, path in enumerate(image_paths)}
        results.sort(
            key=lambda r: path_to_index.get(
                r.get("payload", {}).get("source_image", ""), len(image_paths) if image_paths else 0
            )
        )
        return results

    def query_batch_texts(
        self, texts: list, question: str = "", parallel: bool = True, max_workers: int = 3
    ) -> list:
        if not parallel:
            return [self.query_text(text, question) for text in texts]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.query_text, text, question): i for i, text in enumerate(texts)
            }
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
        return results

    # ================================================================
    # Streaming Methods
    # ================================================================

    def query_stream(self, image_path: str, question: str):
        logger.info("Streaming image query")
        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()
        decoder = self._get_decoder()
        yield from decoder.decode_stream(compact_input, question)

    def query_text_stream(self, text: str, question: str = "", mode: str = "auto"):
        logger.info("Streaming text query")
        text_payload = self.text_processor.compress(text, mode=mode, question=question)
        compact_input = text_payload.to_compact_prompt()
        final_q = question or text_payload.intent or ""
        decoder = self._get_decoder()
        yield from decoder.decode_stream(compact_input, final_q)

    def query_universal_stream(self, text: str = "", image: str = "", question: str = ""):
        has_image = bool(image)
        has_text = bool(text)
        if has_image and has_text:
            image_payload = self.local_processor.process(image)
            text_payload = self.text_processor.compress(text, "auto", question)
            combined = f"[VISUAL]: {image_payload.to_compact_prompt()} | [TEXT]: {text_payload.to_compact_prompt()}"
            final_q = question or text_payload.intent or "Analyze the image and text together."
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
        """Compress a verbose prompt into a concise one WITHOUT generating an answer."""
        return self.text_processor.compress_prompt(text)

    def estimate_cost(
        self, provider: str = "", model: str = "", input_tokens: int = 0, output_tokens: int = 0
    ) -> dict:
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
        self.client.close()
        self._executor.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ================================================================
    # Offline-First & Fallback Decoder
    # ================================================================

    def _decode_offline(self, compact_input: str, question: str) -> Tuple[str, dict]:
        if self._offline_decoder:
            return self._offline_decoder.decode(compact_input, question)
        return self.remote_decoder.decode(compact_input, question)

    def _get_decoder(self):
        """Get the appropriate decoder (offline, remote, or fallback to Ollama)."""
        if self.config.offline_first and self._offline_decoder:
            return self._offline_decoder
        if self._fallback_decoder:
            return _FallbackDecoderWrapper(self.remote_decoder, self._fallback_decoder)
        return self.remote_decoder

    # ================================================================
    # Adaptive Compression
    # ================================================================

    def _estimate_complexity(self, text: str, input_type: str = "text") -> float:
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
        return int(200 + (complexity * 300))

    # ================================================================
    # Semantic Deduplication
    # ================================================================

    def _query_hash(self, text: str, question: str = "") -> str:
        # Hash the FULL input: prefix-only keys made inputs sharing a header collide.
        # Callers namespace `text` by query kind/mode (see _dedup_key).
        content = f"{text}{question.strip()}"
        return hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()

    @staticmethod
    def _dedup_key(kind: str, content: str, compress_only: bool) -> str:
        """Namespace a dedup key so compress-only and full answers never mix."""
        return f"{kind}:{'c' if compress_only else 'q'}:{content}"

    @staticmethod
    def _image_fingerprint(image_path: str) -> str:
        """Content hash of an image, so an overwritten file isn't served a stale answer."""
        h = hashlib.sha256()
        try:
            with open(image_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return f"path:{image_path}"  # let the processor raise the real error
        return h.hexdigest()

    def _is_duplicate(self, text: str, question: str = "") -> Optional[dict]:
        qhash = self._query_hash(text, question)
        if qhash in self._query_cache:
            self._dedup_hits += 1
            logger.info(f"Semantic dedup: reusing cached result for hash {qhash[:8]}...")
            return self._query_cache[qhash]
        return None

    def _cache_result(self, text: str, question: str, result: dict):
        # Never pin a transient failure (e.g. a network blip) as the cached answer
        if str(result.get("answer", "")).startswith("[Remote decode skipped"):
            return
        qhash = self._query_hash(text, question)
        if qhash in self._query_cache:
            self._query_cache.move_to_end(qhash)
        self._query_cache[qhash] = result
        while len(self._query_cache) > _MAX_DEDUP_CACHE_SIZE:
            self._query_cache.popitem(last=False)

    def get_dedup_stats(self) -> dict:
        return {
            "total_queries": len(self._query_cache) + self._dedup_hits,
            "cached_results": len(self._query_cache),
            "dedup_hits": self._dedup_hits,
            "savings_percentage": round(
                (self._dedup_hits / max(len(self._query_cache) + self._dedup_hits, 1)) * 100, 1
            ),
        }

    def clear_dedup_cache(self):
        self._query_cache.clear()
        self._dedup_hits = 0

    # ================================================================
    # Result Builder
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
        api_usage=None,
    ) -> dict:
        tokens_actual = None
        token_source = "estimated"
        if api_usage and api_usage.get("source") == "provider":
            tokens_actual = api_usage.get("total_tokens")
            token_source = "provider"

        if self.tracker and not was_cached and answer:
            used_tokens = tokens_actual or tokens_estimated
            saved = max(0, original_tokens - used_tokens) if original_tokens > 0 else 0
            ratio = float(original_tokens / max(used_tokens, 1)) if original_tokens > 0 else 0.0
            latency = local_time_ms + remote_time_ms
            self.tracker.record_usage(
                query_type=input_type,
                provider=self.config.remote_provider,
                model=self.config.remote_model,
                input_tokens=used_tokens,
                output_tokens=api_usage.get("completion_tokens", 0) if api_usage else 0,
                tokens_saved=int(saved),
                compression_ratio=round(ratio, 2),
                latency_ms=round(latency, 1),
            )

        confidence = None
        if isinstance(payload_dict, dict) and "confidence" in payload_dict:
            confidence = payload_dict["confidence"]
        elif input_type == "image+text" and "image" in payload_dict:
            confidence = payload_dict["image"].get("confidence")

        if confidence is not None and confidence < 0.70:
            logger.warning(f"Low extraction confidence: {confidence:.2f}")

        result = {
            "answer": answer,
            "compact_prompt": compact_prompt,
            "tokens_estimated": tokens_estimated,
            "tokens_actual": tokens_actual,
            "token_source": token_source,
            "was_cached": was_cached,
            "payload": payload_dict,
            "extraction_confidence": confidence,
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
            result["tokens_saved"] = original_tokens - (tokens_actual or tokens_estimated)
        return result
