"""
LatentGatePipeline — Main orchestrator.

Connects all components:
  Image → [LOCAL: X-Encoder → Predictor → SemanticPayload]
        → [Selective Decoding Check]
        → [REMOTE: Y-Decoder (Cloud LLM)] → Final Answer
"""

import logging
from typing import Optional

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.local_processor import LocalProcessor
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.remote_decoder import RemoteDecoder, create_decoder


logger = logging.getLogger("latent_gate.pipeline")


class LatentGatePipeline:
    """
    Main pipeline orchestrator inspired by VL-JEPA architecture.

    Flow:
      Image → [LOCAL: X-Encoder → Predictor → SemanticPayload]
            → [Selective Decoding Check]
            → [REMOTE: Y-Decoder (Cloud LLM)] → Final Answer

    Token savings: Instead of sending ~1000+ tokens (image description),
    we send ~150-200 tokens (structured semantic payload) to the cloud.

    Args:
        config: PipelineConfig with all settings.

    Example:
        >>> from latent_gate import LatentGatePipeline, PipelineConfig
        >>> config = PipelineConfig(vision_model="llava:7b", remote_provider="ollama")
        >>> pipeline = LatentGatePipeline(config)
        >>> result = pipeline.query("photo.jpg", "What objects are visible?")
        >>> print(result["answer"])
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
        self.selective_decoder = SelectiveDecoder(
            similarity_threshold=self.config.similarity_threshold
        )
        self.remote_decoder: RemoteDecoder = create_decoder(self.config)

        logger.info(
            f"Pipeline initialized: "
            f"vision={self.config.vision_model}, "
            f"remote={self.config.remote_provider}/{self.config.remote_model}"
        )

    def _setup_logging(self):
        """Configure logging based on config."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    def query(self, image_path: str, question: str) -> dict:
        """
        Process an image and answer a question about it.

        This is the main entry point for the pipeline.

        Args:
            image_path: Path to the image file.
            question: Question about the image.

        Returns:
            Dictionary with:
              - answer: The final text response.
              - payload: The SemanticPayload dict (for debugging/inspection).
              - compact_prompt: The compact text sent to the cloud LLM.
              - tokens_estimated: Approximate token count sent to cloud.
              - was_cached: True if selective decoding reused previous response.
              - selective_decoding_stats: Running stats for batch processing.
        """
        # ---- STAGE 1: LOCAL PROCESSING (FREE) ----
        logger.info("=" * 50)
        logger.info("STAGE 1: Local Processing (X-Encoder + Predictor)")

        payload = self.local_processor.process(image_path)
        compact_input = payload.to_compact_prompt()

        logger.info(f"Compact payload: ~{payload.estimated_token_count} tokens")
        logger.info(f"Extraction time: {payload.extraction_time_ms:.0f}ms")

        # ---- SELECTIVE DECODING CHECK ----
        if self.config.selective_decoding:
            if not self.selective_decoder.should_decode(payload):
                logger.info("Selective Decoding: Semantics unchanged → reusing previous response")
                return {
                    "answer": self.selective_decoder.previous_response,
                    "payload": payload.to_dict(),
                    "compact_prompt": compact_input,
                    "tokens_estimated": payload.estimated_token_count,
                    "was_cached": True,
                    "selective_decoding_stats": self.selective_decoder.stats,
                }

        # ---- STAGE 2: REMOTE DECODING (PAID — minimal tokens) ----
        logger.info("STAGE 2: Remote Decoding (Y-Decoder)")
        logger.info(
            f"Sending ~{payload.estimated_token_count} tokens "
            f"(vs ~1000+ traditional)"
        )

        answer = self.remote_decoder.decode(compact_input, question)

        # Update selective decoder state
        self.selective_decoder.update(payload, answer)

        return {
            "answer": answer,
            "payload": payload.to_dict(),
            "compact_prompt": compact_input,
            "tokens_estimated": payload.estimated_token_count,
            "was_cached": False,
            "selective_decoding_stats": self.selective_decoder.stats,
        }

    def query_batch(self, image_paths: list, question: str) -> list:
        """
        Process multiple images (e.g., video frames) with selective decoding.

        Selective decoding automatically skips API calls for frames where
        the semantic content hasn't changed significantly.

        Args:
            image_paths: List of image file paths.
            question: Question to ask about each image.

        Returns:
            List of result dictionaries (same format as query()).
        """
        results = []
        for i, path in enumerate(image_paths):
            logger.info(f"Processing frame {i + 1}/{len(image_paths)}: {path}")
            result = self.query(path, question)
            results.append(result)

        stats = self.selective_decoder.stats
        logger.info(f"Batch complete — {stats}")
        return results

    def reset_selective_decoder(self):
        """Reset selective decoder state (e.g., for a new video stream)."""
        self.selective_decoder.reset()
        logger.info("Selective decoder reset")
