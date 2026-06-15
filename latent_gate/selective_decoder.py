"""
Selective Decoder — Only call the expensive remote LLM when semantics change.

Inspired by VL-JEPA's selective decoding mechanism which achieved ~2.85x
reduction in decoding operations while maintaining similar performance.

For video streams: instead of calling the API for every frame, we compare
consecutive SemanticPayloads and skip the API call if the scene hasn't
changed significantly.
"""

import logging
from typing import Optional

from latent_gate.payload import SemanticPayload


logger = logging.getLogger("latent_gate.selective")


class SelectiveDecoder:
    """
    Semantic change detector for streaming/batch processing.

    Compares consecutive SemanticPayloads using Jaccard similarity.
    If the similarity exceeds the threshold, the previous API response
    is reused — saving an expensive cloud LLM call.

    Attributes:
        threshold: Similarity above which frames are considered "same" (0-1).
        call_count: Number of actual API calls made.
        skip_count: Number of API calls avoided.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self.previous_payload: Optional[SemanticPayload] = None
        self.previous_response: str = ""
        self.call_count: int = 0
        self.skip_count: int = 0

    def compute_similarity(
        self, p1: SemanticPayload, p2: SemanticPayload
    ) -> float:
        """
        Compute semantic similarity between two payloads.

        Uses weighted Jaccard similarity across key fields.
        In a true VL-JEPA system, this would be cosine similarity
        in embedding space — but Jaccard on structured fields works
        well for this use case.

        Args:
            p1: First payload.
            p2: Second payload.

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        if not p1 or not p2:
            return 0.0

        sim_objects = self._jaccard(p1.objects_detected, p2.objects_detected)
        sim_actions = self._jaccard(p1.actions_activities, p2.actions_activities)
        sim_scene = 1.0 if p1.scene_type == p2.scene_type else 0.0
        sim_colors = self._jaccard(p1.dominant_colors, p2.dominant_colors)

        # Weighted average — objects and actions matter most
        weighted = (
            0.35 * sim_objects
            + 0.35 * sim_actions
            + 0.20 * sim_scene
            + 0.10 * sim_colors
        )
        return weighted

    def should_decode(self, current_payload: SemanticPayload) -> bool:
        """
        Decide whether to call the remote API or reuse the previous response.

        Args:
            current_payload: The payload from the current frame/image.

        Returns:
            True if remote API should be called, False if previous response can be reused.
        """
        if self.previous_payload is None:
            logger.debug("First frame — must decode")
            return True

        similarity = self.compute_similarity(self.previous_payload, current_payload)
        logger.debug(f"Similarity with previous: {similarity:.3f} (threshold: {self.threshold})")

        if similarity >= self.threshold:
            self.skip_count += 1
            logger.info(f"Selective skip — similarity {similarity:.3f} >= {self.threshold}")
            return False
        else:
            logger.info(f"Semantic change detected — similarity {similarity:.3f} < {self.threshold}")
            return True

    def update(self, payload: SemanticPayload, response: str):
        """Update internal state after a successful decode operation."""
        self.previous_payload = payload
        self.previous_response = response
        self.call_count += 1

    def reset(self):
        """Reset state (e.g., for a new video stream)."""
        self.previous_payload = None
        self.previous_response = ""
        self.call_count = 0
        self.skip_count = 0

    @property
    def stats(self) -> dict:
        """Get selective decoding statistics."""
        total = self.call_count + self.skip_count
        return {
            "total_frames": total,
            "api_calls": self.call_count,
            "skipped": self.skip_count,
            "reduction_ratio": f"{total / max(self.call_count, 1):.2f}x",
            "skip_rate": f"{self.skip_count / max(total, 1) * 100:.1f}%",
        }

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _jaccard(a: list, b: list) -> float:
        """Jaccard similarity between two lists of strings."""
        set_a = set(str(x).lower().strip() for x in a if x)
        set_b = set(str(x).lower().strip() for x in b if x)

        if not set_a and not set_b:
            return 1.0  # Both empty = same
        if not set_a or not set_b:
            return 0.0  # One empty = different

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union
