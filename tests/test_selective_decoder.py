"""Tests for SelectiveDecoder."""

import pytest
from latent_gate.selective_decoder import SelectiveDecoder
from latent_gate.payload import SemanticPayload


class TestSelectiveDecoder:

    def setup_method(self):
        self.decoder = SelectiveDecoder(similarity_threshold=0.85)

    def test_first_frame_always_decodes(self):
        p = SemanticPayload(objects_detected=["cat"])
        assert self.decoder.should_decode(p) is True

    def test_identical_payloads_skip(self):
        p1 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table"],
            actions_activities=["sitting"],
            dominant_colors=["brown"],
        )
        p2 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table"],
            actions_activities=["sitting"],
            dominant_colors=["brown"],
        )

        # First frame always decodes
        assert self.decoder.should_decode(p1) is True
        self.decoder.update(p1, "A cat sitting on a table")

        # Second identical frame should skip
        assert self.decoder.should_decode(p2) is False

    def test_different_payloads_decode(self):
        p1 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table"],
            actions_activities=["sitting"],
        )
        p2 = SemanticPayload(
            scene_type="outdoor",
            objects_detected=["car", "road", "tree"],
            actions_activities=["driving"],
        )

        self.decoder.should_decode(p1)
        self.decoder.update(p1, "Cat on table")

        # Completely different scene should trigger decode
        assert self.decoder.should_decode(p2) is True

    def test_stats_tracking(self):
        p1 = SemanticPayload(objects_detected=["a"])
        p2 = SemanticPayload(objects_detected=["a"])
        p3 = SemanticPayload(objects_detected=["x", "y", "z"])

        self.decoder.should_decode(p1)
        self.decoder.update(p1, "response1")

        self.decoder.should_decode(p2)  # Should skip

        self.decoder.should_decode(p3)
        self.decoder.update(p3, "response3")

        stats = self.decoder.stats
        assert stats["total_frames"] == 3
        assert stats["api_calls"] == 2
        assert stats["skipped"] == 1

    def test_reset(self):
        p = SemanticPayload(objects_detected=["a"])
        self.decoder.should_decode(p)
        self.decoder.update(p, "test")

        self.decoder.reset()

        assert self.decoder.call_count == 0
        assert self.decoder.skip_count == 0
        assert self.decoder.previous_payload is None

    def test_jaccard_identical(self):
        assert self.decoder._jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_jaccard_disjoint(self):
        assert self.decoder._jaccard(["a", "b"], ["c", "d"]) == 0.0

    def test_jaccard_partial(self):
        result = self.decoder._jaccard(["a", "b", "c"], ["a", "b", "d"])
        assert 0.4 < result < 0.7  # 2/4 = 0.5

    def test_jaccard_empty(self):
        assert self.decoder._jaccard([], []) == 1.0
        assert self.decoder._jaccard(["a"], []) == 0.0
        assert self.decoder._jaccard([], ["b"]) == 0.0

    def test_similarity_computation(self):
        p1 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table"],
            actions_activities=["sitting"],
            dominant_colors=["brown"],
        )
        p2 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table"],
            actions_activities=["sitting"],
            dominant_colors=["brown"],
        )
        sim = self.decoder.compute_similarity(p1, p2)
        assert abs(sim - 1.0) < 1e-9

    def test_similarity_partial_overlap(self):
        p1 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table", "lamp"],
            actions_activities=["sitting"],
        )
        p2 = SemanticPayload(
            scene_type="indoor",
            objects_detected=["cat", "table", "book"],
            actions_activities=["reading"],
        )
        sim = self.decoder.compute_similarity(p1, p2)
        assert 0.3 < sim < 0.9  # Partial overlap
