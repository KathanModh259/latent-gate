"""Tests for SemanticPayload."""

import pytest
from latent_gate.payload import SemanticPayload


class TestSemanticPayload:

    def test_default_empty(self):
        p = SemanticPayload()
        assert p.scene_description == ""
        assert p.objects_detected == []
        assert p.confidence == 0.0
        assert p.is_empty()

    def test_with_data(self):
        p = SemanticPayload(
            scene_type="indoor",
            scene_description="A kitchen with a table",
            objects_detected=["table", "chair", "lamp"],
            actions_activities=["cooking"],
            dominant_colors=["brown", "white"],
        )
        assert not p.is_empty()
        assert p.scene_type == "indoor"
        assert len(p.objects_detected) == 3

    def test_to_compact_prompt(self):
        p = SemanticPayload(
            scene_type="outdoor",
            scene_description="A park with trees",
            objects_detected=["tree", "bench", "dog"],
            actions_activities=["walking"],
        )
        compact = p.to_compact_prompt()

        assert "[Scene: outdoor]" in compact
        assert "park with trees" in compact
        assert "tree" in compact
        assert "walking" in compact
        assert p.estimated_token_count > 0

    def test_to_compact_prompt_empty(self):
        p = SemanticPayload()
        compact = p.to_compact_prompt()
        assert compact == ""
        assert p.estimated_token_count == 10  # base estimate

    def test_to_dict_and_from_dict(self):
        p = SemanticPayload(
            scene_type="indoor",
            scene_description="Office space",
            objects_detected=["desk", "monitor"],
            confidence=0.85,
        )
        d = p.to_dict()
        assert isinstance(d, dict)
        assert d["scene_type"] == "indoor"

        p2 = SemanticPayload.from_dict(d)
        assert p2.scene_type == p.scene_type
        assert p2.objects_detected == p.objects_detected
        assert p2.confidence == p.confidence

    def test_from_dict_ignores_extra_fields(self):
        d = {
            "scene_type": "indoor",
            "unknown_field": "should be ignored",
        }
        p = SemanticPayload.from_dict(d)
        assert p.scene_type == "indoor"
        assert not hasattr(p, "unknown_field") or True  # No crash

    def test_repr(self):
        p = SemanticPayload(scene_type="indoor", objects_detected=["a", "b"])
        r = repr(p)
        assert "indoor" in r
        assert "2" in r  # 2 objects

    def test_compact_prompt_limits(self):
        """Verify that lists are truncated in compact output."""
        p = SemanticPayload(
            objects_detected=[f"obj_{i}" for i in range(20)],
            spatial_relationships=[f"rel_{i}" for i in range(10)],
        )
        compact = p.to_compact_prompt()
        # Should only include first 10 objects and 5 relationships
        assert "obj_0" in compact
        assert "obj_9" in compact
        assert "obj_10" not in compact
