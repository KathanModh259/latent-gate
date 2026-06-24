"""
SemanticPayload — The compact representation that replaces raw tokens.

This is the VL-JEPA 'embedding' equivalent: a structured, compact
representation of visual input that captures task-relevant semantics
while abstracting away surface-level details.

Instead of sending ~1000 tokens (image description) to a cloud LLM,
we send this ~150-200 token structured payload.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class SemanticPayload:
    """
    Compact structured representation of visual input.

    Fields capture the essential semantics an LLM needs to reason
    about the image, without the verbose natural language overhead.
    """

    # ---- Core Visual Semantics ----
    scene_description: str = ""
    objects_detected: list = field(default_factory=list)
    spatial_relationships: list = field(default_factory=list)
    actions_activities: list = field(default_factory=list)
    text_in_image: str = ""
    dominant_colors: list = field(default_factory=list)
    scene_type: str = ""  # indoor/outdoor/document/chart/etc.
    confidence: float = 0.0

    # ---- Metadata ----
    extraction_model: str = ""
    extraction_time_ms: float = 0.0
    estimated_token_count: int = 0
    source_image: str = ""

    def to_compact_prompt(self) -> str:
        """
        Convert to minimal text for the cloud LLM.
        This is THE key function — it's what saves you tokens and money.

        Returns a compact string like:
        "[Scene: indoor] Description: A kitchen with ... | Objects: table, chair | ..."
        """
        parts = []

        if self.scene_type:
            parts.append(f"[Scene: {self.scene_type}]")
        if self.scene_description:
            parts.append(f"Description: {self.scene_description}")
        if self.objects_detected:
            parts.append(f"Objects: {', '.join(self.objects_detected[:10])}")
        if self.spatial_relationships:
            parts.append(f"Layout: {'; '.join(self.spatial_relationships[:5])}")
        if self.actions_activities:
            parts.append(f"Actions: {', '.join(self.actions_activities[:5])}")
        if self.text_in_image:
            parts.append(f"Text found: {self.text_in_image[:200]}")
        if self.dominant_colors:
            parts.append(f"Colors: {', '.join(self.dominant_colors[:3])}")

        compact = " | ".join(parts)
        self.estimated_token_count = len(compact.split()) + 10  # rough estimate
        return compact

    def to_dict(self) -> dict:
        """Serialize to dictionary (for caching and logging)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticPayload":
        """Deserialize from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def is_empty(self) -> bool:
        """Check if the payload has any meaningful content."""
        return not (self.scene_description or self.objects_detected or self.text_in_image)

    def __repr__(self) -> str:
        obj_count = len(self.objects_detected)
        return (
            f"SemanticPayload(scene={self.scene_type!r}, "
            f"objects={obj_count}, "
            f"~{self.estimated_token_count} tokens)"
        )
