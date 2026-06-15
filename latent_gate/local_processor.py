"""
Local Processor — X-Encoder + Predictor (runs on Ollama, FREE)

This is where the heavy lifting happens locally:
  1. X-Encoder: Process visual input via multimodal model (LLaVA/BakLLaVA)
  2. Predictor: Structure the raw extraction into a compact SemanticPayload

Inspired by VL-JEPA's X-Encoder (V-JEPA 2 ViT-L) and Predictor (Llama 3 layers).
"""

import json
import base64
import time
import logging
from pathlib import Path

import requests

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.cache import PayloadCache


logger = logging.getLogger("latent_gate.local")


class LocalProcessor:
    """
    Handles all local (free) processing via Ollama.

    Two stages:
      X-Encode: Image → Raw visual extraction (via multimodal model)
      Predict:  Raw extraction → Structured SemanticPayload (via text model)
    """

    # Structured extraction prompt — asks the vision model for parseable JSON
    EXTRACTION_PROMPT = """Analyze this image and extract ONLY the following in valid JSON format.
Be concise — use minimal words for each field:
{
  "scene_type": "indoor/outdoor/document/chart/screenshot/photo/other",
  "scene_description": "1-2 sentence factual description of the scene",
  "objects": ["object1 (key attribute)", "object2 (key attribute)"],
  "spatial_layout": ["obj1 is left of obj2", "obj3 is above obj4"],
  "actions": ["action being performed if any"],
  "text_visible": "any text visible in the image",
  "colors": ["dominant color 1", "dominant color 2"]
}
Return ONLY valid JSON. No markdown, no explanation."""

    # Fallback prompt if JSON parsing fails
    RESTRUCTURE_PROMPT = """Given this raw image analysis, extract a clean structured summary.

Raw analysis:
{raw_text}

Return in this EXACT format (one line each):
SCENE: <type>
DESC: <1 sentence description>
OBJECTS: <comma-separated list>
ACTIONS: <comma-separated list>
TEXT: <any visible text or "none">"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = PayloadCache(config) if config.enable_caching else None

    # ----------------------------------------------------------------
    # Ollama Communication
    # ----------------------------------------------------------------

    def _ollama_generate(
        self,
        model: str,
        prompt: str,
        images: list = None,
    ) -> str:
        """Call Ollama's /api/generate endpoint."""
        url = f"{self.config.ollama_base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_local_summary_tokens * 4,
            },
        }
        if images:
            payload["images"] = images  # list of base64-encoded strings

        try:
            resp = requests.post(
                url, json=payload, timeout=self.config.request_timeout
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. Make sure it's running:\n"
                "  Start:  ollama serve\n"
                "  Check:  curl http://localhost:11434/api/tags"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.config.request_timeout}s. "
                "Try a smaller model or increase request_timeout in config."
            )

    # ----------------------------------------------------------------
    # Image Handling
    # ----------------------------------------------------------------

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Read an image file and return its base64 encoding."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not path.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            raise ValueError(f"Unsupported image format: {path.suffix}")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ----------------------------------------------------------------
    # Stage 1: X-Encoder (Vision Extraction)
    # ----------------------------------------------------------------

    def x_encode(self, image_path: str) -> str:
        """
        X-Encoder: Extract raw visual semantics using local multimodal model.

        This is the heavy lifting done locally for FREE.
        Equivalent to VL-JEPA's X-Encoder (V-JEPA 2 ViT-L).

        Args:
            image_path: Path to the image file.

        Returns:
            Raw text output from the vision model (ideally JSON).
        """
        logger.info(f"X-Encoder: Processing '{image_path}' with {self.config.vision_model}")

        image_b64 = self.encode_image_to_base64(image_path)

        raw_output = self._ollama_generate(
            model=self.config.vision_model,
            prompt=self.EXTRACTION_PROMPT,
            images=[image_b64],
        )

        logger.debug(f"X-Encoder raw output ({len(raw_output)} chars): {raw_output[:200]}...")
        return raw_output

    # ----------------------------------------------------------------
    # Stage 2: Predictor (Structured Compression)
    # ----------------------------------------------------------------

    def predict(self, raw_extraction: str) -> SemanticPayload:
        """
        Predictor: Takes raw vision output and compresses into a clean
        SemanticPayload.

        Tries JSON parsing first (fast path), falls back to LLM
        restructuring if needed (slow path).

        Equivalent to VL-JEPA's Predictor (Llama 3 Transformer layers).

        Args:
            raw_extraction: Raw text from the X-Encoder.

        Returns:
            Structured SemanticPayload.
        """
        logger.info("Predictor: Structuring extraction into SemanticPayload")

        payload = SemanticPayload()

        # --- Fast path: Direct JSON parse ---
        try:
            cleaned = self._clean_json_output(raw_extraction)
            data = json.loads(cleaned)

            payload.scene_type = str(data.get("scene_type", ""))
            payload.scene_description = str(data.get("scene_description", ""))
            payload.objects_detected = list(data.get("objects", []))
            payload.spatial_relationships = list(data.get("spatial_layout", []))
            payload.actions_activities = list(data.get("actions", []))
            payload.text_in_image = str(data.get("text_visible", ""))
            payload.dominant_colors = list(data.get("colors", []))
            payload.confidence = 0.85

            logger.info("Predictor: JSON parsed successfully (fast path)")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # --- Slow path: Use local LLM to restructure ---
            logger.warning(f"Direct JSON parse failed ({e}), using Predictor LLM")

            prompt = self.RESTRUCTURE_PROMPT.format(
                raw_text=raw_extraction[:500]
            )
            structured = self._ollama_generate(
                model=self.config.predictor_model,
                prompt=prompt,
            )
            payload = self._parse_restructured_output(structured)
            payload.confidence = 0.60

        payload.extraction_model = self.config.vision_model
        return payload

    # ----------------------------------------------------------------
    # Full Local Pipeline
    # ----------------------------------------------------------------

    def process(self, image_path: str) -> SemanticPayload:
        """
        Full local processing pipeline.

        X-Encode (vision) → Predict (structure) → SemanticPayload

        Args:
            image_path: Path to the image file.

        Returns:
            Compact SemanticPayload ready to send to cloud LLM.
        """
        start_time = time.time()

        # Check cache first
        if self.cache:
            cached = self.cache.get(image_path)
            if cached:
                logger.info("Cache hit — skipping local processing")
                return cached

        # Stage 1: X-Encode
        raw = self.x_encode(image_path)

        # Stage 2: Predict
        payload = self.predict(raw)
        payload.extraction_time_ms = (time.time() - start_time) * 1000
        payload.source_image = image_path

        # Cache the result
        if self.cache:
            self.cache.put(image_path, payload)

        logger.info(
            f"Local processing complete: {payload.estimated_token_count} tokens, "
            f"{payload.extraction_time_ms:.0f}ms"
        )
        return payload

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _clean_json_output(text: str) -> str:
        """Strip markdown code fences and whitespace from model output."""
        text = text.strip()
        # Remove ```json ... ``` wrapper
        if text.startswith("```"):
            lines = text.split("\n")
            # Drop first line (```json) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return text.strip()

    @staticmethod
    def _parse_restructured_output(text: str) -> SemanticPayload:
        """Parse the fallback SCENE/DESC/OBJECTS format."""
        payload = SemanticPayload()
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SCENE:"):
                payload.scene_type = line.split(":", 1)[1].strip()
            elif line.upper().startswith("DESC:"):
                payload.scene_description = line.split(":", 1)[1].strip()
            elif line.upper().startswith("OBJECTS:"):
                raw = line.split(":", 1)[1].strip()
                payload.objects_detected = [o.strip() for o in raw.split(",") if o.strip()]
            elif line.upper().startswith("ACTIONS:"):
                raw = line.split(":", 1)[1].strip()
                payload.actions_activities = [a.strip() for a in raw.split(",") if a.strip()]
            elif line.upper().startswith("TEXT:"):
                payload.text_in_image = line.split(":", 1)[1].strip()
        return payload
