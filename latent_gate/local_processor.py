"""
Local Processor — X-Encoder + Predictor (runs on Ollama, FREE)

Speed Optimizations:
  1. Uses FastClient (connection pooling + keep_alive)
  2. Shorter extraction prompt (fewer output tokens = faster)
  3. Concurrent image encoding while model processes
  4. Smart JSON parsing (avoids fallback LLM call when possible)
"""

import json
import base64
import time
import logging
import io
from pathlib import Path

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload
from latent_gate.cache import PayloadCache
from latent_gate.fast_client import FastClient

logger = logging.getLogger("latent_gate.local")


class LocalProcessor:
    """
    Handles all local (free) processing via Ollama.

    Two stages:
      X-Encode: Image → Raw visual extraction (via multimodal model)
      Predict:  Raw extraction → Structured SemanticPayload
    """

    # OPTIMIZED: Shorter prompt = fewer tokens generated = faster response
    EXTRACTION_PROMPT = """Extract image info as JSON only:
{"scene_type":"indoor/outdoor/document/chart/photo/other","scene_description":"1 sentence","objects":["obj1","obj2"],"spatial_layout":["rel1"],"actions":["action1"],"text_visible":"any text","colors":["color1","color2"]}"""

    RESTRUCTURE_PROMPT = """Extract from this analysis:
{raw_text}

Format: SCENE:<type>|DESC:<1 line>|OBJECTS:<list>|ACTIONS:<list>|TEXT:<text>"""

    def __init__(self, config: PipelineConfig, client: FastClient = None):
        self.config = config
        self.client = client or FastClient(config)
        self.cache = PayloadCache(config) if config.enable_caching else None

    # ----------------------------------------------------------------
    # Image Handling
    # ----------------------------------------------------------------

    @staticmethod
    def encode_image_to_base64(image_path: str, max_dimension: int = 0) -> str:
        """Read an image file and return its base64 encoding."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            raise ValueError(f"Unsupported image format: {path.suffix}")

        if max_dimension and max_dimension > 0:
            resized = LocalProcessor._try_resize_image(path, max_dimension)
            if resized:
                return base64.b64encode(resized).decode("utf-8")

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _try_resize_image(path: Path, max_dimension: int) -> bytes:
        """Resize large images when Pillow is available; otherwise keep original bytes."""
        try:
            from PIL import Image
        except ImportError:
            return b""

        try:
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                longest = max(width, height)
                if longest <= max_dimension:
                    return b""

                scale = max_dimension / longest
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resample = getattr(Image, "Resampling", Image).LANCZOS
                image = image.convert("RGB")
                image.thumbnail(new_size, resample)

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=85, optimize=True)
                logger.debug(
                    "Resized image before vision processing: %sx%s -> %sx%s",
                    width,
                    height,
                    image.size[0],
                    image.size[1],
                )
                return buffer.getvalue()
        except Exception as e:
            logger.debug(f"Image resize skipped for {path}: {e}")
            return b""

    # ----------------------------------------------------------------
    # Stage 1: X-Encoder (Vision Extraction)
    # ----------------------------------------------------------------

    def x_encode(self, image_path: str) -> str:
        """
        X-Encoder: Extract raw visual semantics using local multimodal model.
        Uses FastClient for connection pooling + keep_alive.
        """
        logger.info(f"X-Encoder: Processing '{image_path}' with {self.config.vision_model}")

        image_b64 = self.encode_image_to_base64(
            image_path,
            max_dimension=self.config.max_image_dimension,
        )

        raw_output = self.client.ollama_generate(
            model=self.config.vision_model,
            prompt=self.EXTRACTION_PROMPT,
            images=[image_b64],
            max_tokens=self.config.max_local_summary_tokens * 3,
        )

        logger.debug(f"X-Encoder output ({len(raw_output)} chars)")
        return raw_output

    # ----------------------------------------------------------------
    # Stage 2: Predictor (Structured Compression)
    # ----------------------------------------------------------------

    def predict(self, raw_extraction: str) -> SemanticPayload:
        """
        Predictor: Compress raw vision output into SemanticPayload.
        Tries fast JSON parse first, falls back to LLM only if needed.
        """
        logger.info("Predictor: Structuring extraction")

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

            logger.info("Predictor: JSON parsed (fast path)")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # --- Medium path: Try to extract JSON from mixed output ---
            json_extracted = self._extract_json_from_text(raw_extraction)
            if json_extracted:
                try:
                    data = json.loads(json_extracted)
                    payload.scene_type = str(data.get("scene_type", ""))
                    payload.scene_description = str(data.get("scene_description", ""))
                    payload.objects_detected = list(data.get("objects", []))
                    payload.spatial_relationships = list(data.get("spatial_layout", []))
                    payload.actions_activities = list(data.get("actions", []))
                    payload.text_in_image = str(data.get("text_visible", ""))
                    payload.dominant_colors = list(data.get("colors", []))
                    payload.confidence = 0.75
                    logger.info("Predictor: JSON extracted from mixed output (medium path)")
                except (json.JSONDecodeError, KeyError, TypeError):
                    json_extracted = None

            if not json_extracted:
                # --- Slow path: Use LLM to restructure ---
                logger.warning(f"JSON parse failed ({e}), using Predictor LLM (slow path)")
                prompt = self.RESTRUCTURE_PROMPT.format(raw_text=raw_extraction[:500])
                structured = self.client.ollama_generate(
                    model=self.config.text_smart_model,
                    prompt=prompt,
                    max_tokens=200,
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
        Full local pipeline: X-Encode → Predict → SemanticPayload
        """
        start_time = time.time()

        # Check cache first
        if self.cache:
            cached = self.cache.get(image_path)
            if cached:
                logger.info("Cache hit — instant return")
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
            f"Local done: ~{payload.estimated_token_count} tokens, "
            f"{payload.extraction_time_ms:.0f}ms"
        )
        return payload

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _clean_json_output(text: str) -> str:
        """Strip markdown code fences and whitespace."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        return text.strip()

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """Try to find a JSON object embedded in mixed text output."""
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return ""

    @staticmethod
    def _parse_restructured_output(text: str) -> SemanticPayload:
        """Parse the fallback pipe-delimited format."""
        payload = SemanticPayload()
        # Support both newline and pipe delimiters
        parts = text.replace("|", "\n").split("\n")
        for part in parts:
            part = part.strip()
            upper = part.upper()
            if upper.startswith("SCENE:"):
                payload.scene_type = part.split(":", 1)[1].strip()
            elif upper.startswith("DESC:"):
                payload.scene_description = part.split(":", 1)[1].strip()
            elif upper.startswith("OBJECTS:"):
                raw = part.split(":", 1)[1].strip()
                payload.objects_detected = [o.strip() for o in raw.split(",") if o.strip()]
            elif upper.startswith("ACTIONS:"):
                raw = part.split(":", 1)[1].strip()
                payload.actions_activities = [a.strip() for a in raw.split(",") if a.strip()]
            elif upper.startswith("TEXT:"):
                payload.text_in_image = part.split(":", 1)[1].strip()
        return payload
