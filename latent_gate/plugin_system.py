"""
Plugin System — Custom processors for domain-specific compression.

Allows users to extend LatentGate with custom processors for specialized
use cases like medical imaging, satellite imagery, document analysis, etc.

Features:
  - Plugin registration and discovery
  - Custom processor interfaces
  - Plugin configuration
  - Lifecycle hooks
"""

import importlib
import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type
from pathlib import Path

try:
    import numpy as np
except ImportError:  # only DocumentPreProcessor needs it (pulled in by the [video] extra)
    np = None

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload

logger = logging.getLogger("latent_gate.plugins")


# ============================================================================
# Plugin Base Classes
# ============================================================================


class ProcessorPlugin(ABC):
    """
    Base class for processor plugins.

    Subclass this to create custom processors for specialized use cases.

    Example:
        class MedicalImageProcessor(ProcessorPlugin):
            name = "medical_image"

            def process(self, image_path: str, **kwargs) -> SemanticPayload:
                # Custom processing logic
                pass
    """

    name: str = "base"
    description: str = "Base processor plugin"
    version: str = "1.0.0"

    def __init__(self, config: Optional[PipelineConfig] = None):
        from latent_gate.config_loader import get_config

        self.config = config or get_config()

    @abstractmethod
    def process(self, *args, **kwargs) -> SemanticPayload:
        """Process input and return a SemanticPayload."""
        pass

    def validate(self) -> List[str]:
        """Validate plugin configuration. Return list of warnings."""
        return []

    def setup(self):
        """Called when the plugin is loaded."""
        pass

    def teardown(self):
        """Called when the plugin is unloaded."""
        pass


class PreProcessorPlugin(ProcessorPlugin):
    """
    Pre-processor plugin for input preprocessing.

    Runs before the main processor to clean/transform input data.
    """

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess the input data."""
        pass


class PostProcessorPlugin(ProcessorPlugin):
    """
    Post-processor plugin for output postprocessing.

    Runs after the main processor to enhance/refine output.
    """

    @abstractmethod
    def postprocess(self, payload: SemanticPayload) -> SemanticPayload:
        """Postprocess the SemanticPayload."""
        pass


class SimilarityPlugin(ProcessorPlugin):
    """
    Similarity plugin for custom similarity calculations.

    Replaces the default Jaccard/cosine similarity with custom logic.
    """

    @abstractmethod
    def compute_similarity(self, p1: SemanticPayload, p2: SemanticPayload) -> float:
        """Compute similarity between two payloads."""
        pass


# ============================================================================
# Plugin Manager
# ============================================================================


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.

    Usage:
        manager = PluginManager()
        manager.load_plugins_from_directory("./plugins")
        manager.register("my_plugin", MyProcessor())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        from latent_gate.config_loader import get_config

        self.config = config or get_config()
        self._plugins: Dict[str, ProcessorPlugin] = {}
        self._plugin_classes: Dict[str, Type[ProcessorPlugin]] = {}

    def register(self, name: str, plugin: ProcessorPlugin):
        """
        Register a plugin instance.

        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, overwriting")

        # Validate plugin
        warnings = plugin.validate()
        for warning in warnings:
            logger.warning(f"Plugin '{name}': {warning}")

        # Call setup hook
        plugin.setup()

        self._plugins[name] = plugin
        logger.info(f"Registered plugin: {name} (v{plugin.version})")

    def register_class(self, name: str, plugin_class: Type[ProcessorPlugin]):
        """
        Register a plugin class (will be instantiated on first use).

        Args:
            name: Plugin name
            plugin_class: Plugin class
        """
        self._plugin_classes[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def get(self, name: str) -> Optional[ProcessorPlugin]:
        """
        Get a registered plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        # Check instance registry first
        if name in self._plugins:
            return self._plugins[name]

        # Check class registry
        if name in self._plugin_classes:
            plugin = self._plugin_classes[name](self.config)
            self.register(name, plugin)
            return plugin

        return None

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        all_plugins = set(self._plugins.keys()) | set(self._plugin_classes.keys())
        return sorted(all_plugins)

    def unload(self, name: str):
        """
        Unload a plugin.

        Args:
            name: Plugin name
        """
        if name in self._plugins:
            plugin = self._plugins[name]
            plugin.teardown()
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")

        if name in self._plugin_classes:
            del self._plugin_classes[name]

    def load_plugins_from_directory(self, directory: str):
        """
        Load plugins from a directory.

        Plugin files should define a `plugin` variable containing
        the plugin instance or class.

        Args:
            directory: Path to plugins directory
        """
        plugins_dir = Path(directory)

        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {directory}")
            return

        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(
                    f"latent_gate.plugins.{module_name}", plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for plugin instance or class
                if hasattr(module, "plugin"):
                    plugin = module.plugin
                    if isinstance(plugin, ProcessorPlugin):
                        self.register(module_name, plugin)
                    elif isinstance(plugin, type) and issubclass(plugin, ProcessorPlugin):
                        self.register_class(module_name, plugin)

                # Also check for PLUGIN_NAME constant
                if hasattr(module, "PLUGIN_NAME"):
                    name = module.PLUGIN_NAME
                    if hasattr(module, "Plugin"):
                        self.register_class(name, module.Plugin)
                    elif hasattr(module, "plugin"):
                        self.register(name, module.plugin)

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

    def load_plugins_from_entry_points(self, group: str = "latent_gate.plugins"):
        """
        Load plugins from Python entry points.

        This allows plugins to be installed as separate packages.

        Args:
            group: Entry point group name
        """
        try:
            from importlib.metadata import entry_points

            eps = entry_points()
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group=group)
            else:
                plugin_eps = eps.get(group, [])

            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    if isinstance(plugin_class, type) and issubclass(plugin_class, ProcessorPlugin):
                        self.register_class(ep.name, plugin_class)
                        logger.info(f"Loaded plugin from entry point: {ep.name}")
                except Exception as e:
                    logger.error(f"Failed to load entry point {ep.name}: {e}")

        except ImportError:
            logger.debug("importlib.metadata not available, skipping entry points")


# ============================================================================
# Built-in Plugins
# ============================================================================


class DocumentPreProcessor(PreProcessorPlugin):
    """
    Pre-processor for document images.

    Applies document-specific preprocessing like:
    - Deskewing (rotation correction)
    - Noise removal (median blur, denoising)
    - Contrast enhancement (CLAHE, adaptive thresholding)
    - Binarization (Otsu's method)
    """

    name = "document_preprocessor"
    description = (
        "Pre-processor for document images with deskew, noise removal, and contrast enhancement"
    )

    def preprocess(self, data: Any) -> Any:
        """
        Preprocess a document image.

        Accepts either:
          - A path string to an image file
          - A numpy array (already loaded image)
          - A PIL Image

        Returns:
            Preprocessed image as numpy array ready for OCR or vision model.
        """
        if np is None:
            logger.warning(
                "DocumentPreProcessor requires numpy/opencv. "
                "Install with: pip install latent-gate[video]"
            )
            return data

        # --- Load image if path or PIL ---
        img = self._load_image(data)
        if img is None:
            logger.warning("DocumentPreProcessor: could not load image, returning as-is")
            return data

        logger.debug("DocumentPreProcessor: applying deskew, denoise, contrast enhancement")

        try:
            import cv2
        except ImportError:
            logger.warning(
                "DocumentPreProcessor requires opencv-python. "
                "Install with: pip install opencv-python"
            )
            # Fallback: convert to numpy array if possible
            if isinstance(data, str):
                from PIL import Image

                return np.array(Image.open(data).convert("L"))
            elif hasattr(data, "shape"):
                return data
            return data

        # 1. Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 2. Denoise (fast non-local means denoising)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, searchWindowSize=21, templateWindowSize=7)

        # 3. Deskew (find angle and rotate)
        coords = np.column_stack(np.where(denoised > 0))
        if len(coords) > 10:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                h, w = denoised.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                denoised = cv2.warpAffine(
                    denoised,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                logger.debug(f"Deskewed by {angle:.2f} degrees")

        # 4. Contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 5. Binarization via Otsu
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        logger.debug("DocumentPreProcessor: preprocessing complete")
        return binary

    def _load_image(self, data: Any):
        """Try to load image from various input types."""
        if isinstance(data, str):
            try:
                from PIL import Image

                pil_img = Image.open(data).convert("RGB")
                return np.array(pil_img)[:, :, ::-1]  # RGB → BGR for OpenCV
            except Exception:
                return None

        if isinstance(data, np.ndarray):
            return data

        # PIL Image
        try:
            from PIL import Image

            if isinstance(data, Image.Image):
                return np.array(data.convert("RGB"))[:, :, ::-1]
        except ImportError:
            pass

        return None

    def process(self, *args, **kwargs) -> SemanticPayload:
        """Required by base class but not used for pre-processors."""
        return SemanticPayload()


class TextEnhancerPostProcessor(PostProcessorPlugin):
    """
    Post-processor for text payloads.

    Enhances extracted semantics with:
    - Grammar correction (basic spelling normalization)
    - Key phrase extraction (TF-IDF-like frequency scoring)
    - Entity linking (capitalized proper noun detection + normalization)
    - Intent refinement (adds missing action verbs to intent)
    """

    name = "text_enhancer"
    description = "Post-processor for text enhancement with grammar correction, key phrase extraction, and entity linking"

    def postprocess(self, payload: SemanticPayload) -> SemanticPayload:
        """
        Post-process a SemanticPayload to enhance extracted text quality.

        Enhances:
          - scene_description: grammar correction
          - objects_detected: entity normalization
          - actions_activities: key phrase extraction
          - text_in_image: entity linking

        Returns:
            Enhanced SemanticPayload.
        """
        logger.debug("TextEnhancerPostProcessor: enhancing payload")

        # --- 1. Grammar correction on scene_description ---
        if payload.scene_description:
            payload.scene_description = self._correct_grammar(payload.scene_description)

        # --- 2. Entity normalization on objects_detected ---
        if payload.objects_detected:
            normalized = []
            for obj in payload.objects_detected:
                normalized.append(self._normalize_entity(str(obj)))
            # Deduplicate preserving order
            seen = set()
            payload.objects_detected = []
            for ent in normalized:
                key = ent.lower().strip()
                if key not in seen:
                    seen.add(key)
                    payload.objects_detected.append(ent)

        # --- 3. Key phrase extraction on actions_activities ---
        if payload.actions_activities:
            enhanced = []
            for action in payload.actions_activities:
                enhanced.append(self._extract_key_phrases(str(action)))
            payload.actions_activities = enhanced

        # --- 4. Entity linking on text_in_image ---
        if payload.text_in_image:
            payload.text_in_image = self._link_entities(payload.text_in_image)

        # --- 5. Intent refinement ---
        if not payload.scene_type and payload.scene_description:
            # Infer scene type from description
            desc_lower = payload.scene_description.lower()
            indoor_words = {"room", "kitchen", "office", "bedroom", "living", "bathroom", "indoor"}
            outdoor_words = {
                "street",
                "building",
                "park",
                "mountain",
                "beach",
                "sky",
                "outdoor",
                "landscape",
            }
            document_words = {"text", "document", "page", "letter", "form", "paper", "sign"}
            chart_words = {"chart", "graph", "plot", "diagram", "table", "data"}

            desc_set = set(desc_lower.split())
            if desc_set & indoor_words:
                payload.scene_type = "indoor"
            elif desc_set & outdoor_words:
                payload.scene_type = "outdoor"
            elif desc_set & document_words:
                payload.scene_type = "document"
            elif desc_set & chart_words:
                payload.scene_type = "chart"

        # --- 6. Fix empty confidence ---
        if payload.confidence == 0.0 and payload.scene_description:
            payload.confidence = 0.75

        logger.debug("TextEnhancerPostProcessor: enhancement complete")
        return payload

    @staticmethod
    def _correct_grammar(text: str) -> str:
        """
        Basic grammar correction without external dependencies.
        Fixes common issues like extra spaces, repeated words, capitalization.
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix repeated words (e.g., "the the" → "the")
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

        # Ensure first letter is capitalized
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Ensure sentence ends with period
        if text and text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def _normalize_entity(entity: str) -> str:
        """
        Normalize entity names: strip articles, fix capitalization.
        """
        entity = entity.strip()
        # Remove leading articles
        entity = re.sub(r"^(a|an|the)\s+", "", entity, flags=re.IGNORECASE)
        # Capitalize first letter of each significant word
        words = entity.split()
        if words:
            words[0] = words[0].capitalize()
            for i in range(1, len(words)):
                if len(words[i]) > 3:
                    words[i] = words[i].capitalize()
        return " ".join(words)

    @staticmethod
    def _extract_key_phrases(action: str) -> str:
        """
        Extract and consolidate key action phrases.
        Removes filler words while preserving core action.
        """
        filler_words = {
            "just",
            "very",
            "really",
            "quite",
            "some",
            "there",
            "that",
            "this",
            "these",
            "those",
            "then",
            "also",
            "too",
            "so",
            "well",
        }

        words = action.split()
        filtered = [w for w in words if w.lower() not in filler_words]

        if filtered:
            result = " ".join(filtered)
            # Remove trailing commas/punctuation
            result = re.sub(r"[,;:\s]+$", "", result)
            return result
        return action

    @staticmethod
    def _link_entities(text: str) -> str:
        """
        Basic entity linking: detect proper nouns (capitalized words)
        and normalize them. Uses word-boundary matching to avoid
        accidentally modifying substrings (e.g., "Apple" inside "Apple pie").
        """
        # Find potential entities (2+ consecutive capitalized words)
        entity_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        entities = entity_pattern.findall(text)

        # Deduplicate entities (case-insensitive)
        seen = {}
        for ent in entities:
            key = ent.lower()
            if key not in seen:
                seen[key] = ent

        # Mark entities with brackets using word-boundary replacement
        result = text
        for entity in entities:
            key = entity.lower()
            if seen.get(key) == entity:
                # Use word-boundary regex to avoid substring matches
                result = re.sub(
                    r"\b" + re.escape(entity) + r"\b",
                    f"[{entity}]",
                    result,
                    count=1,  # Only first occurrence
                )
                seen[key] = None  # Mark as processed

        return result


# ============================================================================
# Plugin Decorators
# ============================================================================


def plugin(name: str, description: str = "", version: str = "1.0.0"):
    """
    Decorator to register a plugin class.

    Usage:
        @plugin("my_processor", "My custom processor")
        class MyProcessor(ProcessorPlugin):
            def process(self, image_path, **kwargs):
                pass
    """

    def decorator(cls):
        cls.name = name
        cls.description = description
        cls.version = version
        return cls

    return decorator


# ============================================================================
# Global Plugin Manager
# ============================================================================

_global_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = PluginManager()
    return _global_manager


def register_plugin(name: str, plugin_instance: ProcessorPlugin):
    """Register a plugin with the global manager."""
    get_plugin_manager().register(name, plugin_instance)


def get_plugin(name: str) -> Optional[ProcessorPlugin]:
    """Get a plugin from the global manager."""
    return get_plugin_manager().get(name)
