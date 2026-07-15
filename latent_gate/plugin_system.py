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
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Type
from pathlib import Path

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
    - Deskewing
    - Noise removal
    - Contrast enhancement
    """

    name = "document_preprocessor"
    description = "Pre-processor for document images"

    def preprocess(self, data: Any) -> Any:
        """Preprocess document image."""
        # TODO: Implement document preprocessing
        return data

    def process(self, *args, **kwargs) -> SemanticPayload:
        """Required by base class but not used for pre-processors."""
        return SemanticPayload()


class TextEnhancerPostProcessor(PostProcessorPlugin):
    """
    Post-processor for text compression results.

    Enhances extracted text with:
    - Grammar correction
    - Key phrase extraction
    - Entity linking
    """

    name = "text_enhancer"
    description = "Post-processor for text enhancement"

    def postprocess(self, payload: SemanticPayload) -> SemanticPayload:
        """Post-process text payload."""
        # TODO: Implement text enhancement
        return payload

    def process(self, *args, **kwargs) -> SemanticPayload:
        """Required by base class but not used for post-processors."""
        return SemanticPayload()


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
