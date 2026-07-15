"""
Configuration Loader — YAML/TOML config file support.

Loads and saves pipeline configuration from files.
Supports YAML, TOML, and JSON formats.

Features:
  - Load config from file
  - Save config to file
  - Environment variable overrides
  - Config validation
  - Default config generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from latent_gate.config import PipelineConfig

logger = logging.getLogger("latent_gate.config")


# ============================================================================
# Format Detection and Loading
# ============================================================================


def load_config(
    filepath: str,
    env_prefix: str = "LATENTGATE_",
) -> PipelineConfig:
    """
    Load configuration from a file.

    Args:
        filepath: Path to config file (YAML, TOML, or JSON)
        env_prefix: Environment variable prefix for overrides

    Returns:
        PipelineConfig instance
    """
    path = Path(filepath)

    if not path.exists():
        logger.warning(f"Config file not found: {filepath}, using defaults")
        return PipelineConfig()

    # Detect format from extension
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        data = _load_yaml(path)
    elif suffix == ".toml":
        data = _load_toml(path)
    elif suffix == ".json":
        data = _load_json(path)
    else:
        # Try JSON as default
        data = _load_json(path)

    # Apply environment variable overrides
    data = _apply_env_overrides(data, env_prefix)

    # Create config from data
    config = _dict_to_config(data)

    logger.info(f"Loaded config from {filepath}")
    return config


def save_config(
    config: PipelineConfig,
    filepath: str,
    fmt: Optional[str] = None,
):
    """
    Save configuration to a file.

    Args:
        config: PipelineConfig instance
        filepath: Output file path
        fmt: Force format (yaml, toml, json). Auto-detected from extension if None.
    """
    path = Path(filepath)

    # Convert config to dict
    data = _config_to_dict(config)

    # Detect format
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            fmt = "yaml"
        elif suffix == ".toml":
            fmt = "toml"
        else:
            fmt = "json"

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    if fmt == "yaml":
        _save_yaml(path, data)
    elif fmt == "toml":
        _save_toml(path, data)
    else:
        _save_json(path, data)

    logger.info(f"Saved config to {filepath}")


# ============================================================================
# Format Loaders
# ============================================================================


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config files. " "Install with: pip install pyyaml"
        )


def _load_toml(path: Path) -> Dict[str, Any]:
    """Load TOML config file."""
    try:
        import tomli

        with open(path, "rb") as f:
            return tomli.load(f)
    except ImportError:
        # Python 3.11+ has tomllib in stdlib
        try:
            import tomllib

            with open(path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            raise ImportError(
                "tomli is required for TOML config files (Python < 3.11). "
                "Install with: pip install tomli"
            )


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON config file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {path}: {e}")


# ============================================================================
# Format Savers
# ============================================================================


def _save_yaml(path: Path, data: Dict[str, Any]):
    """Save YAML config file."""
    try:
        import yaml

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        raise ImportError("PyYAML is required for YAML config files")


def _save_toml(path: Path, data: Dict[str, Any]):
    """Save TOML config file."""
    try:
        import tomli_w

        with open(path, "wb") as f:
            tomli_w.dump(data, f)
    except ImportError:
        raise ImportError(
            "tomli_w is required for writing TOML files. " "Install with: pip install tomli_w"
        )


def _save_json(path: Path, data: Dict[str, Any]):
    """Save JSON config file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Environment Variable Overrides
# ============================================================================


def _apply_env_overrides(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config data.

    Environment variables are mapped as:
        LATENTGATE_VISION_MODEL -> vision_model
        LATENTGATE_REMOTE_PROVIDER -> remote_provider
        etc.
    """
    env_mapping = {
        f"{prefix}OLLAMA_BASE_URL": "ollama_base_url",
        f"{prefix}VISION_MODEL": "vision_model",
        f"{prefix}TEXT_FAST_MODEL": "text_fast_model",
        f"{prefix}TEXT_SMART_MODEL": "text_smart_model",
        f"{prefix}EMBEDDING_MODEL": "embedding_model",
        f"{prefix}PREDICTOR_MODEL": "predictor_model",
        f"{prefix}REMOTE_PROVIDER": "remote_provider",
        f"{prefix}REMOTE_API_KEY": "remote_api_key",
        f"{prefix}REMOTE_MODEL": "remote_model",
        f"{prefix}REMOTE_BASE_URL": "remote_base_url",
        f"{prefix}MAX_LOCAL_SUMMARY_TOKENS": ("max_local_summary_tokens", int),
        f"{prefix}ENABLE_CACHING": ("enable_caching", lambda x: x.lower() in ("true", "1", "yes")),
        f"{prefix}CACHE_DIR": "cache_dir",
        f"{prefix}LOG_LEVEL": "log_level",
        f"{prefix}OFFLINE_FIRST": (
            "offline_first",
            lambda x: x.lower() in ("true", "1", "yes"),
        ),
        f"{prefix}OFFLINE_MODEL": "offline_model",
        f"{prefix}ADAPTIVE_COMPRESSION": (
            "adaptive_compression",
            lambda x: x.lower() in ("true", "1", "yes"),
        ),
        f"{prefix}TARGET_TOKEN_BUDGET": ("target_token_budget", int),
        f"{prefix}SELECTIVE_DECODING": (
            "selective_decoding",
            lambda x: x.lower() in ("true", "1", "yes"),
        ),
        f"{prefix}SIMILARITY_THRESHOLD": ("similarity_threshold", float),
        f"{prefix}USE_EMBEDDINGS": ("use_embeddings", lambda x: x.lower() in ("true", "1", "yes")),
        f"{prefix}TEMPERATURE": ("temperature", float),
        f"{prefix}REQUEST_TIMEOUT": ("request_timeout", int),
    }

    for env_var, mapping in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            if isinstance(mapping, tuple):
                field_name, converter = mapping
                try:
                    data[field_name] = converter(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid env var {env_var}={value}: {e}")
            else:
                data[mapping] = value

    return data


# ============================================================================
# Config Conversion
# ============================================================================


def _config_to_dict(config: PipelineConfig) -> Dict[str, Any]:
    """Convert PipelineConfig to dictionary (excludes secrets)."""
    return {
        "ollama_base_url": config.ollama_base_url,
        "vision_model": config.vision_model,
        "text_fast_model": config.text_fast_model,
        "text_smart_model": config.text_smart_model,
        "embedding_model": config.embedding_model,
        "remote_provider": config.remote_provider,
        "remote_model": config.remote_model,
        "remote_base_url": config.remote_base_url,
        "max_local_summary_tokens": config.max_local_summary_tokens,
        "enable_caching": config.enable_caching,
        "cache_dir": config.cache_dir,
        "log_level": config.log_level,
        "max_image_dimension": config.max_image_dimension,
        "max_concurrent_requests": config.max_concurrent_requests,
        "selective_decoding": config.selective_decoding,
        "similarity_threshold": config.similarity_threshold,
        "use_embeddings": config.use_embeddings,
        "temperature": config.temperature,
        "request_timeout": config.request_timeout,
        "offline_first": config.offline_first,
        "offline_model": config.offline_model,
        "adaptive_compression": config.adaptive_compression,
        "target_token_budget": config.target_token_budget,
    }


def _dict_to_config(data: Dict[str, Any]) -> PipelineConfig:
    """Convert dictionary to PipelineConfig."""
    # Filter to only valid fields
    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k in valid_fields:
            filtered[k] = v
        else:
            logger.warning(f"Unknown config key '{k}' — ignored (typo?)")

    return PipelineConfig(**filtered)


# ============================================================================
# Default Config Generation
# ============================================================================


def generate_default_config(filepath: str = "latentgate.yaml"):
    """
    Generate a default configuration file.

    Args:
        filepath: Output file path
    """
    config = PipelineConfig()
    save_config(config, filepath)
    logger.info(f"Generated default config at {filepath}")


# ============================================================================
# Convenience Functions
# ============================================================================


def get_config(
    config_file: Optional[str] = None,
    env_prefix: str = "LATENTGATE_",
) -> PipelineConfig:
    """
    Get configuration from file or environment.

    Args:
        config_file: Optional config file path
        env_prefix: Environment variable prefix

    Returns:
        PipelineConfig instance
    """
    # Check standard locations
    if config_file is None:
        standard_locations = [
            "latentgate.yaml",
            "latentgate.yml",
            "latentgate.toml",
            "latentgate.json",
            ".latentgate.yaml",
            ".latentgate.yml",
            ".latentgate.toml",
            ".latentgate.json",
            os.path.expanduser("~/.config/latentgate/config.yaml"),
            os.path.expanduser("~/.config/latentgate/config.toml"),
        ]

        for location in standard_locations:
            if os.path.exists(location):
                config_file = location
                break

    if config_file and os.path.exists(config_file):
        return load_config(config_file, env_prefix)

    # Apply env overrides to default config
    config = PipelineConfig()
    config_dict = _config_to_dict(config)

    # Apply env overrides with proper type conversion
    config_dict = _apply_env_overrides(config_dict, env_prefix)
    config = _dict_to_config(config_dict)

    return config
