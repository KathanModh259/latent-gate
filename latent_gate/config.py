"""
Configuration for the LatentGate pipeline.
All settings in one place — models, URLs, API keys, thresholds.
"""

import os
from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Central configuration for the LatentGate pipeline."""

    # ---- Local (Ollama) Settings ----
    ollama_base_url: str = "http://localhost:11434"

    # Vision model for X-Encoder stage (must be multimodal)
    vision_model: str = "llava:7b"

    # Text model for Predictor stage (structures raw extraction)
    predictor_model: str = "llama3:8b"

    # ---- Remote (Cloud LLM) Settings ----
    remote_provider: str = "openai"  # "openai" | "anthropic" | "ollama" | "custom"
    remote_api_key: str = ""  # Set via env var or direct
    remote_model: str = "gpt-4o-mini"  # Cheaper model — input is already pre-processed
    remote_base_url: str = ""  # For custom endpoints (e.g., Azure, Together AI)

    # ---- Pipeline Settings ----
    max_local_summary_tokens: int = 200  # Cap on local summary size
    enable_caching: bool = True  # Cache local extractions to disk
    cache_dir: str = ".latentgate_cache"
    log_level: str = "INFO"

    # ---- Offline-First Mode ----
    offline_first: bool = False  # Use local Ollama for final answer when no API key
    offline_model: str = "llama3:8b"  # Model to use for offline answering

    # ---- Adaptive Compression ----
    adaptive_compression: bool = False  # Dynamically adjust compression based on complexity
    target_token_budget: int = 0  # Target output tokens (0 = auto-detect)

    # ---- Selective Decoding ----
    selective_decoding: bool = True  # Only call remote if semantics changed
    similarity_threshold: float = 0.85  # How similar = "unchanged" (0.0 to 1.0)
    use_embeddings: bool = True  # Use cosine similarity (requires sentence-transformers)

    # ---- Ollama Generation Options ----
    temperature: float = 0.1  # Low temp for factual extraction
    request_timeout: int = 120  # Seconds before timeout

    def __post_init__(self):
        """Load API key from environment if not set directly."""
        if not self.remote_api_key:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
                "groq": "GROQ_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "together": "TOGETHER_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
                "bedrock": "AWS_ACCESS_KEY_ID",
            }
            env_var = env_map.get(self.remote_provider, "")
            if env_var:
                self.remote_api_key = os.getenv(env_var, "")

    def validate(self) -> list:
        """Return list of validation warnings (empty = all good)."""
        warnings = []
        if self.remote_provider != "ollama" and not self.remote_api_key:
            warnings.append(
                f"No API key set for '{self.remote_provider}'. "
                f"Set it via config or environment variable."
            )
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            warnings.append("similarity_threshold must be between 0.0 and 1.0")
        return warnings
