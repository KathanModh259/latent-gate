"""
Configuration for the LatentGate pipeline.
All settings in one place — models, URLs, API keys, thresholds.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# Canonical default remote models per provider — import this instead of duplicating.
DEFAULT_REMOTE_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
    "ollama": "llama3:8b",
    "groq": "llama-3.3-70b-versatile",
    "deepseek": "deepseek-chat",
    "together": "meta-llama/Llama-3-70b-chat-hf",
    "azure": "gpt-4o-mini",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}


@dataclass
class PipelineConfig:
    """Central configuration for the LatentGate pipeline."""

    # ---- Local (Ollama) Settings ----
    ollama_base_url: str = "http://localhost:11434"

    # Vision model for X-Encoder stage (must be multimodal — uses llava:7b)
    vision_model: str = "llava:7b"

    # Text models for Predictor stage — smart model routing picks the best one
    # text_fast_model: lightweight model for short/normal text compression
    text_fast_model: str = "phi3:mini"
    # text_smart_model: higher-quality model for complex/long reasoning
    text_smart_model: str = "qwen2:7b"

    # Embedding model for RAG / vector search workflows
    embedding_model: str = "nomic-embed-text"

    # ---- Remote (Cloud LLM) Settings ----
    remote_provider: str = "openai"
    remote_api_key: str = ""
    remote_model: str = "gpt-4o-mini"
    remote_base_url: str = ""

    # ---- Pipeline Settings ----
    max_local_summary_tokens: int = 200
    enable_caching: bool = True
    cache_dir: str = ".latentgate_cache"
    log_level: str = "INFO"
    max_image_dimension: int = 1280
    max_concurrent_requests: int = 3
    allowed_image_roots: list = field(default_factory=list)

    # ---- Cost Tracking ----
    track_costs: bool = False
    cost_db_path: str = ".latentgate_costs.db"

    # ---- Offline-First Mode ----
    offline_first: bool = False
    offline_model: str = "llama3:8b"

    # ---- Adaptive Compression ----
    adaptive_compression: bool = False
    target_token_budget: int = 0

    # ---- Selective Decoding ----
    selective_decoding: bool = True
    similarity_threshold: float = 0.85
    use_embeddings: bool = True

    # ---- Ollama Generation Options ----
    temperature: float = 0.1
    request_timeout: int = 120

    # ---- Backward Compatibility ----
    # If predictor_model is set explicitly, it overrides text_fast_model
    # so existing configs using "llama3:8b" continue to work.
    predictor_model: Optional[str] = "llama3:8b"

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

        # If predictor_model was set (e.g. from legacy env var), use it as text_fast_model
        if self.predictor_model and self.predictor_model != "phi3:mini":
            self.text_fast_model = self.predictor_model

    def get_model_for_task(
        self,
        task: str,
        text: str = "",
        fallback_chain: Optional[list] = None,
    ) -> str:
        """
        Route to the best model for the given task.

        Args:
            task: "image" | "text_fast" | "text_smart" | "embedding"
            text: The input text (used to determine complexity for text tasks)
            fallback_chain: Optional list of model names to try in order

        Returns:
            Model name string
        """
        if task == "image":
            return self.vision_model

        if task == "embedding":
            return self.embedding_model

        if task == "text_smart":
            return self.text_smart_model

        # text_fast — determine complexity
        if task == "text_fast":
            if text and self._is_complex(text):
                return self.text_smart_model
            return self.text_fast_model

        # Default fallback
        return self.text_fast_model

    def get_fallback_chain(self, task: str) -> list:
        """
        Return the ordered fallback chain for a given task.

        Returns list of model names to try, e.g. ["phi3:mini", "llama3:8b"]
        """
        if task == "image":
            return [self.vision_model, "llama3.2-vision:11b", "llava:13b"]

        if task == "text_fast":
            return [self.text_fast_model, "llama3:8b"]

        if task == "text_smart":
            return [self.text_smart_model, "llama3:8b"]

        if task == "embedding":
            return [self.embedding_model, "all-minilm:33m"]

        return [self.text_fast_model, "llama3:8b"]

    @staticmethod
    def _is_complex(text: str) -> bool:
        """Determine if text is complex enough to warrant the smart model."""
        word_count = len(text.split())
        if word_count > 1500:
            return True
        complexity_indicators = [
            "```", "def ", "class ", "function ",
            "import ", "export ", "interface ",
            "constraints", "requirements", "architecture",
            "pipeline", "implementation", "deployment",
            "migration", "optimization", "benchmark",
        ]
        indicator_count = sum(1 for ind in complexity_indicators if ind in text)
        return indicator_count >= 3 or word_count > 800

    def validate(self) -> list:
        """Return list of validation warnings/errors (empty = all good)."""
        warnings = []
        if self.remote_provider != "ollama" and not self.remote_api_key:
            warnings.append(
                f"No API key set for '{self.remote_provider}'. "
                f"Set it via config or environment variable."
            )
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            warnings.append("similarity_threshold must be between 0.0 and 1.0")
        if self.temperature < 0 or self.temperature > 2.0:
            warnings.append("temperature must be between 0.0 and 2.0")
        if self.request_timeout < 1:
            warnings.append("request_timeout must be at least 1 second")
        if self.max_local_summary_tokens < 10:
            warnings.append("max_local_summary_tokens should be at least 10")
        if self.max_image_dimension < 256:
            warnings.append("max_image_dimension should be at least 256 pixels")
        if self.max_concurrent_requests < 1:
            warnings.append("max_concurrent_requests must be at least 1")
        if not self.ollama_base_url:
            warnings.append("ollama_base_url is empty")
        if self.ollama_base_url and not self.ollama_base_url.startswith(("http://", "https://")):
            warnings.append(
                f"ollama_base_url should start with http:// or https://, got: {self.ollama_base_url}"
            )
        return warnings
