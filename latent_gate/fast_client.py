"""
FastClient — Optimized HTTP client for Ollama with connection pooling,
model preloading, and keep_alive for maximum speed.

Key optimizations:
  1. Session reuse (connection pooling) — avoids TCP handshake per request
  2. Model preloading — warm up models on init so first call is fast
  3. keep_alive — tells Ollama to keep model in GPU memory between calls
  4. Timeouts tuned per operation type
"""

import logging
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from latent_gate.config import PipelineConfig


logger = logging.getLogger("latent_gate.client")


class FastClient:
    """
    Optimized HTTP client for Ollama and remote APIs.

    Uses a persistent requests.Session with connection pooling,
    retries, and model warm-up for maximum throughput.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # --- Persistent session with connection pooling ---
        self._ollama_session = self._create_session(
            pool_connections=4,
            pool_maxsize=8,
            retries=2,
        )
        self._remote_session = self._create_session(
            pool_connections=2,
            pool_maxsize=4,
            retries=3,
        )

        logger.info("FastClient initialized with connection pooling")

    @staticmethod
    def _create_session(
        pool_connections: int = 4,
        pool_maxsize: int = 8,
        retries: int = 2,
    ) -> requests.Session:
        """Create a session with connection pooling and retry logic."""
        session = requests.Session()

        retry_strategy = Retry(
            total=retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
        )

        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ----------------------------------------------------------------
    # Model Preloading
    # ----------------------------------------------------------------

    def preload_models(self):
        """
        Warm up Ollama models so they're loaded in GPU memory.

        Without this, the first request has a ~5-15s cold start
        while Ollama loads the model. With preloading, first
        request is as fast as subsequent ones.
        """
        models_to_load = set()
        models_to_load.add(self.config.vision_model)
        models_to_load.add(self.config.predictor_model)
        if self.config.remote_provider == "ollama":
            models_to_load.add(self.config.remote_model)

        for model in models_to_load:
            try:
                logger.info(f"Preloading model: {model}")
                self._ollama_session.post(
                    f"{self.config.ollama_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "keep_alive": "10m",   # Keep in memory for 10 min
                    },
                    timeout=60,
                )
            except Exception as e:
                logger.warning(f"Failed to preload {model}: {e}")

    # ----------------------------------------------------------------
    # Ollama Calls (Local)
    # ----------------------------------------------------------------

    def ollama_generate(
        self,
        model: str,
        prompt: str,
        images: list = None,
        max_tokens: int = 800,
    ) -> str:
        """
        Optimized Ollama generate call with session reuse + keep_alive.

        ~30-50% faster than creating a new connection each time.
        """
        url = f"{self.config.ollama_base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",            # Keep model loaded in GPU
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,             # Smaller context = faster
            },
        }
        if images:
            payload["images"] = images

        try:
            resp = self._ollama_session.post(
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

    # ----------------------------------------------------------------
    # Remote API Calls (Cloud)
    # ----------------------------------------------------------------

    def remote_post(
        self,
        url: str,
        headers: dict,
        json_payload: dict,
    ) -> dict:
        """Optimized remote API call with session reuse."""
        resp = self._remote_session.post(
            url,
            headers=headers,
            json=json_payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ----------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------

    def close(self):
        """Close all sessions and free connections."""
        self._ollama_session.close()
        self._remote_session.close()
        logger.info("FastClient sessions closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
