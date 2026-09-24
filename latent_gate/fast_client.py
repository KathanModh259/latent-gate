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
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from latent_gate.config import PipelineConfig

logger = logging.getLogger("latent_gate.client")


class OllamaUnavailableError(ConnectionError):
    """The Ollama server itself is unreachable (as opposed to one model failing)."""


def is_model_installed(model: str, installed: List[str]) -> bool:
    """
    Return True if `model` (as named in config) is among Ollama's installed models.

    Ollama reports untagged pulls with an implicit tag (``nomic-embed-text`` is listed
    as ``nomic-embed-text:latest``), so an untagged name matches its ``:latest`` form.
    """

    def normalize(name: str) -> str:
        return name if ":" in name else f"{name}:latest"

    return normalize(model) in {normalize(n) for n in installed}


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

    def list_local_models(self, timeout: float = 2.0) -> Optional[List[str]]:
        """
        Return model names installed in Ollama, or None if Ollama is unreachable.

        Uses a bare requests call (no retry adapter) so an offline Ollama is
        detected in ~timeout seconds instead of retries x backoff.
        """
        try:
            resp = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=timeout)
            resp.raise_for_status()
            return [m.get("name", "") for m in resp.json().get("models", [])]
        except (requests.RequestException, ValueError):
            return None

    def preload_models(self) -> List[str]:
        """
        Warm up installed Ollama models so they're resident in GPU memory.

        Returns the list of models that were successfully warmed.
        """
        wanted = {
            self.config.vision_model,
            self.config.text_fast_model,
            self.config.text_smart_model,
        }
        if self.config.offline_first and self.config.offline_model:
            wanted.add(self.config.offline_model)
        if self.config.remote_provider == "ollama":
            wanted.add(self.config.remote_model)
        wanted.discard("")

        installed = self.list_local_models()
        if installed is None:
            logger.warning(
                f"Ollama not reachable at {self.config.ollama_base_url} - skipping model "
                "preload. Start it with `ollama serve`."
            )
            return []

        to_load = []
        for model in sorted(wanted):
            if is_model_installed(model, installed):
                to_load.append(model)
            else:
                logger.warning(f"Model '{model}' is not pulled. Run: ollama pull {model}")

        warmed = []
        for model in to_load:
            try:
                logger.info(f"Preloading model: {model}")
                resp = self._ollama_session.post(
                    f"{self.config.ollama_base_url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": "10m"},
                    timeout=60,
                )
                resp.raise_for_status()
                warmed.append(model)
            except Exception as e:
                logger.warning(f"Failed to preload {model}: {e}")
        return warmed

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
            "keep_alive": "10m",  # Keep model loaded in GPU
            "options": {
                "temperature": self.config.temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,  # Smaller context = faster
            },
        }
        if images:
            payload["images"] = images

        try:
            resp = self._ollama_session.post(url, json=payload, timeout=self.config.request_timeout)
            resp.raise_for_status()
            return resp.json().get("response", "")

        except requests.exceptions.ConnectionError:
            raise OllamaUnavailableError(
                "Cannot connect to Ollama. Make sure it's running:\n"
                "  Start:  ollama serve\n"
                "  Check:  curl http://localhost:11434/api/tags"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.config.request_timeout}s. "
                "The model may be loading or the request is too large."
            )
        except (requests.exceptions.HTTPError, requests.exceptions.RetryError) as e:
            # Mapped to ConnectionError so callers' model fallback chains engage
            # (e.g. 404 when the model hasn't been pulled).
            raise ConnectionError(
                f"Ollama HTTP error for model '{model}': {e}. "
                f"If the model is missing, run: ollama pull {model}"
            ) from e

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
        try:
            resp = self._remote_session.post(
                url,
                headers=headers,
                json=json_payload,
                timeout=self.config.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to remote API at {url}. "
                "Check your network connection and API endpoint."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request to {url} timed out after {self.config.request_timeout}s.")

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "unknown")
            raise ConnectionError(f"Rate limited by API (429). Retry after {retry_after}s.")
        if resp.status_code == 401:
            raise PermissionError("API authentication failed (401). Check your API key.")
        if resp.status_code == 403:
            raise PermissionError("API access denied (403). Check your API key permissions.")

        resp.raise_for_status()

        try:
            return resp.json()
        except ValueError:
            raise ValueError(
                f"API returned non-JSON response (status {resp.status_code}): " f"{resp.text[:200]}"
            )

    def post_stream(
        self,
        url: str,
        headers: dict,
        json_payload: dict,
    ):
        """Optimized streaming remote API call with session reuse."""
        try:
            resp = self._remote_session.post(
                url,
                headers=headers,
                json=json_payload,
                stream=True,
                timeout=self.config.request_timeout,
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to streaming API at {url}. "
                "Check your network connection and API endpoint."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Streaming request to {url} timed out after {self.config.request_timeout}s."
            )

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "unknown")
            raise ConnectionError(
                f"Rate limited by streaming API (429). Retry after {retry_after}s."
            )

        resp.raise_for_status()
        return resp

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
