"""
Payload Cache — Content-hash based caching to avoid redundant processing.

If the same image is processed twice, the second call returns instantly
from the local disk cache instead of re-running the Ollama pipeline.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload


logger = logging.getLogger("latent_gate.cache")


class PayloadCache:
    """
    Disk-based cache for SemanticPayloads.

    Uses MD5 hash of image content as the cache key, so:
    - Same image file = cache hit (even if renamed)
    - Modified image = cache miss (correct behavior)
    """

    def __init__(self, config: PipelineConfig):
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict = {}  # In-memory LRU for hot paths
        logger.info(f"Cache initialized at: {self.cache_dir}")

    def _get_cache_key(self, image_path: str) -> str:
        """Generate a cache key from the image file's content hash."""
        with open(image_path, "rb") as f:
            content_hash = hashlib.md5(f.read()).hexdigest()
        return content_hash

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, image_path: str) -> Optional[SemanticPayload]:
        """
        Look up a cached payload for the given image.

        Args:
            image_path: Path to the image file.

        Returns:
            SemanticPayload if cached, None if cache miss.
        """
        try:
            cache_key = self._get_cache_key(image_path)
        except FileNotFoundError:
            return None

        # Check in-memory cache first
        if cache_key in self._memory_cache:
            logger.debug(f"Memory cache hit: {cache_key[:8]}...")
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                payload = SemanticPayload.from_dict(data)
                self._memory_cache[cache_key] = payload  # Promote to memory
                logger.debug(f"Disk cache hit: {cache_key[:8]}...")
                return payload
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt cache entry {cache_key[:8]}: {e}")
                cache_path.unlink(missing_ok=True)

        return None

    def put(self, image_path: str, payload: SemanticPayload):
        """
        Store a payload in the cache.

        Args:
            image_path: Path to the original image file.
            payload: The SemanticPayload to cache.
        """
        try:
            cache_key = self._get_cache_key(image_path)
        except FileNotFoundError:
            logger.warning(f"Cannot cache — image not found: {image_path}")
            return

        # Write to disk
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, "w") as f:
            json.dump(payload.to_dict(), f, indent=2)

        # Store in memory
        self._memory_cache[cache_key] = payload
        logger.debug(f"Cached: {cache_key[:8]}... → {cache_path.name}")

    def clear(self):
        """Clear all cached entries (both memory and disk)."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        """Number of entries in disk cache."""
        return len(list(self.cache_dir.glob("*.json")))
