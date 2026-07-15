"""
Payload Cache — Content-hash based caching to avoid redundant processing.

If the same image is processed twice, the second call returns instantly
from the local disk cache instead of re-running the Ollama pipeline.
"""

import json
import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from latent_gate.config import PipelineConfig
from latent_gate.payload import SemanticPayload

logger = logging.getLogger("latent_gate.cache")

_MAX_MEMORY_CACHE_SIZE = 500


class PayloadCache:
    """
    Disk-based cache for SemanticPayloads.

    Uses SHA-256 hash of image content as the cache key, so:
    - Same image file = cache hit (even if renamed)
    - Modified image = cache miss (correct behavior)
    """

    def __init__(self, config: PipelineConfig):
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: OrderedDict = OrderedDict()  # Bounded LRU for hot paths
        self._lock = threading.RLock()
        logger.info(f"Cache initialized at: {self.cache_dir}")

    def _get_cache_key(self, image_path: str) -> str:
        """Generate a cache key from the image file's content hash."""
        hasher = hashlib.sha256()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

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

        with self._lock:
            # Check in-memory cache first
            if cache_key in self._memory_cache:
                self._memory_cache.move_to_end(cache_key)
                logger.debug(f"Memory cache hit: {cache_key[:8]}...")
                return self._memory_cache[cache_key]

        # Check disk cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                payload = SemanticPayload.from_dict(data)
                with self._lock:
                    self._memory_cache[cache_key] = payload  # Promote to memory
                    self._memory_cache.move_to_end(cache_key)
                    # Evict oldest if over limit
                    while len(self._memory_cache) > _MAX_MEMORY_CACHE_SIZE:
                        self._memory_cache.popitem(last=False)
                logger.debug(f"Disk cache hit: {cache_key[:8]}...")
                return payload
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
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
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, "w") as f:
                json.dump(payload.to_dict(), f, indent=2)
        except (OSError, IOError) as e:
            logger.warning(f"Failed to write cache to disk: {e}")
            return

        # Store in memory
        with self._lock:
            self._memory_cache[cache_key] = payload
            self._memory_cache.move_to_end(cache_key)
            while len(self._memory_cache) > _MAX_MEMORY_CACHE_SIZE:
                self._memory_cache.popitem(last=False)
        logger.debug(f"Cached: {cache_key[:8]}... → {cache_path.name}")

    def clear(self):
        """Clear all cached entries (both memory and disk)."""
        with self._lock:
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        """Number of entries in disk cache."""
        with self._lock:
            return sum(1 for _ in self.cache_dir.glob("*.json"))
