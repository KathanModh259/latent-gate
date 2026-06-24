"""
Async Pipeline — Async versions of core LatentGate methods.

Provides non-blocking versions of the pipeline for use in async applications.
Uses asyncio.to_thread() to run synchronous operations in a thread pool.

Features:
  - Async versions of all query methods
  - Concurrent processing for multiple queries
  - Integration with async frameworks (FastAPI, aiohttp, etc.)
"""

import asyncio
import logging
from typing import Optional, List

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline

logger = logging.getLogger("latent_gate.async")


class AsyncLatentGatePipeline:
    """
    Async wrapper for LatentGatePipeline.

    Usage:
        async with AsyncLatentGatePipeline(config) as pipeline:
            result = await pipeline.query("photo.jpg", "What is this?")
            result = await pipeline.query_text("Long prompt...")
    """

    def __init__(self, config: Optional[PipelineConfig] = None, preload: bool = True):
        self.config = config or PipelineConfig()
        self._pipeline: Optional[LatentGatePipeline] = None
        self._preload = preload

    async def __aenter__(self):
        await self._init_pipeline()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def _init_pipeline(self):
        """Initialize the pipeline in a thread."""

        def _init():
            return LatentGatePipeline(self.config, preload=self._preload)

        self._pipeline = await asyncio.to_thread(_init)

    async def close(self):
        """Clean up resources."""
        if self._pipeline:
            await asyncio.to_thread(self._pipeline.close)
            self._pipeline = None

    @property
    def pipeline(self) -> LatentGatePipeline:
        """Get the underlying pipeline."""
        if not self._pipeline:
            raise RuntimeError(
                "Pipeline not initialized. Use 'async with' or call _init_pipeline()"
            )
        return self._pipeline

    # ================================================================
    # Async Query Methods
    # ================================================================

    async def query(self, image_path: str, question: str) -> dict:
        """
        Process an image and answer a question about it (async).

        Args:
            image_path: Path to the image file.
            question: Question about the image.

        Returns:
            Dictionary with answer and metadata.
        """
        return await asyncio.to_thread(self.pipeline.query, image_path, question)

    async def query_text(self, text: str, question: str = "", mode: str = "auto") -> dict:
        """
        Compress text and query the remote LLM (async).

        Args:
            text: Text to compress and query.
            question: Specific question about the text.
            mode: Compression mode.

        Returns:
            Dictionary with answer and metadata.
        """
        return await asyncio.to_thread(self.pipeline.query_text, text, question, mode)

    async def query_conversation(self, messages: list, new_question: str) -> dict:
        """
        Compress conversation history and ask a new question (async).

        Args:
            messages: Conversation messages [{role, content}].
            new_question: New question to ask.

        Returns:
            Dictionary with answer and metadata.
        """
        return await asyncio.to_thread(self.pipeline.query_conversation, messages, new_question)

    async def query_documents(self, documents: list, question: str) -> dict:
        """
        Compress RAG documents and answer a question (async).

        Args:
            documents: List of document strings.
            question: Question about the documents.

        Returns:
            Dictionary with answer and metadata.
        """
        return await asyncio.to_thread(self.pipeline.query_documents, documents, question)

    async def query_universal(self, text: str = "", image: str = "", question: str = "") -> dict:
        """
        Universal query - auto-detects input type (async).

        Args:
            text: Text input.
            image: Image path.
            question: Question.

        Returns:
            Dictionary with answer and metadata.
        """
        return await asyncio.to_thread(self.pipeline.query_universal, text, image, question)

    async def query_batch(
        self, image_paths: list, question: str, parallel: bool = False, max_workers: int = 3
    ) -> list:
        """
        Process multiple images with selective decoding (async).

        Args:
            image_paths: List of image file paths.
            question: Question to ask about each image.
            parallel: If True, process images in parallel.
            max_workers: Maximum parallel workers.

        Returns:
            List of result dictionaries.
        """
        return await asyncio.to_thread(
            self.pipeline.query_batch, image_paths, question, parallel, max_workers
        )

    # ================================================================
    # Concurrent Processing Methods
    # ================================================================

    async def query_many_images(
        self,
        image_paths: List[str],
        question: str,
        max_concurrent: int = 3,
    ) -> List[dict]:
        """
        Process multiple images concurrently.

        Args:
            image_paths: List of image file paths.
            question: Question to ask about each image.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of result dictionaries.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_one(path: str) -> dict:
            async with semaphore:
                return await self.query(path, question)

        tasks = [_process_one(path) for path in image_paths]
        return await asyncio.gather(*tasks)

    async def query_many_texts(
        self,
        texts: List[str],
        question: str = "",
        max_concurrent: int = 3,
    ) -> List[dict]:
        """
        Process multiple texts concurrently.

        Args:
            texts: List of text strings.
            question: Question to ask about each text.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of result dictionaries.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_one(text: str) -> dict:
            async with semaphore:
                return await self.query_text(text, question)

        tasks = [_process_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    # ================================================================
    # Utility Methods
    # ================================================================

    def reset_selective_decoder(self):
        """Reset the selective decoder state."""
        self.pipeline.reset_selective_decoder()

    @property
    def stats(self) -> dict:
        """Get selective decoding statistics."""
        return self.pipeline.selective_decoder.stats


# ============================================================================
# Convenience Functions
# ============================================================================


async def async_query(
    image_path: str,
    question: str,
    config: Optional[PipelineConfig] = None,
) -> dict:
    """
    Convenience function for async image query.

    Args:
        image_path: Path to the image file.
        question: Question about the image.
        config: Optional pipeline configuration.

    Returns:
        Dictionary with answer and metadata.
    """
    async with AsyncLatentGatePipeline(config) as pipeline:
        return await pipeline.query(image_path, question)


async def async_query_text(
    text: str,
    question: str = "",
    config: Optional[PipelineConfig] = None,
) -> dict:
    """
    Convenience function for async text query.

    Args:
        text: Text to compress and query.
        question: Specific question about the text.
        config: Optional pipeline configuration.

    Returns:
        Dictionary with answer and metadata.
    """
    async with AsyncLatentGatePipeline(config) as pipeline:
        return await pipeline.query_text(text, question)


async def async_query_batch(
    image_paths: List[str],
    question: str,
    config: Optional[PipelineConfig] = None,
) -> List[dict]:
    """
    Convenience function for async batch processing.

    Args:
        image_paths: List of image file paths.
        question: Question to ask about each image.
        config: Optional pipeline configuration.

    Returns:
        List of result dictionaries.
    """
    async with AsyncLatentGatePipeline(config) as pipeline:
        return await pipeline.query_batch(image_paths, question)
