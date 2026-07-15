"""
Video Processor — Direct video file input with automatic frame extraction.

Extends LatentGate beyond video streams to handle video files directly.
Automatically extracts frames at configurable FPS and processes them
with selective decoding for optimal API usage.

Features:
  - Automatic frame extraction from video files
  - Configurable FPS and frame quality
  - Batch processing with selective decoding
  - Progress callbacks for long videos
  - Temporary file cleanup
"""

import os
import logging
import tempfile
import time
from typing import Optional, Callable, List
from dataclasses import dataclass

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline

logger = logging.getLogger("latent_gate.video")


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    fps: float = 1.0  # Frames per second to extract
    max_frames: int = 100  # Maximum frames to process
    quality: int = 95  # JPEG quality (1-100)
    start_time: float = 0.0  # Start time in seconds
    end_time: Optional[float] = None  # End time in seconds (None = end of video)
    resize_width: Optional[int] = None  # Resize frames to this width (None = keep original)


class VideoProcessor:
    """
    Process video files with automatic frame extraction and selective decoding.

    Usage:
        processor = VideoProcessor(config)
        results = processor.process_video("video.mp4", "Describe the action")
    """

    def __init__(
        self, config: Optional[PipelineConfig] = None, video_config: Optional[VideoConfig] = None
    ):
        from latent_gate.config_loader import get_config
        self.config = config or get_config()
        self.video_config = video_config or VideoConfig()
        self.pipeline = LatentGatePipeline(self.config, preload=True)

    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save extracted frames.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            List of extracted frame file paths.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video processing. "
                "Install with: pip install latent-gate[video]"
            )

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        logger.info(
            f"Video: {video_fps:.1f} FPS, {total_frames} frames, " f"{duration:.1f}s duration"
        )

        # Calculate frame interval
        frame_interval = int(video_fps / self.video_config.fps) if video_fps > 0 else 30

        # Calculate start/end frames
        start_frame = (
            int(self.video_config.start_time * video_fps) if self.video_config.start_time > 0 else 0
        )
        end_frame = (
            int(self.video_config.end_time * video_fps)
            if self.video_config.end_time
            else total_frames
        )
        end_frame = min(end_frame, total_frames)

        # Limit max frames
        if self.video_config.max_frames:
            max_end_frame = start_frame + (self.video_config.max_frames * frame_interval)
            end_frame = min(end_frame, max_end_frame)

        frame_paths = []
        frame_count = 0
        saved_count = 0

        # Seek to start position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Resize if configured
                if self.video_config.resize_width:
                    height, width = frame.shape[:2]
                    new_width = self.video_config.resize_width
                    new_height = int(height * (new_width / width))
                    frame = cv2.resize(frame, (new_width, new_height))

                # Save frame
                path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.video_config.quality])
                frame_paths.append(path)
                saved_count += 1

                # Progress callback
                if progress_callback and saved_count % 10 == 0:
                    progress_callback(saved_count, (end_frame - start_frame) // frame_interval)

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {saved_count} frames")

        return frame_paths

    def process_video(
        self,
        video_path: str,
        question: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Process a video file with automatic frame extraction and selective decoding.

        Args:
            video_path: Path to the video file.
            question: Question to ask about each frame.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            Dictionary with results and statistics.
        """
        start_time = time.time()

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract frames
            logger.info(f"Extracting frames from {video_path}")
            frame_paths = self.extract_frames(video_path, tmpdir, progress_callback)

            if not frame_paths:
                raise ValueError("No frames extracted from video")

            # Process frames with selective decoding
            logger.info(f"Processing {len(frame_paths)} frames with selective decoding")
            results = self.pipeline.query_batch(frame_paths, question)

            # Calculate statistics
            total_time = time.time() - start_time
            stats = results[-1]["selective_decoding_stats"] if results else {}

            # Build response
            response = {
                "video_path": video_path,
                "question": question,
                "total_frames": len(frame_paths),
                "results": results,
                "statistics": stats,
                "timing": {
                    "extraction_ms": 0,  # TODO: Track extraction time
                    "processing_ms": sum(r["timing"]["total_ms"] for r in results),
                    "total_ms": round(total_time * 1000, 1),
                },
                "summary": self._generate_summary(results),
            }

            return response

    def process_video_frames(
        self,
        frame_paths: List[str],
        question: str,
    ) -> dict:
        """
        Process pre-extracted video frames.

        Args:
            frame_paths: List of frame file paths.
            question: Question to ask about each frame.

        Returns:
            Dictionary with results and statistics.
        """
        start_time = time.time()

        # Process frames with selective decoding
        results = self.pipeline.query_batch(frame_paths, question)

        # Calculate statistics
        total_time = time.time() - start_time
        stats = results[-1]["selective_decoding_stats"] if results else {}

        return {
            "total_frames": len(frame_paths),
            "results": results,
            "statistics": stats,
            "timing": {
                "processing_ms": sum(r["timing"]["total_ms"] for r in results),
                "total_ms": round(total_time * 1000, 1),
            },
            "summary": self._generate_summary(results),
        }

    def _generate_summary(self, results: List[dict]) -> dict:
        """Generate a summary of video processing results."""
        if not results:
            return {}

        decoded_results = [r for r in results if not r.get("was_cached", False)]

        return {
            "unique_scenes": len(decoded_results),
            "total_frames": len(results),
            "skip_rate": results[-1].get("selective_decoding_stats", {}).get("skip_rate", "0%"),
            "average_tokens": sum(r["tokens_estimated"] for r in decoded_results)
            / max(len(decoded_results), 1),
        }

    def close(self):
        """Clean up resources."""
        self.pipeline.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# Convenience Functions
# ============================================================================


def process_video_file(
    video_path: str,
    question: str,
    config: Optional[PipelineConfig] = None,
    video_config: Optional[VideoConfig] = None,
    **kwargs,
) -> dict:
    """
    Convenience function to process a video file.

    Args:
        video_path: Path to the video file.
        question: Question to ask about each frame.
        config: Optional pipeline configuration.
        video_config: Optional video configuration.

    Returns:
        Dictionary with results and statistics.
    """
    processor = VideoProcessor(config, video_config)
    try:
        return processor.process_video(video_path, question, **kwargs)
    finally:
        processor.close()
