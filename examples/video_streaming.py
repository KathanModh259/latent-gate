"""
Example: Video Streaming with Selective Decoding
==================================================
Process video frames and only call the API when the scene changes.
Achieves ~2.85x reduction in API calls (matching VL-JEPA's findings).

Prerequisites:
    ollama pull llava:7b
    ollama pull llama3:8b
    pip install opencv-python  (for frame extraction)
"""

import os
import glob
import tempfile
from pathlib import Path

from latent_gate import LatentGatePipeline, PipelineConfig


def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> list:
    """Extract frames from a video file at the given FPS."""
    try:
        import cv2
    except ImportError:
        print("Install opencv-python: pip install opencv-python")
        return []

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 30

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames at {fps} FPS")
    return frame_paths


def main():
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        # KEY: Enable selective decoding
        selective_decoding=True,
        similarity_threshold=0.85,
        enable_caching=True,
        log_level="WARNING",
    )

    pipeline = LatentGatePipeline(config)

    # Option 1: Use pre-extracted frames
    frame_paths = sorted(glob.glob("video_frames/*.jpg"))

    # Option 2: Extract frames from a video file
    if not frame_paths:
        video_file = "video.mp4"
        if os.path.exists(video_file):
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_paths = extract_frames(video_file, tmpdir, fps=1)
        else:
            print("No frames found. Place frames in video_frames/ or provide video.mp4")
            return

    # Process all frames
    results = pipeline.query_batch(
        image_paths=frame_paths,
        question="Describe the current action being performed."
    )

    # Print results
    print("\n" + "=" * 60)
    print("SELECTIVE DECODING RESULTS")
    print("=" * 60)

    stats = results[-1]["selective_decoding_stats"]
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"API calls made:        {stats['api_calls']}")
    print(f"Frames skipped:        {stats['skipped']}")
    print(f"Cost reduction:        {stats['reduction_ratio']}")
    print(f"Skip rate:             {stats['skip_rate']}")

    print("\nFrame-by-frame:")
    for i, r in enumerate(results):
        status = "SKIPPED" if r["was_cached"] else "DECODED"
        answer_preview = r["answer"][:60] + "..." if len(r["answer"]) > 60 else r["answer"]
        print(f"  Frame {i+1}: [{status}] {answer_preview}")


if __name__ == "__main__":
    main()
