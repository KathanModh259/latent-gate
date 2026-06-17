#!/usr/bin/env python3
"""
Compress an image to a compact semantic payload for Claude.

Usage:
    python compress_image.py <image_path> [question]

Output:
    JSON with compact_payload, tokens_saved, extracted_data, answer
"""

import sys
import json

from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: compress_image.py <image_path> [question]"}))
        sys.exit(1)

    image_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 else "Describe this image"

    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        enable_caching=True,
        log_level="ERROR",  # Silent
    )

    try:
        pipeline = LatentGatePipeline(config, preload=False)
        result = pipeline.query(image_path, question)

        output = {
            "status": "success",
            "compact_payload": result["compact_prompt"],
            "tokens_estimated": result["tokens_estimated"],
            "tokens_saved": 1200 - result["tokens_estimated"],
            "savings_percent": round((1 - result["tokens_estimated"] / 1200) * 100, 1),
            "extracted_data": result["payload"],
            "preview_answer": result["answer"][:200],
            "timing_ms": result.get("timing", {}).get("total_ms", 0),
        }

        print(json.dumps(output, indent=2))

    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "error": f"File not found: {e}"}))
        sys.exit(1)
    except ConnectionError:
        print(json.dumps({
            "status": "error",
            "error": "Ollama not running. Start with: ollama serve",
            "fallback": "Proceed with raw image analysis"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
