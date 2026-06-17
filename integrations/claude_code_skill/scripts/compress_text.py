#!/usr/bin/env python3
"""
Compress long text/prompt to a compact semantic payload for Claude.

Usage:
    python compress_text.py <text_file_or_string> [mode]

Modes: auto, compress, summarize, condense, code
"""

import sys
import json
import os

from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: compress_text.py <text_file> [mode]"}))
        sys.exit(1)

    text_input = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "auto"

    # If first arg is a file path, read it; otherwise treat as text
    if os.path.exists(text_input):
        with open(text_input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = text_input

    config = PipelineConfig(
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        log_level="ERROR",
    )

    try:
        pipeline = LatentGatePipeline(config, preload=False)
        result = pipeline.query_text(text, mode=mode)

        output = {
            "status": "success",
            "compact_payload": result["compact_prompt"],
            "original_tokens": result.get("original_tokens", 0),
            "compressed_tokens": result["tokens_estimated"],
            "compression_ratio": result.get("compression_ratio", "1.0x"),
            "tokens_saved": result.get("tokens_saved", 0),
            "extracted_data": result["payload"],
            "preview_answer": result["answer"][:200],
        }

        print(json.dumps(output, indent=2))

    except ConnectionError:
        print(json.dumps({
            "status": "error",
            "error": "Ollama not running. Start with: ollama serve",
            "fallback": "Proceed with raw text"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
