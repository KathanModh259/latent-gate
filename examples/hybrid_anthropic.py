"""
Example: Hybrid — Local Processing + Anthropic Claude
======================================================
Use Claude's reasoning capabilities with pre-processed visual input.

Prerequisites:
    ollama pull llava:7b
    ollama pull llama3:8b
    export ANTHROPIC_API_KEY="sk-ant-your-key"
"""

import os
from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="anthropic",
        remote_model="claude-sonnet-4-20250514",
        remote_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        selective_decoding=False,
    )

    pipeline = LatentGatePipeline(config)

    result = pipeline.query(
        image_path="diagram.png",
        question="Explain the architecture shown and identify potential bottlenecks."
    )

    print(f"\nAnswer: {result['answer']}")
    print(f"\nCompact prompt sent to Claude:")
    print(f"  {result['compact_prompt']}")


if __name__ == "__main__":
    main()
