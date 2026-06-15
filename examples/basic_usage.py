"""
Example: Basic Usage — Fully Local (Zero Cost)
===============================================
Everything runs on Ollama. No API keys needed. No internet needed.

Prerequisites:
    ollama pull llava:7b
    ollama pull llama3:8b
"""

from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    config = PipelineConfig(
        # X-Encoder: local multimodal vision model
        vision_model="llava:7b",
        # Predictor: local text model for structuring
        predictor_model="llama3:8b",
        # Y-Decoder: also local (fully free pipeline)
        remote_provider="ollama",
        remote_model="llama3:8b",
        # Settings
        enable_caching=True,
        selective_decoding=False,
        log_level="INFO",
    )

    pipeline = LatentGatePipeline(config)

    result = pipeline.query(
        image_path="test_image.jpg",
        question="What objects are visible and what is happening?"
    )

    print(f"\nAnswer: {result['answer']}")
    print(f"Tokens sent to decoder: ~{result['tokens_estimated']}")
    print(f"Extraction time: {result['payload']['extraction_time_ms']:.0f}ms")
    print(f"\nCompact prompt sent:")
    print(f"  {result['compact_prompt']}")


if __name__ == "__main__":
    main()
