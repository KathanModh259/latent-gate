"""
Command-Line Interface for LatentGate.

Usage:
    python -m latent_gate image.jpg "What is in this image?"
    python -m latent_gate image.jpg "Describe the scene" --provider openai --api-key sk-...
    python -m latent_gate image.jpg "Extract text" --provider ollama --remote-model llama3:8b
"""

import sys
import json
import argparse

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline


def main():
    parser = argparse.ArgumentParser(
        prog="latent-gate",
        description=(
            "LatentGate — Process Locally. Send Smart. Pay Less.\n"
            "A VL-JEPA inspired vision-language pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m latent_gate photo.jpg \"What is this?\"\n"
            "  python -m latent_gate invoice.png \"Extract total\" --provider openai\n"
            "  python -m latent_gate scene.jpg \"Describe\" --vision-model bakllava\n"
        ),
    )

    parser.add_argument("image", help="Path to image file")
    parser.add_argument("question", help="Question about the image")

    parser.add_argument(
        "--provider", default="ollama",
        choices=["openai", "anthropic", "google", "ollama"],
        help="Remote LLM provider (default: ollama)",
    )
    parser.add_argument(
        "--vision-model", default="llava:7b",
        help="Ollama vision model for X-Encoder (default: llava:7b)",
    )
    parser.add_argument(
        "--predictor-model", default="llama3:8b",
        help="Ollama text model for Predictor (default: llama3:8b)",
    )
    parser.add_argument(
        "--remote-model", default="",
        help="Remote model name (default: gpt-4o-mini for openai, llama3:8b for ollama)",
    )
    parser.add_argument(
        "--api-key", default="",
        help="API key for cloud provider (or set via env variable)",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output full result as JSON",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Determine remote model default based on provider
    if not args.remote_model:
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
            "ollama": "llama3:8b",
        }
        args.remote_model = defaults.get(args.provider, "gpt-4o-mini")

    # Build config
    config = PipelineConfig(
        ollama_base_url=args.ollama_url,
        vision_model=args.vision_model,
        predictor_model=args.predictor_model,
        remote_provider=args.provider,
        remote_model=args.remote_model,
        remote_api_key=args.api_key,
        enable_caching=not args.no_cache,
        log_level="DEBUG" if args.verbose else "WARNING",
        selective_decoding=False,  # Single image, no need
    )

    # Run pipeline
    try:
        pipeline = LatentGatePipeline(config)
        result = pipeline.query(args.image, args.question)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'=' * 50}")
        print(f"Answer: {result['answer']}")
        print(f"{'=' * 50}")
        print(f"Tokens sent to API: ~{result['tokens_estimated']}")
        print(f"Cached: {result['was_cached']}")


if __name__ == "__main__":
    main()
