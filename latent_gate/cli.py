"""
Command-Line Interface for LatentGate.

Usage:
    # Image mode
    latent-gate image.jpg "What is this?" --provider ollama

    # Text mode
    latent-gate --text "Your prompt here" --provider openai

    # From file (best for long text)
    latent-gate --text-file prompt.txt --provider openai

    # Pipe from stdin (best for scripts)
    cat prompt.txt | latent-gate --text-file - --provider openai

    # Compress only (no LLM call)
    latent-gate --text-file prompt.txt --compress-only
"""

import sys
import json
import argparse

from latent_gate.config import PipelineConfig
from latent_gate.pipeline import LatentGatePipeline


def _read_stdin():
    """Read all text from stdin if data is available."""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read()


def main():
    parser = argparse.ArgumentParser(
        prog="latent-gate",
        description=(
            "LatentGate - Process Locally. Send Smart. Pay Less.\n"
            "Compress images & text locally via Ollama, send compact payloads to any LLM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  latent-gate photo.jpg "What is this?"\n'
            '  latent-gate --text "Your prompt" --provider openai\n'
            "  latent-gate --text-file prompt.txt --provider openai\n"
            "  type prompt.txt | latent-gate --text-file -\n"
            '  latent-gate photo.jpg "Analyze" --text "Extra context..."\n'
            "\n"
            "Tip: For long prompts, use --text-file instead of --text\n"
            "     to avoid shell quoting issues."
        ),
    )

    # Input sources
    parser.add_argument("image", nargs="?", default="", help="Path to image file (optional)")
    parser.add_argument("question", nargs="?", default="", help="Question (optional)")
    parser.add_argument("--text", "-t", default="", help="Text prompt (short text only)")
    parser.add_argument(
        "--text-file", "-tf", default="",
        help="Read text from file. Use '-' to read from stdin (pipe).",
    )
    parser.add_argument(
        "--compress-only",
        action="store_true",
        help="Compress prompt only (no LLM call). Shows token savings.",
    )

    # Provider settings
    parser.add_argument(
        "--provider", "-p",
        default="ollama",
        choices=[
            "openai", "anthropic", "google", "ollama",
            "groq", "deepseek", "together", "azure", "bedrock",
        ],
        help="Remote LLM provider (default: ollama)",
    )
    parser.add_argument("--vision-model", default="llava:7b", help="Ollama vision model")
    parser.add_argument("--predictor-model", default="llama3:8b", help="Ollama text model")
    parser.add_argument("--remote-model", default="", help="Remote model name")
    parser.add_argument("--api-key", default="", help="API key for cloud provider")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL")

    # Compression settings
    parser.add_argument(
        "--mode", "-m",
        default="auto",
        choices=["auto", "compress", "summarize", "condense", "code"],
        help="Text compression mode (default: auto-detect)",
    )

    # Output settings
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--json", action="store_true", help="Output full result as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # ---- Determine remote model default ----
    if not args.remote_model:
        defaults = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "google": "gemini-2.0-flash",
            "ollama": "llama3:8b",
            "groq": "llama-3.3-70b-versatile",
            "deepseek": "deepseek-chat",
            "together": "meta-llama/Llama-3-70b-chat-hf",
            "azure": "gpt-4o-mini",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        }
        args.remote_model = defaults.get(args.provider, "gpt-4o-mini")

    # ---- Read text input ----
    text_input = args.text

    if args.text_file:
        if args.text_file == "-":
            # Read from stdin (piped input)
            text_input = _read_stdin()
            if not text_input:
                print("Error: No input received from stdin. Pipe text or use --text-file <path>", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                with open(args.text_file, "r", encoding="utf-8") as f:
                    text_input = f.read()
            except FileNotFoundError:
                print(f"Error: File not found: {args.text_file}", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading file: {e}", file=sys.stderr)
                sys.exit(1)
    elif not sys.stdin.isatty():
        # Auto-detect piped input when no --text-file given
        text_input = _read_stdin()

    # ---- Validate: need at least one input ----
    if not args.image and not text_input:
        parser.error(
            "No input provided. Give me something to work with:\n"
            "  An image:       latent-gate photo.jpg \"What is this?\"\n"
            "  Short text:     latent-gate --text \"your prompt\"\n"
            "  Long text:      latent-gate --text-file prompt.txt\n"
            "  Piped text:     cat prompt.txt | latent-gate --text-file -"
        )

    # ---- Build config ----
    config = PipelineConfig(
        ollama_base_url=args.ollama_url,
        vision_model=args.vision_model,
        predictor_model=args.predictor_model,
        remote_provider=args.provider,
        remote_model=args.remote_model,
        remote_api_key=args.api_key,
        enable_caching=not args.no_cache,
        log_level="DEBUG" if args.verbose else "WARNING",
        selective_decoding=False,
    )

    # ---- Run pipeline ----
    try:
        pipeline = LatentGatePipeline(config, preload=False)

        if args.compress_only and text_input:
            result = pipeline.compress_prompt(text_input)
            result["input_type"] = "compress_only"
        elif args.image and text_input:
            result = pipeline.query_universal(
                image=args.image, text=text_input, question=args.question
            )
        elif args.image:
            result = pipeline.query(args.image, args.question or "Describe this image.")
        else:
            result = pipeline.query_text(text_input, question=args.question, mode=args.mode)

        pipeline.close()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as e:
        print(f"Connection Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Output ----
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\n{'=' * 55}")
        print(f"  Mode:            {result.get('input_type', 'unknown')}")
        if result.get("input_type") == "compress_only":
            print(f"  Original:        {result['original_tokens']} tokens")
            print(f"  Compressed:      {result['compressed_tokens']} tokens")
            print(
                f"  Saved:           ~{result['tokens_saved']} tokens ({result['compression_ratio']})"
            )
            print(f"\n  Compressed prompt:\n  {result['compressed_prompt']}")
        else:
            print(f"  Answer:          {result['answer']}")
            print(f"  Tokens sent:     ~{result['tokens_estimated']}")
            if "original_tokens" in result:
                print(f"  Original tokens: ~{result['original_tokens']}")
                print(f"  Compression:     {result['compression_ratio']}")
                print(f"  Tokens saved:    ~{result['tokens_saved']}")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
