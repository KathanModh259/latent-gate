"""
Example: Provider Configurations
================================
All supported providers in one file. Pick the one that fits your use case.

Prerequisites:
    ollama pull llava:7b
    ollama pull llama3:8b
"""

from latent_gate import LatentGatePipeline, PipelineConfig


def fully_local():
    """Fully local — zero cost, no API keys needed."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
    )


def openai_hybrid():
    """Local processing + GPT-4o-mini (~83% token savings)."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="openai",
        remote_model="gpt-4o-mini",
    )


def anthropic_hybrid():
    """Local processing + Claude Sonnet."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="anthropic",
        remote_model="claude-sonnet-4-20250514",
    )


def google_hybrid():
    """Local processing + Gemini Flash."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="google",
        remote_model="gemini-2.0-flash",
    )


def groq_fast():
    """Local processing + Groq (ultra-fast inference)."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="groq",
        remote_model="llama-3.1-8b-instant",
    )


def deepseek_hybrid():
    """Local processing + DeepSeek."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="deepseek",
        remote_model="deepseek-chat",
    )


def custom_endpoint():
    """Any OpenAI-compatible endpoint (Azure, vLLM, LM Studio, etc.)."""
    return PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="openai",
        remote_model="gpt-4o-mini",
        remote_api_key="your-api-key",
        remote_base_url="https://your-endpoint.com/v1",
    )


def main():
    configs = {
        "fully_local": fully_local,
        "openai_hybrid": openai_hybrid,
        "anthropic_hybrid": anthropic_hybrid,
        "google_hybrid": google_hybrid,
        "groq_fast": groq_fast,
        "deepseek_hybrid": deepseek_hybrid,
        "custom_endpoint": custom_endpoint,
    }

    for name, fn in configs.items():
        config = fn()
        print(f"{name:25s} -> {config.remote_provider}/{config.remote_model}")

    # Use any config with the pipeline
    config = openai_hybrid()
    pipeline = LatentGatePipeline(config, preload=False)

    result = pipeline.query("test_image.jpg", "What do you see?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Tokens sent: ~{result['tokens_estimated']}")
    pipeline.close()


if __name__ == "__main__":
    main()
