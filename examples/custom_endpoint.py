"""
Example: Custom API Endpoint
==============================
Works with any OpenAI-compatible endpoint:
  - Azure OpenAI
  - Together AI
  - Groq
  - vLLM
  - LM Studio
  - Any OpenAI-compatible server
"""

from latent_gate import LatentGatePipeline, PipelineConfig


def azure_openai():
    """Azure OpenAI endpoint."""
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="openai",  # Azure uses OpenAI-compatible format
        remote_model="gpt-4o-mini",
        remote_api_key="your-azure-api-key",
        remote_base_url="https://your-resource.openai.azure.com/openai/deployments/your-deployment",
    )
    return LatentGatePipeline(config)


def together_ai():
    """Together AI endpoint."""
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="openai",
        remote_model="meta-llama/Llama-3-70b-chat-hf",
        remote_api_key="your-together-api-key",
        remote_base_url="https://api.together.xyz/v1",
    )
    return LatentGatePipeline(config)


def groq():
    """Groq endpoint (ultra-fast inference)."""
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="openai",
        remote_model="llama-3.1-70b-versatile",
        remote_api_key="your-groq-api-key",
        remote_base_url="https://api.groq.com/openai/v1",
    )
    return LatentGatePipeline(config)


def main():
    # Pick your provider
    pipeline = together_ai()

    result = pipeline.query("image.jpg", "Describe what you see")
    print(f"Answer: {result['answer']}")
    print(f"Tokens sent: ~{result['tokens_estimated']}")


if __name__ == "__main__":
    main()
