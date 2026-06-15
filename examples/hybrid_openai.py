"""
Example: Hybrid — Local Processing + OpenAI (Cost Optimized)
=============================================================
Heavy lifting locally (FREE), smart reasoning via GPT-4o-mini (CHEAP).

Cost comparison:
    Traditional: ~1200 input tokens per image → $0.18/1K queries
    LatentGate:  ~200 input tokens per image  → $0.03/1K queries
    Savings: ~83%

Prerequisites:
    ollama pull llava:7b
    ollama pull llama3:8b
    export OPENAI_API_KEY="sk-your-key"
"""

import os
from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    config = PipelineConfig(
        # Local stages (FREE)
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        # Remote stage (PAID — but minimal tokens)
        remote_provider="openai",
        remote_model="gpt-4o-mini",
        remote_api_key=os.getenv("OPENAI_API_KEY", ""),
        # Settings
        enable_caching=True,
    )

    pipeline = LatentGatePipeline(config)

    result = pipeline.query(
        image_path="invoice.png",
        question="Extract the total amount, invoice date, and vendor name."
    )

    print(f"\nAnswer: {result['answer']}")
    print(f"Tokens sent to OpenAI: ~{result['tokens_estimated']}")
    print(f"Traditional would send: ~1200 tokens")
    savings = max(1, (1200 - result['tokens_estimated']))
    print(f"Tokens saved: ~{savings} ({savings/12:.0f}% reduction)")


if __name__ == "__main__":
    main()
