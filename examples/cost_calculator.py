"""
Token Cost Calculator
=====================
Compare costs between traditional VLM API calls and LatentGate pipeline.
Run this without any dependencies — it's pure math.
"""


def calculate():
    # Pricing per 1M tokens (as of mid-2025)
    pricing = {
        "gpt-4o":           {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "claude-sonnet-4":  {"input": 3.00,  "output": 15.00},
        "claude-haiku":     {"input": 0.25,  "output": 1.25},
        "gemini-2.0-flash": {"input": 0.10,  "output": 0.40},
    }

    traditional_tokens = 1200  # Image description + prompt
    latentgate_tokens = 200    # Compact semantic payload + prompt

    print("=" * 70)
    print("TOKEN COST COMPARISON: Traditional VLM vs LatentGate Pipeline")
    print("=" * 70)
    print(f"{'Model':<22} {'Traditional':>14} {'LatentGate':>14} {'Savings':>10}")
    print("-" * 70)

    for model, prices in pricing.items():
        trad = (traditional_tokens / 1_000_000) * prices["input"] * 1000
        lg = (latentgate_tokens / 1_000_000) * prices["input"] * 1000
        pct = (1 - lg / trad) * 100

        print(f"{model:<22} ${trad:>10.4f}/1K   ${lg:>10.4f}/1K   {pct:>6.0f}%")

    print("-" * 70)
    ratio = traditional_tokens / latentgate_tokens
    print(f"Token reduction: {traditional_tokens} → {latentgate_tokens} ({ratio:.1f}x fewer)")

    # Scale calculation
    queries = [1_000, 10_000, 100_000, 1_000_000]
    model = "gpt-4o-mini"
    price = pricing[model]["input"]

    print(f"\nAt scale with {model} (${price}/MTok input):")
    print(f"{'Queries':>12} {'Traditional':>14} {'LatentGate':>14} {'Saved':>12}")
    print("-" * 55)

    for q in queries:
        trad = (q * traditional_tokens / 1_000_000) * price
        lg = (q * latentgate_tokens / 1_000_000) * price
        saved = trad - lg
        print(f"{q:>12,} ${trad:>12.2f} ${lg:>12.2f} ${saved:>10.2f}")

    # With selective decoding
    print(f"\nWith Selective Decoding (video streams, ~2.85x reduction):")
    print(f"{'Frames':>12} {'Traditional':>14} {'LatentGate':>14} {'Saved':>12}")
    print("-" * 55)

    for q in queries:
        trad = (q * traditional_tokens / 1_000_000) * price
        effective_calls = q / 2.85
        lg = (effective_calls * latentgate_tokens / 1_000_000) * price
        saved = trad - lg
        print(f"{q:>12,} ${trad:>12.2f} ${lg:>12.2f} ${saved:>10.2f}")


if __name__ == "__main__":
    calculate()
