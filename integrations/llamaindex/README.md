# LatentGate + LlamaIndex Integration

Use LatentGate as a LlamaIndex query engine for image analysis, document compression, and more.

## Installation

```bash
pip install latent-gate
```

## Usage

### As a Query Engine

```python
from integrations.llamaindex.latent_gate_retriever import LatentGateQueryEngine

engine = LatentGateQueryEngine(provider="openai", model="gpt-4o-mini")

# Image query
response = engine.query("What is in this image?", image_path="photo.jpg")
print(response.answer)

# Text compression
response = engine.query("Summarize this", text="Your long text here...")
print(response.answer)

# Document compression
response = engine.compress_documents(
    documents=["doc1 text...", "doc2 text..."],
    question="How do I implement JWT?"
)
print(response.answer)
```

### As a Retriever

```python
from integrations.llamaindex.latent_gate_retriever import LatentGateRetriever

retriever = LatentGateRetriever(provider="openai", model="gpt-4o-mini")

# Compress and query documents
response = retriever.retrieve(
    documents=["RAG doc 1...", "RAG doc 2..."],
    question="What are the best practices?"
)

print(response.answer)
print(f"Tokens saved: {response.metadata['tokens_saved']}")
```

### Cost Estimation

```python
engine = LatentGateQueryEngine()

cost_estimate = engine.get_cost_estimate(
    provider="openai",
    model="gpt-4o-mini",
    input_tokens=1000,
    output_tokens=200,
)

print(f"Estimated cost: ${cost_estimate['estimated_cost']:.4f}")
print(f"With compression: ${cost_estimate['compressed_cost']:.4f}")
print(f"Savings: {cost_estimate['savings_percentage']:.1f}%")
```

## Features

- **Query Engine**: Process images and text through LatentGate
- **Document Retriever**: Compress RAG documents before querying
- **Cost Tracking**: Built-in cost estimation and savings tracking
- **Token Savings**: ~80% reduction in token usage
- **Selective Decoding**: Skip redundant API calls

## Requirements

- Python 3.10+
- Ollama running locally with `llava:7b` and `llama3:8b` models
- API key for your chosen provider
