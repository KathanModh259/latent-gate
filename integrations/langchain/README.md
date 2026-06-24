# LatentGate + LangChain Integration

Use LatentGate as a LangChain component for image analysis, text compression, and more.

## Installation

```bash
pip install latent-gate langchain-core
```

## Usage

### As an LLM

```python
from integrations.langchain.latent_gate_chain import LatentGateChain

chain = LatentGateChain(provider="openai", model="gpt-4o-mini")

# Text query
result = chain.invoke("Summarize this long document...")

# Image query
result = chain.invoke("What is in this image?", image_path="photo.jpg")
```

### As a Tool

```python
from integrations.langchain.latent_gate_chain import create_latent_gate_tool, create_text_compression_tool

# Image analysis tool
image_tool = create_latent_gate_tool(provider="openai", model="gpt-4o-mini")

# Use in an agent
result = image_tool.run("photo.jpg | What objects are in this image?")

# Text compression tool
text_tool = create_text_compression_tool()

# Compress long prompts
result = text_tool.run("Your 1000-word prompt here...")
```

### In a Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from integrations.langchain.latent_gate_chain import LatentGateChain

# Create pipeline
llm = LatentGateChain(provider="openai", model="gpt-4o-mini")

# Build chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"input": "Analyze this image"})
```

## Features

- **Image Analysis**: Process images locally, send compact payloads to cloud LLMs
- **Text Compression**: Compress long prompts before sending to LLMs
- **Token Savings**: ~80% reduction in token usage
- **Selective Decoding**: Skip redundant API calls for similar content
- **Cost Tracking**: Built-in cost tracking and analytics

## Requirements

- Python 3.10+
- Ollama running locally with `llava:7b` and `llama3:8b` models
- API key for your chosen provider (OpenAI, Anthropic, Google, etc.)
