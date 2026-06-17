# 🟠 LatentGate Function Schemas

Drop-in function/tool definitions for OpenAI, Anthropic Claude, and Google Gemini.

Add LatentGate compression as tools your AI agents can call automatically.

## Quick Start

### OpenAI

```python
from openai import OpenAI
from functions import LATENTGATE_FUNCTIONS, execute_function

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    tools=LATENTGATE_FUNCTIONS,
)

# Execute any tool calls
for tc in response.choices[0].message.tool_calls or []:
    result = execute_function(tc.function)
```

### Anthropic Claude

```python
from anthropic import Anthropic
from functions import ANTHROPIC_TOOLS, execute_function

client = Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=ANTHROPIC_TOOLS,
    messages=[...],
)
```

## What's Included

| Function | When Agent Uses It |
|---|---|
| `compress_image` | Before analyzing any image |
| `compress_text` | For long prompts >500 tokens |
| `compress_documents` | For RAG queries |

## Use Cases

- Building chatbots that handle images frequently
- Agents that work with long documents
- RAG pipelines burning too many tokens
- Production apps with cost concerns
