# LatentGate Modelfiles

Ollama Modelfiles that force models to output structured JSON instead of conversational replies.

## Problem

Without Modelfiles, Ollama models often ignore "Return ONLY JSON" instructions and reply conversationally:

```
User: Extract image info as JSON...
Model: Sure! I'd be happy to help you analyze this image. Here's what I can see...
```

## Solution

These Modelfiles use `SYSTEM` prompts to force structured output:

```
User: Extract image info as JSON...
Model: {"scene_type":"indoor","scene_description":"kitchen","objects":["fridge","stove"]}
```

## Install

```bash
# Vision model (for image compression)
ollama create latentgate-vision -f Modelfile.vision

# Text compressor (for prompt compression)
ollama create latentgate-compressor -f Modelfile.text

# Remote decoder (for answering from compressed payloads)
ollama create latentgate-decoder -f Modelfile.decoder
```

## Use

```python
from latent_gate import LatentGatePipeline, PipelineConfig

config = PipelineConfig(
    vision_model="latentgate-vision",        # Use custom Modelfile
    predictor_model="latentgate-compressor",  # Use custom Modelfile
    remote_provider="ollama",
    remote_model="latentgate-decoder",        # Use custom Modelfile
)
```

Or in CLI:
```bash
python -m latent_gate photo.jpg "Describe" \
    --vision-model latentgate-vision \
    --predictor-model latentgate-compressor \
    --remote-model latentgate-decoder \
    --provider ollama
```

## Available Modelfiles

| Modelfile | Base Model | Purpose |
|-----------|-----------|---------|
| `Modelfile.vision` | llava:7b | Forces JSON output from image analysis |
| `Modelfile.text` | llama3:8b | Forces JSON output from text compression |
| `Modelfile.decoder` | llama3:8b | Concise answers from compressed payloads |

## Customization

Edit the `SYSTEM` prompt in any Modelfile to change the output format or behavior.
