# How LatentGate Works — Deep Dive

## The Core Idea

Traditional VLM API calls send the **raw image** (encoded as base64 or vision tokens) 
to a cloud LLM. This means the cloud model has to do ALL the work:
- Parse the image pixels
- Identify objects, scenes, text
- Understand spatial relationships
- Then reason about your question

**LatentGate flips this**: we do steps 1-4 **locally for free** using Ollama, 
then send only a **compact structured summary** (~150-200 tokens) to the cloud LLM 
for the final reasoning step.

## Mapping to VL-JEPA Architecture

This pipeline is inspired by Meta FAIR's VL-JEPA paper (arXiv:2512.10942).

### VL-JEPA's Key Insight

> "Instead of autoregressively generating tokens as in classical VLMs, VL-JEPA 
> predicts continuous embeddings of the target texts. By learning in an abstract 
> representation space, the model focuses on task-relevant semantics while 
> abstracting away surface-level linguistic variability."

### How We Apply This

| VL-JEPA Component | Paper Implementation | LatentGate Implementation |
|---|---|---|
| **X-Encoder** | V-JEPA 2 ViT-L (304M params) | LLaVA via Ollama (local, free) |
| **Predictor** | Llama 3 Transformer layers (490M params) | Ollama text LLM structures extraction |
| **Y-Encoder** | EmbeddingGemma-300M | SemanticPayload dataclass |
| **Y-Decoder** | Lightweight text decoder | Cloud LLM API (minimal input) |

### Selective Decoding

VL-JEPA showed that for video streams, you don't need to decode every frame.
Their selective decoding at 0.35 Hz matched uniform decoding at 1 Hz — a 
**~2.85× reduction** in decoding operations.

LatentGate implements this by comparing consecutive SemanticPayloads using 
Jaccard similarity. If the scene hasn't changed enough (above the threshold), 
the previous API response is reused.

## Data Flow

```
1. IMAGE INPUT
   └─→ Read image file, encode to base64

2. X-ENCODER (Local, Free)
   └─→ Send to Ollama's LLaVA model
   └─→ Prompt asks for structured JSON extraction
   └─→ Output: raw JSON with scene_type, objects, actions, etc.

3. PREDICTOR (Local, Free)  
   └─→ Parse JSON (fast path) or restructure via text LLM (slow path)
   └─→ Output: SemanticPayload dataclass

4. SELECTIVE DECODING CHECK (if enabled)
   └─→ Compare current payload with previous
   └─→ If similar enough → reuse previous response (skip API call)
   └─→ If different → proceed to step 5

5. Y-DECODER (Remote, Paid — but minimal tokens)
   └─→ Convert SemanticPayload → compact text (~150-200 tokens)
   └─→ Send to cloud LLM with user's question
   └─→ Cloud LLM reasons about the structured data
   └─→ Output: final answer
```

## Why This Works

1. **Token Efficiency**: A structured payload like `[Scene: indoor] Objects: table, chair, lamp` 
   conveys the same information as a 500-word description but in ~30 tokens.

2. **Task Separation**: Vision understanding (what's in the image) and reasoning 
   (answering questions about it) are different tasks. Local models handle vision 
   well; cloud models excel at reasoning.

3. **Caching**: Same image = same payload. Content-hash caching means repeated 
   queries about the same image skip local processing entirely.

4. **Selective Decoding**: For video, consecutive frames often show the same scene. 
   Why pay for an API call when nothing changed?
