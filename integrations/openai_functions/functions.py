"""
LatentGate Function Schemas for OpenAI / Anthropic / Google
============================================================
Drop-in function definitions to add LatentGate compression to your
OpenAI/Claude/Gemini agent.

Usage with OpenAI:
    from openai import OpenAI
    from latentgate_functions import LATENTGATE_FUNCTIONS, execute_function
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[...],
        tools=LATENTGATE_FUNCTIONS,
    )
    
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            result = execute_function(tool_call.function)
"""

import json

# ============================================================================
# OpenAI Function Definitions (works with tool_choice)
# ============================================================================

LATENTGATE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "compress_image",
            "description": (
                "Compress an image to a compact ~150 token semantic payload "
                "instead of ~1200 tokens. Use BEFORE analyzing any image to "
                "save 80%+ on token costs. Returns structured scene data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional focus question",
                    },
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_text",
            "description": (
                "Compress long text/prompt locally. Use when text > 500 tokens. "
                "Returns intent, entities, constraints, and key data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "compress", "summarize", "condense", "code"],
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compress_documents",
            "description": (
                "Compress multiple RAG documents into key facts. Use when "
                "answering questions over multiple retrieved documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "question": {"type": "string"},
                },
                "required": ["documents", "question"],
            },
        },
    },
]


# ============================================================================
# Anthropic Tools Format
# ============================================================================

ANTHROPIC_TOOLS = [
    {
        "name": f["function"]["name"],
        "description": f["function"]["description"],
        "input_schema": f["function"]["parameters"],
    }
    for f in LATENTGATE_FUNCTIONS
]


# ============================================================================
# Function Executor
# ============================================================================

def execute_function(tool_call) -> str:
    """
    Execute a LatentGate function call and return JSON string result.
    
    Works with both OpenAI tool_call objects and dict-style calls.
    """
    from latent_gate import LatentGatePipeline, PipelineConfig
    
    # Handle both OpenAI object style and dict style
    if hasattr(tool_call, 'name'):
        func_name = tool_call.name
        args = json.loads(tool_call.arguments)
    else:
        func_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
    
    config = PipelineConfig(
        vision_model="llava:7b",
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        log_level="ERROR",
    )
    pipeline = LatentGatePipeline(config, preload=False)
    
    try:
        if func_name == "compress_image":
            result = pipeline.query(
                args["image_path"],
                args.get("question", "Describe this image"),
            )
            return json.dumps({
                "compact_payload": result["compact_prompt"],
                "tokens_saved": 1200 - result["tokens_estimated"],
                "extracted_data": result["payload"],
            })
        
        elif func_name == "compress_text":
            result = pipeline.query_text(
                args["text"],
                mode=args.get("mode", "auto"),
            )
            return json.dumps({
                "compact_payload": result["compact_prompt"],
                "original_tokens": result.get("original_tokens", 0),
                "compressed_tokens": result["tokens_estimated"],
                "compression_ratio": result.get("compression_ratio", "1.0x"),
            })
        
        elif func_name == "compress_documents":
            result = pipeline.query_documents(
                args["documents"],
                args["question"],
            )
            return json.dumps({
                "compact_payload": result["compact_prompt"],
                "compressed_tokens": result["tokens_estimated"],
                "compression_ratio": result.get("compression_ratio", "1.0x"),
            })
        
        else:
            return json.dumps({"error": f"Unknown function: {func_name}"})
    
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================================
# Example: OpenAI Agent with LatentGate
# ============================================================================

EXAMPLE_OPENAI_USAGE = """
from openai import OpenAI
from latentgate_functions import LATENTGATE_FUNCTIONS, execute_function

client = OpenAI()

messages = [
    {"role": "system", "content": (
        "You have access to LatentGate compression tools. ALWAYS use "
        "compress_image before analyzing images, and compress_text for "
        "any input >500 tokens. This saves the user money."
    )},
    {"role": "user", "content": "Analyze this screenshot: /path/to/img.png"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=LATENTGATE_FUNCTIONS,
)

# Handle tool calls
msg = response.choices[0].message
if msg.tool_calls:
    messages.append(msg)
    for tc in msg.tool_calls:
        result = execute_function(tc.function)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    
    # Get final answer with compressed input
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    print(final.choices[0].message.content)
"""


EXAMPLE_ANTHROPIC_USAGE = """
from anthropic import Anthropic
from latentgate_functions import ANTHROPIC_TOOLS, execute_function

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=ANTHROPIC_TOOLS,
    messages=[
        {"role": "user", "content": "Analyze /path/to/screenshot.png"}
    ],
)

# Handle tool use blocks
for block in response.content:
    if block.type == "tool_use":
        result = execute_function({
            "name": block.name,
            "arguments": block.input,
        })
        print(result)
"""

if __name__ == "__main__":
    print("LatentGate Function Schemas")
    print("=" * 50)
    print(f"OpenAI functions: {len(LATENTGATE_FUNCTIONS)}")
    print(f"Anthropic tools:  {len(ANTHROPIC_TOOLS)}")
    print()
    print("Example OpenAI usage:")
    print(EXAMPLE_OPENAI_USAGE)
