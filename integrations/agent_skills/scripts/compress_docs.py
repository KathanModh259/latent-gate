#!/usr/bin/env python3
"""
Compress multiple RAG documents into key facts for Claude.

Usage:
    python compress_docs.py "<question>" <doc1.txt> <doc2.txt> ...
"""

import sys
import json
import os

from latent_gate import LatentGatePipeline, PipelineConfig


def main():
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "Usage: compress_docs.py <question> <doc1> <doc2> ..."
        }))
        sys.exit(1)

    question = sys.argv[1]
    doc_paths = sys.argv[2:]

    # Read all documents
    documents = []
    for path in doc_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                documents.append(f.read())
        else:
            documents.append(path)  # treat as raw text

    config = PipelineConfig(
        predictor_model="llama3:8b",
        remote_provider="ollama",
        remote_model="llama3:8b",
        log_level="ERROR",
    )

    try:
        pipeline = LatentGatePipeline(config, preload=False)
        result = pipeline.query_documents(documents, question)

        output = {
            "status": "success",
            "compact_payload": result["compact_prompt"],
            "original_tokens": result.get("original_tokens", 0),
            "compressed_tokens": result["tokens_estimated"],
            "compression_ratio": result.get("compression_ratio", "1.0x"),
            "tokens_saved": result.get("tokens_saved", 0),
            "documents_processed": len(documents),
            "extracted_facts": result["payload"],
            "preview_answer": result["answer"][:300],
        }

        print(json.dumps(output, indent=2))

    except ConnectionError:
        print(json.dumps({
            "status": "error",
            "error": "Ollama not running. Start with: ollama serve",
            "fallback": "Proceed with raw documents"
        }))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
