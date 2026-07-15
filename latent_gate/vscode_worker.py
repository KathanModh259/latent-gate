"""Persistent JSON-lines worker for the VS Code extension.

The extension keeps this process alive and sends structured requests over stdin.
That avoids rebuilding Python scripts and reinitializing the pipeline for every
command.
"""

import json
import sys
import traceback
from typing import Any, Dict, Optional

from latent_gate import LatentGatePipeline, PipelineConfig


class Worker:
    def __init__(self):
        self.pipeline: Optional[LatentGatePipeline] = None
        self.config_signature: str = ""

    def close(self) -> None:
        if self.pipeline:
            self.pipeline.close()
            self.pipeline = None

    def get_pipeline(self, raw_config: Dict[str, Any]) -> LatentGatePipeline:
        signature = json.dumps(raw_config, sort_keys=True)
        if self.pipeline and signature == self.config_signature:
            return self.pipeline

        self.close()
        config = PipelineConfig(
            ollama_base_url=raw_config.get("ollamaBaseUrl", "http://localhost:11434"),
            vision_model=raw_config.get("visionModel", "llava:7b"),
            predictor_model=raw_config.get("predictorModel", "phi3:mini"),
            remote_provider=raw_config.get("remoteProvider", "ollama"),
            remote_model=raw_config.get("remoteModel", "phi3:mini"),
            remote_api_key=raw_config.get("remoteApiKey", ""),
            selective_decoding=raw_config.get("selectiveDecoding", True),
            similarity_threshold=raw_config.get("similarityThreshold", 0.85),
            use_embeddings=raw_config.get("useEmbeddings", True),
            enable_caching=raw_config.get("enableCaching", True),
            log_level=raw_config.get("logLevel", "WARNING"),
            max_image_dimension=raw_config.get("maxImageDimension", 1280),
            max_concurrent_requests=raw_config.get("maxConcurrentRequests", 3),
        )
        self.pipeline = LatentGatePipeline(config, preload=False)
        self.config_signature = signature
        return self.pipeline

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        command = message.get("command")
        if command == "shutdown":
            self.close()
            return {"ok": True}

        pipeline = self.get_pipeline(message.get("config", {}))

        if command == "compress_image":
            result = pipeline.query(
                message["imagePath"],
                message.get("question", "Describe this image"),
            )
            return {
                "answer": result["answer"],
                "compact_prompt": result["compact_prompt"],
                "tokens_estimated": result["tokens_estimated"],
                "tokens_saved": result.get("tokens_saved", 0),
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "payload": result["payload"],
                "timing": result["timing"],
            }

        if command == "compress_text":
            payload = pipeline.text_processor.compress(
                message["text"],
                mode=message.get("mode", "auto"),
            )
            compact = payload.to_compact_prompt()
            return {
                "answer": compact,
                "compact_prompt": compact,
                "tokens_estimated": payload.compressed_token_count,
                "original_tokens": payload.original_token_count,
                "compression_ratio": (
                    f"{payload.compression_ratio:.1f}x"
                    if payload.compression_ratio > 0
                    else "1.0x"
                ),
                "tokens_saved": payload.original_token_count - payload.compressed_token_count,
                "payload": payload.to_dict(),
                "timing": {
                    "local_ms": payload.processing_time_ms,
                    "remote_ms": 0,
                    "total_ms": payload.processing_time_ms,
                },
            }

        if command == "compress_document":
            result = pipeline.query_documents(
                [message["text"]],
                message.get("question", "Summarize the key points"),
            )
            return {
                "answer": result["answer"],
                "compact_prompt": result["compact_prompt"],
                "tokens_estimated": result["tokens_estimated"],
                "original_tokens": result.get("original_tokens", 0),
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "tokens_saved": result.get("tokens_saved", 0),
                "payload": result["payload"],
                "timing": result["timing"],
            }

        raise ValueError(f"Unknown command: {command}")


def main() -> None:
    worker = Worker()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                response = {
                    "id": message.get("id"),
                    "ok": True,
                    "result": worker.handle(message),
                }
            except Exception as e:
                response = {
                    "id": message.get("id") if "message" in locals() else None,
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            print(json.dumps(response), flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
