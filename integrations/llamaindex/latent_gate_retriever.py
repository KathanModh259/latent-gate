"""
LatentGate LlamaIndex Integration — Use LatentGate as a LlamaIndex component.

Wraps LatentGatePipeline for use in LlamaIndex queries and pipelines.

Usage:
    from integrations.llamaindex.latent_gate_retriever import LatentGateQueryEngine
    
    engine = LatentGateQueryEngine(provider="openai", model="gpt-4o-mini")
    response = engine.query("What is in this image?", image_path="photo.jpg")
"""

from typing import Any, Dict, List, Optional
from pydantic import Field

from latent_gate import LatentGatePipeline, PipelineConfig


class LatentGateQueryEngine:
    """
    LlamaIndex-compatible query engine for LatentGate.
    
    Wraps LatentGatePipeline for use in LlamaIndex pipelines.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        vision_model: str = "llava:7b",
        predictor_model: str = "llama3:8b",
        temperature: float = 0.3,
    ):
        """
        Initialize the query engine.
        
        Args:
            provider: Remote LLM provider
            model: Remote model name
            vision_model: Ollama vision model
            predictor_model: Ollama text model
            temperature: Generation temperature
        """
        config = PipelineConfig(
            vision_model=vision_model,
            predictor_model=predictor_model,
            remote_provider=provider,
            remote_model=model,
            temperature=temperature,
        )
        self._pipeline = LatentGatePipeline(config, preload=True)
    
    def query(
        self,
        query_str: str,
        image_path: Optional[str] = None,
        text: Optional[str] = None,
        **kwargs: Any,
    ) -> "LatentGateResponse":
        """
        Query with text, image, or both.
        
        Args:
            query_str: The question to ask
            image_path: Optional image path
            text: Optional text to compress
            **kwargs: Additional arguments
            
        Returns:
            LatentGateResponse object
        """
        if image_path and text:
            result = self._pipeline.query_universal(
                text=text, image=image_path, question=query_str
            )
        elif image_path:
            result = self._pipeline.query(image_path, query_str)
        elif text:
            result = self._pipeline.query_text(text, question=query_str)
        else:
            result = self._pipeline.query_text(query_str)
        
        return LatentGateResponse(
            answer=result["answer"],
            metadata={
                "tokens_estimated": result.get("tokens_estimated", 0),
                "tokens_saved": result.get("tokens_saved", 0),
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "timing": result.get("timing", {}),
                "input_type": result.get("input_type", "text"),
            },
        )
    
    def compress_documents(self, documents: List[str], question: str) -> "LatentGateResponse":
        """
        Compress RAG documents and answer a question.
        
        Args:
            documents: List of document strings
            question: Question about the documents
            
        Returns:
            LatentGateResponse object
        """
        result = self._pipeline.query_documents(documents, question)
        
        return LatentGateResponse(
            answer=result["answer"],
            metadata={
                "tokens_estimated": result.get("tokens_estimated", 0),
                "tokens_saved": result.get("tokens_saved", 0),
                "compression_ratio": result.get("compression_ratio", "1.0x"),
                "timing": result.get("timing", {}),
                "input_type": "documents",
                "document_count": len(documents),
            },
        )
    
    def get_cost_estimate(
        self,
        provider: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> Dict[str, Any]:
        """Get cost estimate for a query."""
        return self._pipeline.estimate_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class LatentGateResponse:
    """Response object for LatentGate queries."""
    
    def __init__(self, answer: str, metadata: Optional[Dict[str, Any]] = None):
        self.answer = answer
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return self.answer
    
    def __repr__(self) -> str:
        return f"LatentGateResponse(answer='{self.answer[:50]}...', metadata={self.metadata})"


class LatentGateRetriever:
    """
    LlamaIndex-compatible retriever for document compression.
    
    Use this when you have a list of documents and want to compress them
    before querying.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ):
        self.engine = LatentGateQueryEngine(provider=provider, model=model)
    
    def retrieve(self, documents: List[str], question: str) -> LatentGateResponse:
        """
        Retrieve and compress documents for a question.
        
        Args:
            documents: List of document strings
            question: Question to answer
            
        Returns:
            LatentGateResponse with compressed answer
        """
        return self.engine.compress_documents(documents, question)
