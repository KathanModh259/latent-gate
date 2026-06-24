"""
LatentGate LangChain Integration — Use LatentGate as a LangChain component.

Wraps LatentGatePipeline for use in LangChain chains, agents, and tools.

Usage:
    from integrations.langchain.latent_gate_chain import LatentGateChain
    
    chain = LatentGateChain(provider="openai", model="gpt-4o-mini")
    result = chain.invoke({"image": "photo.jpg", "question": "What is this?"})
"""

from typing import Any, Dict, List, Optional

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.tools import Tool
    from pydantic import Field
except ImportError as e:
    raise ImportError(
        "LangChain dependencies not installed.\n"
        "Install with: pip install latent-gate[langchain]\n"
        "Or directly:  pip install langchain-core"
    ) from e

from latent_gate import LatentGatePipeline, PipelineConfig


class LatentGateChain(LLM):
    """
    LangChain LLM wrapper for LatentGate.
    
    Wraps the LatentGate pipeline as a LangChain-compatible LLM.
    Supports image queries, text compression, and universal mode.
    """
    
    provider: str = Field(default="openai", description="Remote LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Remote model name")
    vision_model: str = Field(default="llava:7b", description="Ollama vision model")
    predictor_model: str = Field(default="llama3:8b", description="Ollama text model")
    temperature: float = Field(default=0.3, description="Generation temperature")
    
    _pipeline: Optional[LatentGatePipeline] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def _init_pipeline(self):
        """Initialize the LatentGate pipeline."""
        if self._pipeline is None:
            config = PipelineConfig(
                vision_model=self.vision_model,
                predictor_model=self.predictor_model,
                remote_provider=self.provider,
                remote_model=self.model,
                temperature=self.temperature,
            )
            self._pipeline = LatentGatePipeline(config, preload=True)
    
    @property
    def _llm_type(self) -> str:
        return "latent-gate"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Process a prompt through LatentGate."""
        self._init_pipeline()
        
        image_path = kwargs.get("image_path")
        
        if image_path:
            result = self._pipeline.query(image_path, prompt)
        else:
            result = self._pipeline.query_text(prompt)
        
        return result["answer"]
    
    def query_image(self, image_path: str, question: str) -> Dict[str, Any]:
        """Query an image directly."""
        self._init_pipeline()
        return self._pipeline.query(image_path, question)
    
    def query_text(self, text: str, question: str = "") -> Dict[str, Any]:
        """Query text directly."""
        self._init_pipeline()
        return self._pipeline.query_text(text, question=question)


def create_latent_gate_tool(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    name: str = "latent_gate",
) -> Tool:
    """
    Create a LangChain Tool for LatentGate image analysis.
    
    Args:
        provider: Remote LLM provider
        model: Remote model name
        name: Tool name
        
    Returns:
        LangChain Tool instance
    """
    chain = LatentGateChain(provider=provider, model=model)
    
    def analyze_image(input_text: str) -> str:
        """Analyze an image using LatentGate. Input format: 'image_path | question'"""
        parts = input_text.split("|", 1)
        if len(parts) != 2:
            return "Error: Input must be 'image_path | question'"
        
        image_path = parts[0].strip()
        question = parts[1].strip()
        
        result = chain.query_image(image_path, question)
        return result["answer"]
    
    return Tool(
        name=name,
        description="Analyze images using local vision processing. Input: 'image_path | question'",
        func=analyze_image,
    )


def create_text_compression_tool(
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    name: str = "compress_text",
) -> Tool:
    """
    Create a LangChain Tool for text compression.
    
    Args:
        provider: Remote LLM provider
        model: Remote model name
        name: Tool name
        
    Returns:
        LangChain Tool instance
    """
    chain = LatentGateChain(provider=provider, model=model)
    
    def compress_text(input_text: str) -> str:
        """Compress and process long text using LatentGate."""
        result = chain.query_text(input_text)
        return result["answer"]
    
    return Tool(
        name=name,
        description="Compress long text and get a concise response. Useful for prompts over 500 tokens.",
        func=compress_text,
    )
