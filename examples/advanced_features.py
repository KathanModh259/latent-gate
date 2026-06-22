"""
Example: Advanced Features Showcase
====================================
Demonstrates the new features added in v0.5.0:
  - Cosine similarity for selective decoding
  - FastAPI server
  - Video processing
  - Cost tracking
  - Async support
  - Configuration persistence
  - Structured logging
  - Docker deployment
  - Plugin system
  - Multi-language support

Prerequisites:
    pip install latent-gate[all]
    ollama pull llava:7b
    ollama pull llama3:8b
"""

import asyncio
from latent_gate import (
    LatentGatePipeline,
    PipelineConfig,
    AsyncLatentGatePipeline,
    VideoProcessor,
    VideoConfig,
    CostTracker,
    MultiLanguageProcessor,
    get_plugin_manager,
    setup_logging,
    load_config,
)


def example_cosine_similarity():
    """Example: Using cosine similarity for selective decoding."""
    print("\n=== Cosine Similarity for Selective Decoding ===")
    
    config = PipelineConfig(
        use_embeddings=True,  # Enable cosine similarity
        similarity_threshold=0.85,
    )
    
    with LatentGatePipeline(config) as pipeline:
        # The selective decoder now uses cosine similarity
        # when sentence-transformers is installed
        result = pipeline.query("photo.jpg", "Describe this")
        print(f"Similarity method: cosine (if available)")
        print(f"Tokens used: {result['tokens_estimated']}")


def example_video_processing():
    """Example: Direct video file input with automatic frame extraction."""
    print("\n=== Video Processing ===")
    
    video_config = VideoConfig(
        fps=1.0,           # Extract 1 frame per second
        max_frames=50,     # Process up to 50 frames
        quality=95,        # JPEG quality
    )
    
    with VideoProcessor(config=PipelineConfig(), video_config=video_config) as processor:
        # Process a video file directly
        result = processor.process_video(
            "video.mp4",
            "Describe the action in this video"
        )
        
        print(f"Total frames: {result['total_frames']}")
        print(f"Statistics: {result['statistics']}")


def example_cost_tracking():
    """Example: Cost tracking with analytics."""
    print("\n=== Cost Tracking ===")
    
    tracker = CostTracker(db_path="example_costs.db")
    
    # Record some usage
    tracker.record_usage(
        query_type="image",
        provider="openai",
        model="gpt-4o-mini",
        input_tokens=150,
        output_tokens=200,
        tokens_saved=1000,
        compression_ratio=6.7,
        latency_ms=1500,
    )
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Total cost: ${stats['total_cost']:.6f}")
    print(f"Tokens saved: {stats['total_tokens_saved']}")
    
    # Get cost projection
    projection = tracker.get_cost_projection(
        daily_queries=1000,
        provider="openai",
        model="gpt-4o-mini"
    )
    print(f"\nCost Projection (1000 queries/day):")
    print(f"  Without compression: ${projection['without_compression']['monthly']:.2f}/month")
    print(f"  With compression: ${projection['with_compression']['monthly']:.2f}/month")
    print(f"  Savings: {projection['savings']['percentage']:.1f}%")


async def example_async_support():
    """Example: Async pipeline for non-blocking operations."""
    print("\n=== Async Support ===")
    
    async with AsyncLatentGatePipeline() as pipeline:
        # Single query
        result = await pipeline.query("photo.jpg", "What is this?")
        print(f"Async query result: {result['tokens_estimated']} tokens")
        
        # Concurrent batch processing
        results = await pipeline.query_many_images(
            ["img1.jpg", "img2.jpg", "img3.jpg"],
            "Describe each image",
            max_concurrent=3,
        )
        print(f"Processed {len(results)} images concurrently")


def example_configuration():
    """Example: Configuration persistence with YAML/TOML."""
    print("\n=== Configuration Persistence ===")
    
    # Load config from file
    try:
        config = load_config("latentgate.yaml")
        print(f"Loaded config: provider={config.remote_provider}")
    except FileNotFoundError:
        print("No config file found, using defaults")
        config = PipelineConfig()
    
    # Or use environment variables
    # LATENTGATE_REMOTE_PROVIDER=anthropic
    # LATENTGATE_REMOTE_MODEL=claude-3-5-sonnet-20241022


def example_structured_logging():
    """Example: Structured JSON logging."""
    print("\n=== Structured Logging ===")
    
    # Set up JSON logging
    setup_logging(
        level="INFO",
        log_file="latentgate.log",
        json_format=True,
        log_rotation=True,
    )
    
    # Logs will now be in JSON format with timestamps,
    # correlation IDs, and performance metrics


def example_plugin_system():
    """Example: Plugin system for custom processors."""
    print("\n=== Plugin System ===")
    
    # Get the plugin manager
    manager = get_plugin_manager()
    
    # List available plugins
    plugins = manager.list_plugins()
    print(f"Available plugins: {plugins}")
    
    # Plugins can be loaded from directories or entry points
    # manager.load_plugins_from_directory("./plugins")
    # manager.load_plugins_from_entry_points()


def example_multi_language():
    """Example: Multi-language support."""
    print("\n=== Multi-Language Support ===")
    
    # Detect language
    processor = MultiLanguageProcessor(translate_to_en=True)
    
    texts = [
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "Bonjour, comment allez-vous?",
        "こんにちは、お元気ですか？",
    ]
    
    for text in texts:
        processed, lang_info = processor.process(text)
        print(f"  {lang_info.name}: '{text[:30]}...'")
        print(f"    -> Translated: '{processed[:30]}...'")


def main():
    """Run all examples."""
    print("LatentGate v1.0.0 - Advanced Features Showcase")
    print("=" * 50)
    
    # Note: These examples require Ollama running with models
    # Some examples may fail without the proper setup
    
    try:
        example_configuration()
        example_structured_logging()
        example_cost_tracking()
        example_plugin_system()
        example_multi_language()
        
        # These require Ollama
        # example_cosine_similarity()
        # example_video_processing()
        # asyncio.run(example_async_support())
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Some examples require Ollama to be running")


if __name__ == "__main__":
    main()
