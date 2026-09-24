"""End-to-end integration test for LatentGate pipeline.

Requires Ollama running locally. Tests are skipped if Ollama is unavailable.
"""

import pytest

LONG_PROMPT = (
    "I need you to help me build a REST API with FastAPI that handles user "
    "authentication using JWT tokens, includes rate limiting, has PostgreSQL "
    "database integration with SQLAlchemy ORM, supports file uploads to S3, "
    "and generates comprehensive OpenAPI documentation. The API should follow "
    "clean architecture patterns with repository pattern for data access. "
    "Please also include proper error handling, logging, and health check "
    "endpoints. The system should support role-based access control with "
    "admin and regular user roles. Each endpoint should have proper input "
    "validation using Pydantic models. The database should use Alembic for "
    "migrations. Include Docker support with multi-stage builds and a "
    "docker-compose file for local development. Add comprehensive unit tests "
    "with pytest and integration tests. The API should support both JSON and "
    "form-data request bodies. Include WebSocket support for real-time "
    "notifications. Add Redis caching for frequently accessed endpoints."
)


@pytest.fixture(scope="module")
def pipeline():
    """Create a LatentGate pipeline instance (module-scoped for efficiency)."""
    from latent_gate import LatentGatePipeline, PipelineConfig

    config = PipelineConfig(
        remote_provider="ollama",
        remote_model="llama3:8b",
        log_level="ERROR",
    )
    pl = LatentGatePipeline(config, preload=False)
    yield pl
    pl.close()


class TestLatentGateE2E:
    """End-to-end tests for the LatentGate pipeline."""

    def test_text_compression(self, pipeline, requires_ollama):
        """Test 1: Text compression (query_text)."""
        result = pipeline.query_text(LONG_PROMPT, mode="compress")
        assert "compact_prompt" in result
        assert result.get("tokens_estimated", 0) > 0
        assert result.get("original_tokens", 0) > 0
        assert result["input_type"] == "text"

    def test_compress_only_mode(self, pipeline, requires_ollama):
        """Test 2: Compress-only mode (no remote LLM call)."""
        result = pipeline.query_text(LONG_PROMPT, compress_only=True)
        assert "compact_prompt" in result
        assert result.get("tokens_estimated", 0) > 0
        assert not result.get("answer"), "Should be empty for compress-only"

    def test_conversation_compression(self, pipeline, requires_ollama):
        """Test 3: Conversation compression."""
        messages = [
            {"role": "user", "content": "How do I set up a Python virtual environment?"},
            {"role": "assistant", "content": "You can use python -m venv myenv to create one."},
            {"role": "user", "content": "What about installing packages?"},
            {"role": "assistant", "content": "Use pip install package_name."},
        ]
        result = pipeline.query_conversation(messages, "How do I freeze my dependencies?")
        assert "compact_prompt" in result
        assert result["input_type"] == "conversation"

    def test_document_compression(self, pipeline, requires_ollama):
        """Test 4: RAG Document compression."""
        docs = [
            "FastAPI is a modern web framework for building APIs with Python.",
            "SQLAlchemy is the Python SQL toolkit and ORM.",
            "JWT is a compact, URL-safe means of representing claims.",
        ]
        result = pipeline.query_documents(docs, "Which framework should I use for my API?")
        assert "compact_prompt" in result
        assert result["input_type"] == "documents"

    def test_prompt_compression(self, pipeline, requires_ollama):
        """Test 5: Direct prompt compression (compress_prompt)."""
        result = pipeline.compress_prompt(LONG_PROMPT)
        assert "compressed_prompt" in result
        assert result.get("original_tokens", 0) > 0
        assert result.get("compressed_tokens", 0) > 0
        assert result.get("tokens_saved", 0) >= 0
