"""Shared test configuration and fixtures."""

import functools
import pytest


@functools.lru_cache(maxsize=1)
def ollama_available() -> bool:
    """Check if Ollama is running and reachable (cached after first call)."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except (ImportError, ConnectionError, OSError):
        return False


def pytest_addoption(parser):
    """Add custom CLI options."""
    parser.addoption(
        "--skip-ollama-check",
        action="store_true",
        default=False,
        help="Skip the Ollama connectivity check (no network call to Ollama)",
    )


@pytest.fixture(scope="session")
def requires_ollama(request):
    """
    Session-scoped fixture: skip test if Ollama is not available.

    The check is deferred until a test actually requests this fixture,
    so test collection is fast even when Ollama is down.
    """
    if request.config.getoption("--skip-ollama-check"):
        pytest.skip("Ollama check skipped via --skip-ollama-check")
    if not ollama_available():
        pytest.skip("Ollama is not running. Start with: ollama serve")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_ollama: mark test as requiring a running Ollama instance",
    )
