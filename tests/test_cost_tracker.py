"""Tests for CostTracker."""

import sqlite3
import pytest
from latent_gate.cost_tracker import CostTracker, UsageRecord

@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_usage.db"
    yield db_path

def test_cost_tracker_lifecycle(temp_db):
    """Test cost tracker context manager and basic DB initialization."""
    with CostTracker(db_path=str(temp_db)) as tracker:
        assert tracker._conn is not None
        
        # Check tables were created
        cursor = tracker._conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usage'")
        assert cursor.fetchone() is not None

    # Should be closed after context
    assert tracker._conn is None

def test_record_usage(temp_db):
    """Test recording usage accurately updates DB and cache."""
    with CostTracker(db_path=str(temp_db)) as tracker:
        tracker.record_usage(
            query_type="image_compression",
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=200,
            tokens_saved=500,
            compression_ratio=1.5,
            latency_ms=150.0,
            was_cached=False
        )
        
        stats = tracker.get_session_statistics()
        assert stats["total_queries"] == 1
        assert stats["total_tokens_saved"] == 500
        assert stats["total_cost"] > 0
        
        # Check persistence
        cursor = tracker._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage")
        assert cursor.fetchone()[0] == 1

def test_pricing_calculation(temp_db):
    """Test exact pricing calculation against DEFAULT_PRICING."""
    with CostTracker(db_path=str(temp_db)) as tracker:
        # GPT-4o input 0.0025, output 0.01 per 1k
        tracker.record_usage(
            query_type="test",
            provider="openai",
            model="gpt-4o",
            input_tokens=2000,
            output_tokens=1000,
            tokens_saved=2000,
            compression_ratio=2.0,
            latency_ms=100.0,
            was_cached=False
        )
        
        stats = tracker.get_session_statistics()
        # Cost should be: (2000 / 1000) * 0.0025 + (1000 / 1000) * 0.01 = 0.005 + 0.01 = 0.015
        assert abs(stats["total_cost"] - 0.015) < 1e-6
        
        # Savings: 2000 tokens saved
        assert stats["total_tokens_saved"] == 2000

def test_clear_history(temp_db):
    """Test clearing the database."""
    with CostTracker(db_path=str(temp_db)) as tracker:
        tracker.record_usage("test", "openai", "gpt-4o-mini", 100, 50, 200, 3.0, 100.0)
        tracker.clear_history()
        
        cursor = tracker._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage")
        assert cursor.fetchone()[0] == 0
