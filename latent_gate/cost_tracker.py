"""
Cost Tracker — Persistent cost tracking with analytics.

Tracks API usage, token consumption, and costs across sessions.
Provides analytics, projections, and exportable reports.

Features:
  - SQLite-based persistent storage
  - Token usage tracking per provider/model
  - Cost calculations with configurable rates
  - Session and query-level statistics
  - Exportable reports (JSON, CSV)
  - Cost projections and alerts
"""

import json
import csv
import sqlite3
import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("latent_gate.cost")


# ============================================================================
# Token Pricing (per 1K tokens)
# ============================================================================

DEFAULT_PRICING = {
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    },
    "google": {
        "gemini-2.0-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    },
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},  # Free local inference
    },
    "groq": {
        "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
        "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    },
    "deepseek": {
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
        "deepseek-coder": {"input": 0.00014, "output": 0.00028},
    },
    "together": {
        "meta-llama/Llama-3-8b-chat-hf": {"input": 0.0002, "output": 0.0002},
        "meta-llama/Llama-3-70b-chat-hf": {"input": 0.0009, "output": 0.0009},
    },
    "azure": {
        "default": {"input": 0.005, "output": 0.015},  # Same as OpenAI
    },
    "bedrock": {
        "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
    },
}


@dataclass
class UsageRecord:
    """Single usage record for tracking."""

    timestamp: float
    session_id: str
    query_type: str  # image, text, conversation, documents, universal
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    tokens_saved: int
    compression_ratio: float
    latency_ms: float
    was_cached: bool


class CostTracker:
    """
    Track API usage and costs with persistent storage.

    Usage:
        tracker = CostTracker()
        tracker.record_usage(...)
        stats = tracker.get_statistics()
        tracker.export_report("report.json")
    """

    def __init__(
        self,
        db_path: str = ".latentgate_costs.db",
        pricing: Optional[Dict] = None,
        session_id: Optional[str] = None,
    ):
        self.db_path = db_path
        self.pricing = pricing or DEFAULT_PRICING
        self.session_id = session_id or f"session_{int(time.time())}"

        # Initialize database
        self._init_db()

        # In-memory cache for current session
        self._session_records: List[UsageRecord] = []
        self._session_start = time.time()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT NOT NULL,
                query_type TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                estimated_cost REAL NOT NULL,
                tokens_saved INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                latency_ms REAL NOT NULL,
                was_cached BOOLEAN NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session ON usage(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON usage(timestamp)
        """)

        conn.commit()
        conn.close()

    def record_usage(
        self,
        query_type: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        compression_ratio: float,
        latency_ms: float,
        was_cached: bool = False,
    ):
        """
        Record a usage event.

        Args:
            query_type: Type of query (image, text, conversation, documents, universal)
            provider: LLM provider (openai, anthropic, google, ollama)
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tokens_saved: Number of tokens saved by compression
            compression_ratio: Compression ratio achieved
            latency_ms: Request latency in milliseconds
            was_cached: Whether the result was cached
        """
        # Calculate cost
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=time.time(),
            session_id=self.session_id,
            query_type=query_type,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            latency_ms=latency_ms,
            was_cached=was_cached,
        )

        # Add to in-memory cache
        self._session_records.append(record)

        # Persist to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO usage (
                timestamp, session_id, query_type, provider, model,
                input_tokens, output_tokens, estimated_cost, tokens_saved,
                compression_ratio, latency_ms, was_cached
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record.timestamp,
                record.session_id,
                record.query_type,
                record.provider,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.estimated_cost,
                record.tokens_saved,
                record.compression_ratio,
                record.latency_ms,
                record.was_cached,
            ),
        )

        conn.commit()
        conn.close()

        logger.debug(f"Recorded usage: {query_type} via {provider}/{model} - ${cost:.6f}")

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        provider_pricing = self.pricing.get(provider, {})
        model_pricing = provider_pricing.get(
            model, provider_pricing.get("default", {"input": 0, "output": 0})
        )

        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]

        return input_cost + output_cost

    def get_statistics(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get usage statistics.

        Args:
            start_time: Start timestamp (None = all time)
            end_time: End timestamp (None = now)
            session_id: Filter by session (None = all sessions)

        Returns:
            Dictionary with statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM usage WHERE 1=1"
        params = []

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return self._empty_statistics()

        # Column indices: 0=id, 1=timestamp, 2=session_id, 3=query_type,
        # 4=provider, 5=model, 6=input_tokens, 7=output_tokens, 8=estimated_cost,
        # 9=tokens_saved, 10=compression_ratio, 11=latency_ms, 12=was_cached

        # Calculate statistics
        total_input_tokens = sum(r[6] for r in rows)
        total_output_tokens = sum(r[7] for r in rows)
        total_cost = sum(r[8] for r in rows)
        total_saved = sum(r[9] for r in rows)
        total_latency = sum(r[11] for r in rows)
        by_provider = {}
        for row in rows:
            provider = row[4]
            if provider not in by_provider:
                by_provider[provider] = {
                    "queries": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "tokens_saved": 0,
                }
            by_provider[provider]["queries"] += 1
            by_provider[provider]["input_tokens"] += row[6]
            by_provider[provider]["output_tokens"] += row[7]
            by_provider[provider]["cost"] += row[8]
            by_provider[provider]["tokens_saved"] += row[9]

        # Group by query type
        by_type = {}
        for row in rows:
            query_type = row[3]
            if query_type not in by_type:
                by_type[query_type] = {"queries": 0, "cost": 0.0, "tokens_saved": 0}
            by_type[query_type]["queries"] += 1
            by_type[query_type]["cost"] += row[8]
            by_type[query_type]["tokens_saved"] += row[9]

        return {
            "total_queries": len(rows),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 6),
            "total_tokens_saved": total_saved,
            "average_latency_ms": round(total_latency / len(rows), 2),
            "average_compression_ratio": round(sum(r[10] for r in rows) / len(rows), 2),
            "by_provider": by_provider,
            "by_type": by_type,
            "time_range": {
                "start": min(r[1] for r in rows),
                "end": max(r[1] for r in rows),
            },
        }

    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            "total_queries": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "total_tokens_saved": 0,
            "average_latency_ms": 0.0,
            "average_compression_ratio": 0.0,
            "by_provider": {},
            "by_type": {},
            "time_range": {"start": 0, "end": 0},
        }

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        if not self._session_records:
            return self._empty_statistics()

        total_input = sum(r.input_tokens for r in self._session_records)
        total_output = sum(r.output_tokens for r in self._session_records)
        total_cost = sum(r.estimated_cost for r in self._session_records)
        total_saved = sum(r.tokens_saved for r in self._session_records)
        total_latency = sum(r.latency_ms for r in self._session_records)

        return {
            "session_id": self.session_id,
            "duration_seconds": round(time.time() - self._session_start, 2),
            "total_queries": len(self._session_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": round(total_cost, 6),
            "total_tokens_saved": total_saved,
            "average_latency_ms": round(total_latency / len(self._session_records), 2),
            "tokens_saved_percentage": round(
                (total_saved / max(total_input + total_saved, 1)) * 100, 2
            ),
        }

    def get_cost_projection(
        self,
        daily_queries: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 200,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Project costs based on usage patterns.

        Args:
            daily_queries: Expected queries per day
            avg_input_tokens: Average input tokens per query
            avg_output_tokens: Average output tokens per query
            provider: LLM provider
            model: Model name

        Returns:
            Cost projection dictionary.
        """
        # Cost per query
        cost_per_query = self._calculate_cost(provider, model, avg_input_tokens, avg_output_tokens)

        # Projections
        daily_cost = cost_per_query * daily_queries
        weekly_cost = daily_cost * 7
        monthly_cost = daily_cost * 30
        yearly_cost = daily_cost * 365

        # With compression (assuming 80% reduction)
        compressed_input = avg_input_tokens * 0.2
        compressed_cost = self._calculate_cost(provider, model, compressed_input, avg_output_tokens)
        compressed_daily = compressed_cost * daily_queries

        return {
            "scenario": {
                "daily_queries": daily_queries,
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "provider": provider,
                "model": model,
            },
            "without_compression": {
                "cost_per_query": round(cost_per_query, 6),
                "daily": round(daily_cost, 4),
                "weekly": round(weekly_cost, 4),
                "monthly": round(monthly_cost, 2),
                "yearly": round(yearly_cost, 2),
            },
            "with_compression": {
                "cost_per_query": round(compressed_cost, 6),
                "daily": round(compressed_daily, 4),
                "weekly": round(compressed_daily * 7, 4),
                "monthly": round(compressed_daily * 30, 2),
                "yearly": round(compressed_daily * 365, 2),
            },
            "savings": {
                "per_query": round(cost_per_query - compressed_cost, 6),
                "daily": round(daily_cost - compressed_daily, 4),
                "weekly": round((daily_cost - compressed_daily) * 7, 4),
                "monthly": round((daily_cost - compressed_daily) * 30, 2),
                "yearly": round((daily_cost - compressed_daily) * 365, 2),
                "percentage": round((1 - compressed_cost / max(cost_per_query, 0.000001)) * 100, 2),
            },
        }

    def export_report(
        self,
        filepath: str,
        fmt: str = "json",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """
        Export usage report to file.

        Args:
            filepath: Output file path
            fmt: Export format ("json" or "csv")
            start_time: Start timestamp filter
            end_time: End timestamp filter
        """
        stats = self.get_statistics(start_time, end_time)

        if fmt == "json":
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)

        elif fmt == "csv":
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM usage WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "session_id",
                        "query_type",
                        "provider",
                        "model",
                        "input_tokens",
                        "output_tokens",
                        "estimated_cost",
                        "tokens_saved",
                        "compression_ratio",
                        "latency_ms",
                        "was_cached",
                    ]
                )
                # Skip id column (index 0) from SELECT *
                writer.writerows(row[1:] for row in rows)

        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Exported report to {filepath} ({fmt})")

    def clear_history(self, before_timestamp: Optional[float] = None):
        """
        Clear usage history.

        Args:
            before_timestamp: Only clear records before this timestamp (None = clear all)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if before_timestamp:
            cursor.execute("DELETE FROM usage WHERE timestamp < ?", (before_timestamp,))
        else:
            cursor.execute("DELETE FROM usage")

        conn.commit()
        conn.close()

        logger.info("Cleared usage history")


# ============================================================================
# Convenience Functions
# ============================================================================


def get_tracker(db_path: str = ".latentgate_costs.db") -> CostTracker:
    """Get a cost tracker instance."""
    return CostTracker(db_path)
