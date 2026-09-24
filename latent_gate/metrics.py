"""
Prometheus /metrics endpoint for LatentGate.

Provides:
  - Request counters (total, by status code, by endpoint)
  - Token usage tracking (input, output, saved)
  - Request latency histogram
  - Active requests gauge
  - Pipeline state gauge

Mount this on the FastAPI app:
    from latent_gate.metrics import setup_metrics
    metrics_app = setup_metrics()
    app.mount("/metrics", metrics_app)

Requires:
    pip install prometheus-client
"""

import logging
import time

logger = logging.getLogger("latent_gate.metrics")

# Lazy import — metrics are disabled when prometheus_client is not installed
_metrics_enabled = False
_counter_requests = None
_counter_errors = None
_counter_tokens_input = None
_counter_tokens_output = None
_counter_tokens_saved = None
_histogram_latency = None
_gauge_active_requests = None
_gauge_pipeline_ready = None

try:
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

    _counter_requests = Counter(
        "latentgate_requests_total",
        "Total HTTP requests processed",
        ["endpoint", "method", "status_code"],
    )
    _counter_errors = Counter(
        "latentgate_errors_total",
        "Total errors by type",
        ["error_type"],
    )
    _counter_tokens_input = Counter(
        "latentgate_tokens_input_total",
        "Total input tokens processed",
    )
    _counter_tokens_output = Counter(
        "latentgate_tokens_output_total",
        "Total output tokens generated",
    )
    _counter_tokens_saved = Counter(
        "latentgate_tokens_saved_total",
        "Total tokens saved via compression",
    )
    _histogram_latency = Histogram(
        "latentgate_request_duration_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    )
    _gauge_active_requests = Gauge(
        "latentgate_active_requests",
        "Number of requests currently being processed",
    )
    _gauge_pipeline_ready = Gauge(
        "latentgate_pipeline_ready",
        "Whether the pipeline is initialized (1 = ready, 0 = not ready)",
    )

    _metrics_enabled = True
    logger.info("Prometheus metrics enabled")

except ImportError:
    logger.info("prometheus-client not installed — metrics disabled")


def setup_metrics():
    """Return an ASGI app that serves /metrics, or None if disabled."""
    if _metrics_enabled:
        return make_asgi_app()
    return None


def get_metrics_app():
    """Alias for setup_metrics() — returns the metrics ASGI app or None."""
    return setup_metrics()


# ============================================================================
# Middleware helpers — called from the FastAPI endpoints
# ============================================================================


def record_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_ms: float,
):
    """Record a request metric."""
    if not _metrics_enabled:
        return
    _counter_requests.labels(
        endpoint=endpoint,
        method=method,
        status_code=str(status_code),
    ).inc()
    _histogram_latency.labels(endpoint=endpoint).observe(duration_ms / 1000.0)
    _gauge_active_requests.dec()


def record_active_request_start():
    """Increment the active request gauge."""
    if _metrics_enabled:
        _gauge_active_requests.inc()


def record_error(error_type: str = "internal"):
    """Record an error."""
    if _metrics_enabled:
        _counter_errors.labels(error_type=error_type).inc()


def record_tokens(input_tokens: int = 0, output_tokens: int = 0, saved: int = 0):
    """Record token usage."""
    if not _metrics_enabled:
        return
    if input_tokens:
        _counter_tokens_input.inc(input_tokens)
    if output_tokens:
        _counter_tokens_output.inc(output_tokens)
    if saved:
        _counter_tokens_saved.inc(saved)


def set_pipeline_ready(ready: bool):
    """Set the pipeline ready gauge (1 = ready, 0 = not ready)."""
    if _metrics_enabled:
        _gauge_pipeline_ready.set(1 if ready else 0)


# ============================================================================
# FastAPI middleware
# ============================================================================


class MetricsMiddleware:
    """
    ASGI middleware to automatically record request metrics.

    Usage:
        app.add_middleware(MetricsMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/unknown")

        status_code = [200]  # mutable box for the send wrapper

        async def _send_with_metrics(message):
            if message["type"] == "http.response.start":
                status_code[0] = message["status"]
            await send(message)

        try:
            record_active_request_start()
            await self.app(scope, receive, _send_with_metrics)
        except Exception:
            record_error("unhandled")
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            record_request(
                endpoint=path,
                method=method,
                status_code=status_code[0],
                duration_ms=duration_ms,
            )
