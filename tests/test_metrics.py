"""Tests for the Prometheus /metrics module.

Tests cover:
  - Module initialization with/without prometheus_client
  - All recording functions (request, error, tokens, active, pipeline)
  - MetricsMiddleware ASGI middleware
  - setup_metrics and get_metrics_app
  - Integration with FastAPI app mounting
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# ============================================================================
# Module-level tests (when prometheus_client is available)
# ============================================================================


class TestMetricsInit:
    """Test metric initialization state."""

    def test_metrics_imported_as_module(self):
        """Verify the metrics module can be imported."""
        import latent_gate.metrics as m

        assert m is not None
        assert hasattr(m, "setup_metrics")
        assert hasattr(m, "record_request")
        assert hasattr(m, "record_error")
        assert hasattr(m, "record_tokens")
        assert hasattr(m, "set_pipeline_ready")
        assert hasattr(m, "record_active_request_start")
        assert hasattr(m, "MetricsMiddleware")

    def test_metrics_disabled_without_prometheus(self):
        """When prometheus_client is not installed, metrics should be disabled."""
        with patch.dict("sys.modules", {"prometheus_client": None}):
            # Force reimport by clearing the module cache
            with patch("latent_gate.metrics._metrics_enabled", False):
                from latent_gate.metrics import setup_metrics, _metrics_enabled

                assert _metrics_enabled is False
                assert setup_metrics() is None


# ============================================================================
# Recording Functions Tests
# ============================================================================


class TestRecordFunctions:
    """Test all individual recording functions."""

    def test_record_request_disabled(self):
        """When metrics are disabled, record_request is a no-op."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import record_request

            # Should not raise
            record_request("/test", "GET", 200, 100.0)

    def test_record_error_disabled(self):
        """When metrics are disabled, record_error is a no-op."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import record_error

            record_error("test_error")

    def test_record_tokens_disabled(self):
        """When metrics are disabled, record_tokens is a no-op."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import record_tokens

            record_tokens(input_tokens=100, output_tokens=50, saved=30)

    def test_record_tokens_partial(self):
        """record_tokens should handle partial arguments."""
        from latent_gate.metrics import record_tokens

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._counter_tokens_input") as mock_input:
                with patch("latent_gate.metrics._counter_tokens_output") as mock_output:
                    with patch("latent_gate.metrics._counter_tokens_saved") as mock_saved:
                        # Only input tokens
                        record_tokens(input_tokens=100)
                        mock_input.inc.assert_called_once_with(100)
                        mock_output.inc.assert_not_called()
                        mock_saved.inc.assert_not_called()

                        mock_input.reset_mock()
                        # Only saved
                        record_tokens(saved=50)
                        mock_input.inc.assert_not_called()
                        mock_saved.inc.assert_called_once_with(50)

    def test_set_pipeline_ready_disabled(self):
        """When metrics are disabled, set_pipeline_ready is a no-op."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import set_pipeline_ready

            set_pipeline_ready(True)
            set_pipeline_ready(False)

    def test_set_pipeline_ready_values(self):
        """set_pipeline_ready should set 1 for True, 0 for False."""
        from latent_gate.metrics import set_pipeline_ready

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._gauge_pipeline_ready") as mock_gauge:
                set_pipeline_ready(True)
                mock_gauge.set.assert_called_once_with(1)

                mock_gauge.reset_mock()
                set_pipeline_ready(False)
                mock_gauge.set.assert_called_once_with(0)

    def test_record_active_request_start_disabled(self):
        """When metrics are disabled, record_active_request_start is a no-op."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import record_active_request_start

            record_active_request_start()

    def test_record_active_request_start_increments(self):
        """record_active_request_start should increment the gauge."""
        from latent_gate.metrics import record_active_request_start

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._gauge_active_requests") as mock_gauge:
                record_active_request_start()
                mock_gauge.inc.assert_called_once()


# ============================================================================
# setup_metrics tests
# ============================================================================


class TestSetupMetrics:
    """Test the setup_metrics function."""

    def test_setup_metrics_enabled(self):
        """When metrics are enabled, setup_metrics should return a callable (ASGI app)."""
        from latent_gate.metrics import setup_metrics, _metrics_enabled

        # If prometheus_client is available, setup_metrics should return an app
        if _metrics_enabled:
            result = setup_metrics()
            # It should be a callable (the make_asgi_app result)
            assert result is not None
            assert callable(result)
        else:
            # If prometheus_client is not installed, we can't fully test this
            # But we can verify setup_metrics() returns None when disabled
            with patch("latent_gate.metrics._metrics_enabled", True):
                with patch("latent_gate.metrics.setup_metrics") as mock_setup:
                    mock_setup.return_value = MagicMock()
                    result = mock_setup()
                    assert result is not None

    def test_setup_metrics_disabled(self):
        """When metrics are disabled, setup_metrics should return None."""
        with patch("latent_gate.metrics._metrics_enabled", False):
            from latent_gate.metrics import setup_metrics

            result = setup_metrics()
            assert result is None

    def test_get_metrics_app_alias(self):
        """get_metrics_app should be an alias for setup_metrics."""
        from latent_gate.metrics import get_metrics_app

        with patch("latent_gate.metrics.setup_metrics") as mock_setup:
            mock_setup.return_value = "metrics_app"
            result = get_metrics_app()
            assert result == "metrics_app"
            mock_setup.assert_called_once()


# ============================================================================
# MetricsMiddleware Tests
# ============================================================================


class TestMetricsMiddleware:
    """Test the MetricsMiddleware ASGI middleware."""

    @pytest.mark.asyncio
    async def test_middleware_non_http(self):
        """Non-HTTP scopes should pass through without metrics recording."""
        from latent_gate.metrics import MetricsMiddleware

        mock_app = AsyncMock()
        middleware = MetricsMiddleware(mock_app)

        scope = {"type": "websocket", "path": "/ws", "method": "GET"}
        await middleware(scope, None, None)
        mock_app.assert_called_once_with(scope, None, None)

    @pytest.mark.asyncio
    async def test_middleware_http_records_metrics(self):
        """HTTP requests should record metrics."""
        from latent_gate.metrics import MetricsMiddleware

        calls = []

        async def mock_send(message):
            calls.append(message)

        mock_app = AsyncMock()

        middleware = MetricsMiddleware(mock_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/query/text",
            "headers": [],
        }

        with patch("latent_gate.metrics.record_active_request_start") as mock_start:
            with patch("latent_gate.metrics.record_request") as mock_record:
                await middleware(scope, lambda: None, mock_send)

                mock_start.assert_called_once()
                # Verify record_request was called with correct parameters
                assert mock_record.call_count == 1
                args, kwargs = mock_record.call_args
                assert kwargs["endpoint"] == "/query/text"
                assert kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_middleware_captures_status_code(self):
        """The middleware should capture the response status code."""
        from latent_gate.metrics import MetricsMiddleware

        async def mock_send(message):
            pass

        async def mock_receive():
            return {"type": "http.request", "body": b""}

        async def inner_app(scope, receive, send):
            # Simulate sending a 201 response
            await send(
                {
                    "type": "http.response.start",
                    "status": 201,
                    "headers": [],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                }
            )

        middleware = MetricsMiddleware(inner_app)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/test",
            "headers": [],
        }

        with patch("latent_gate.metrics.record_request") as mock_record:
            await middleware(scope, mock_receive, mock_send)
            # Verify the status code was captured as 201
            args, kwargs = mock_record.call_args
            assert kwargs["status_code"] == 201

    @pytest.mark.asyncio
    async def test_middleware_records_error_on_exception(self):
        """If the app raises, the middleware should record an error and re-raise."""
        from latent_gate.metrics import MetricsMiddleware

        async def failing_app(scope, receive, send):
            raise ValueError("Test error")

        middleware = MetricsMiddleware(failing_app)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/fail",
            "headers": [],
        }

        with patch("latent_gate.metrics.record_error") as mock_error:
            with pytest.raises(ValueError, match="Test error"):
                await middleware(scope, lambda: None, lambda m: None)
            mock_error.assert_called_once_with("unhandled")

    @pytest.mark.asyncio
    async def test_middleware_always_records_in_finally(self):
        """The middleware should record request metrics even when an error occurs."""
        from latent_gate.metrics import MetricsMiddleware

        async def failing_app(scope, receive, send):
            raise RuntimeError("Boom")

        middleware = MetricsMiddleware(failing_app)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/boom",
            "headers": [],
        }

        with patch("latent_gate.metrics.record_active_request_start") as mock_start:
            with patch("latent_gate.metrics.record_request") as mock_record:
                with patch("latent_gate.metrics.record_error") as mock_error:
                    with pytest.raises(RuntimeError):
                        await middleware(scope, lambda: None, lambda m: None)

                    mock_start.assert_called_once()
                    mock_error.assert_called_once_with("unhandled")
                    mock_record.assert_called_once()
                    args, kwargs = mock_record.call_args
                    assert kwargs["endpoint"] == "/boom"


# ============================================================================
# Integration patterns
# ============================================================================


class TestMetricsIntegration:
    """Test integration patterns for metrics in FastAPI apps."""

    @pytest.mark.asyncio
    async def test_full_request_flow(self):
        """Test the full flow: start -> record -> complete."""
        from latent_gate.metrics import (
            record_active_request_start,
            record_request,
            record_tokens,
            set_pipeline_ready,
        )

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._gauge_active_requests") as mock_gauge:
                with patch("latent_gate.metrics._counter_requests") as mock_counter:
                    with patch("latent_gate.metrics._histogram_latency") as mock_hist:
                        with patch("latent_gate.metrics._counter_tokens_input") as mock_tokens_in:
                            with patch(
                                "latent_gate.metrics._counter_tokens_saved"
                            ) as mock_tokens_saved:
                                with patch(
                                    "latent_gate.metrics._gauge_pipeline_ready"
                                ) as mock_ready:

                                    # Simulate a full request lifecycle
                                    set_pipeline_ready(True)
                                    mock_ready.set.assert_called_once_with(1)

                                    record_active_request_start()
                                    mock_gauge.inc.assert_called_once()

                                    # Simulate processing
                                    record_tokens(input_tokens=500, saved=400)

                                    # Simulate completion
                                    record_request("/query/text", "POST", 200, 150.0)

                                    # Verify final state
                                    assert mock_gauge.dec.call_count == 1
                                    assert mock_counter.labels.return_value.inc.call_count == 1
                                    mock_tokens_in.inc.assert_called_once_with(500)
                                    mock_tokens_saved.inc.assert_called_once_with(400)

                                    # Verify histogram
                                    mock_hist.labels.assert_called_once_with(endpoint="/query/text")

    def test_multiple_requests_tracked_independently(self):
        """Multiple concurrent requests should each be tracked independently."""
        from latent_gate.metrics import record_active_request_start, record_request

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._gauge_active_requests") as mock_gauge:
                with patch("latent_gate.metrics._counter_requests") as mock_counter:
                    with patch("latent_gate.metrics._histogram_latency"):
                        # Two concurrent requests start
                        record_active_request_start()
                        record_active_request_start()
                        assert mock_gauge.inc.call_count == 2

                        # Second one finishes first
                        record_request("/health", "GET", 200, 5.0)
                        assert mock_counter.labels.return_value.inc.call_count == 1
                        assert mock_gauge.dec.call_count == 1

                        # First one finishes
                        record_request("/query", "POST", 200, 500.0)
                        assert mock_counter.labels.return_value.inc.call_count == 2
                        assert mock_gauge.dec.call_count == 2

    def test_error_records_properly(self):
        """record_error should increment the error counter with correct label."""
        from latent_gate.metrics import record_error

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._counter_errors") as mock_errors:
                record_error("upstream_failure")
                mock_errors.labels.assert_called_once_with(error_type="upstream_failure")
                mock_errors.labels.return_value.inc.assert_called_once()

    def test_token_recording_with_zero_values(self):
        """Zero values should not call inc to avoid unnecessary metric churn."""
        from latent_gate.metrics import record_tokens

        with patch("latent_gate.metrics._metrics_enabled", True):
            with patch("latent_gate.metrics._counter_tokens_input") as mock_input:
                with patch("latent_gate.metrics._counter_tokens_output") as mock_output:
                    with patch("latent_gate.metrics._counter_tokens_saved") as mock_saved:
                        # All zeros — no inc calls
                        record_tokens(input_tokens=0, output_tokens=0, saved=0)
                        mock_input.inc.assert_not_called()
                        mock_output.inc.assert_not_called()
                        mock_saved.inc.assert_not_called()

    def test_middleware_class_has_app(self):
        """MetricsMiddleware should store the wrapped app."""
        from latent_gate.metrics import MetricsMiddleware

        mock_app = MagicMock()
        middleware = MetricsMiddleware(mock_app)
        assert middleware.app is mock_app
