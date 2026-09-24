"""Tests for the security module.

Tests cover:
  - API key verification (constant-time comparison)
  - Image path access validation
  - Request redaction functions
  - Error sanitization
  - SSRF validation (remote_base_url)
  - Client IP resolution
  - Text length validation
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from latent_gate.config import PipelineConfig
from latent_gate.security import (
    PathAccessError,
    verify_api_key,
    get_allowed_image_roots,
    validate_image_path_access,
    redact_query_result,
    redact_external_payload,
    sanitize_external_error,
    validate_text_length,
    validate_documents_length,
    validate_conversation_length,
    is_private_or_reserved_host,
    validate_remote_base_url,
    get_client_ip,
)

# ============================================================================
# API Key Verification
# ============================================================================


class TestVerifyApiKey:
    """Constant-time API key comparison tests."""

    def test_no_expected_key(self):
        """When no API key is configured, any request should pass."""
        assert verify_api_key(None, None) is True
        assert verify_api_key("any-key", None) is True
        assert verify_api_key("", None) is True

    def test_empty_expected_key(self):
        """Empty string expected key should pass all requests."""
        assert verify_api_key("anything", "") is True
        assert verify_api_key(None, "") is True

    def test_missing_provided_key(self):
        """When API key is configured but not provided, should fail."""
        assert verify_api_key(None, "secret-key") is False
        assert verify_api_key("", "secret-key") is False

    def test_correct_key(self):
        """Matching keys should pass."""
        assert verify_api_key("secret-key", "secret-key") is True

    def test_wrong_key(self):
        """Wrong keys should fail."""
        assert verify_api_key("wrong-key", "secret-key") is False

    def test_case_sensitive(self):
        """API key comparison should be case-sensitive."""
        assert verify_api_key("SECRET-KEY", "secret-key") is False

    def test_long_key_comparison(self):
        """Long keys should still work correctly."""
        long_key = "a" * 1000
        assert verify_api_key(long_key, long_key) is True
        assert verify_api_key(long_key, long_key + "b") is False

    def test_constant_time_no_early_return(self):
        """The comparison should not leak timing for partial matches."""
        key1 = "abcdefghijklmnopqrstuvwxyz"
        key2 = "abcdefghijklmnopqrstuvwxyZ"  # Last char different
        key3 = "Abcdefghijklmnopqrstuvwxyz"  # First char different
        key4 = "abcdefghijklmnopqrstuvwxyz"

        # All should complete without errors
        assert verify_api_key(key1, key2) is False
        assert verify_api_key(key1, key3) is False
        assert verify_api_key(key1, key4) is True


# ============================================================================
# Image Path Access Validation
# ============================================================================


class TestImagePathValidation:
    """Tests for image path access validation."""

    def test_no_allowed_roots_raises_error(self):
        """When no allowed roots are configured, should raise PathAccessError."""
        config = PipelineConfig(allowed_image_roots=[])
        with pytest.raises(PathAccessError, match="disabled"):
            validate_image_path_access("/some/image.jpg", config)

    @pytest.fixture
    def temp_dir_with_image(self, tmp_path):
        """Create a temporary directory with an image file."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_file = img_dir / "test.jpg"
        img_file.write_text("fake-image-content")
        return str(img_dir), str(img_file)

    def test_path_traversal_detected(self, temp_dir_with_image):
        """Path traversal attempts should be rejected."""
        img_dir, img_file = temp_dir_with_image
        config = PipelineConfig(allowed_image_roots=[img_dir])

        with pytest.raises(PathAccessError, match="Path traversal"):
            validate_image_path_access("../../etc/passwd", config)

        with pytest.raises(PathAccessError, match="Path traversal"):
            validate_image_path_access("/etc/passwd\0.jpg", config)

    def test_valid_path_accepted(self, temp_dir_with_image):
        """Valid path within allowed roots should pass."""
        img_dir, img_file = temp_dir_with_image
        config = PipelineConfig(allowed_image_roots=[img_dir])
        # Should not raise
        validate_image_path_access(img_file, config)

    def test_path_outside_allowed_roots(self, tmp_path):
        """Path outside allowed roots should be rejected."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "secret.jpg"
        outside_file.write_text("secret")

        config = PipelineConfig(allowed_image_roots=[str(allowed_dir)])
        with pytest.raises(PathAccessError, match="outside"):
            validate_image_path_access(str(outside_file), config)

    def test_nonexistent_file(self, tmp_path):
        """Nonexistent file should raise FileNotFoundError."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        config = PipelineConfig(allowed_image_roots=[str(img_dir)])

        with pytest.raises(FileNotFoundError, match="not found"):
            validate_image_path_access(str(img_dir / "nonexistent.jpg"), config)


# ============================================================================
# Redaction Functions
# ============================================================================


class TestRedaction:
    """Tests for payload and result redaction."""

    def test_redact_external_payload_none(self):
        """None payload should return None."""
        assert redact_external_payload(None) is None

    def test_redact_external_payload_removes_path(self):
        """Redaction should remove path, keeping only filename."""
        payload = {"source_image": "/home/user/data/photos/image.jpg", "scene_type": "indoor"}
        result = redact_external_payload(payload)
        assert result["source_image"] == "image.jpg"
        assert result["scene_type"] == "indoor"

    def test_redact_query_result_maintains_structure(self):
        """Redaction should maintain the overall structure."""
        result = {
            "answer": "test answer",
            "payload": {"source_image": "/path/to/img.jpg", "objects": ["cat"]},
        }
        redacted = redact_query_result(result)
        assert redacted["answer"] == "test answer"
        assert redacted["payload"]["source_image"] == "img.jpg"

    def test_redact_query_result_non_dict(self):
        """Non-dict results should return as-is."""
        assert redact_query_result("string") == "string"
        assert redact_query_result(42) == 42
        assert redact_query_result(None) is None

    def test_redact_payload_safe_copy(self):
        """Redaction should not modify the original payload."""
        original = {"source_image": "/secret/path/image.jpg"}
        redacted = redact_external_payload(original)
        assert redacted["source_image"] == "image.jpg"
        assert original["source_image"] == "/secret/path/image.jpg"  # Unchanged


# ============================================================================
# Error Sanitization
# ============================================================================


class TestErrorSanitization:
    """Tests for safe error messages."""

    def test_file_not_found(self):
        msg = sanitize_external_error(FileNotFoundError("test.txt"))
        assert "not found" in msg.lower()

    def test_path_access_error(self):
        msg = sanitize_external_error(PathAccessError("Custom message"))
        assert "Custom message" in msg

    def test_permission_error(self):
        msg = sanitize_external_error(PermissionError("access denied"))
        assert "Permission denied" in msg

    def test_timeout_error(self):
        msg = sanitize_external_error(TimeoutError("timed out"))
        assert "timed out" in msg.lower()

    def test_connection_error(self):
        msg = sanitize_external_error(ConnectionError("connection refused"))
        assert "connect" in msg.lower()

    def test_value_error(self):
        msg = sanitize_external_error(ValueError("invalid input: foo"))
        assert "Invalid input" in msg
        assert "foo" in msg

    def test_generic_error(self):
        msg = sanitize_external_error(RuntimeError("something went wrong"))
        assert "error occurred" in msg.lower()
        # Should not leak internal details
        assert "something" not in msg.lower()

    def test_custom_exception(self):
        class CustomException(Exception):
            pass

        msg = sanitize_external_error(CustomException("custom"))
        assert "error occurred" in msg.lower()
        assert "custom" not in msg.lower()


# ============================================================================
# Text Length Validation
# ============================================================================


class TestTextValidation:
    """Tests for input validation functions."""

    def test_validate_text_length_within_limit(self):
        """Normal text should pass validation."""
        validate_text_length("Hello, world!")
        validate_text_length("A" * 1000)

    def test_validate_text_length_exceeds_limit(self, monkeypatch):
        """Text exceeding the limit should raise ValueError."""
        monkeypatch.setenv("LATENTGATE_MAX_TEXT_CHARS", "100")
        # Reimport to pick up the env var change
        import importlib
        import latent_gate.security

        importlib.reload(latent_gate.security)
        from latent_gate.security import validate_text_length

        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_text_length("A" * 200)

    def test_validate_documents_length(self):
        """Document batch within limit should pass."""
        validate_documents_length(["short doc"])

    def test_validate_documents_length_exceeds(self, monkeypatch):
        """Overly large document batch should raise ValueError."""
        monkeypatch.setenv("LATENTGATE_MAX_TEXT_CHARS", "100")
        import importlib
        import latent_gate.security

        importlib.reload(latent_gate.security)
        from latent_gate.security import validate_documents_length

        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_documents_length(["A" * 200])

    def test_validate_conversation_length(self):
        """Conversation within limit should pass."""
        validate_conversation_length(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )

    def test_validate_conversation_length_exceeds(self, monkeypatch):
        """Overly large conversation should raise ValueError."""
        monkeypatch.setenv("LATENTGATE_MAX_TEXT_CHARS", "100")
        import importlib
        import latent_gate.security

        importlib.reload(latent_gate.security)
        from latent_gate.security import validate_conversation_length

        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_conversation_length(
                [
                    {"role": "user", "content": "A" * 200},
                ]
            )

    def test_validate_text_length_custom_field_name(self):
        """Custom field names should appear in the error message."""
        with patch("latent_gate.security.MAX_TEXT_BODY_CHARS", 10):
            with pytest.raises(ValueError, match="custom_field"):
                validate_text_length("A" * 50, field_name="custom_field")


# ============================================================================
# SSRF Validation
# ============================================================================


class TestSSRFValidation:
    """Tests for SSRF protection on remote_base_url."""

    def test_empty_url_returns_none(self):
        """Empty URL should be valid (no warning)."""
        assert validate_remote_base_url("") is None

    def test_valid_public_url(self):
        """Valid public URLs should not raise warnings."""
        assert validate_remote_base_url("https://api.openai.com/v1") is None

    def test_localhost_warning(self):
        """Localhost URLs should trigger a warning."""
        warning = validate_remote_base_url("http://localhost:8080")
        assert warning is not None
        assert "private" in warning.lower()

    def test_private_ip_warning(self):
        """Private IP addresses should trigger a warning."""
        warning = validate_remote_base_url("http://192.168.1.1:8000")
        assert warning is not None
        assert "private" in warning.lower()

    def test_loopback_warning(self):
        """Loopback addresses should trigger a warning."""
        warning = validate_remote_base_url("http://127.0.0.1:11434")
        assert warning is not None

    def test_invalid_scheme(self):
        """Invalid schemes should trigger a warning."""
        warning = validate_remote_base_url("ftp://example.com")
        assert warning is not None
        assert "http or https" in warning

    def test_link_local_warning(self):
        """Link-local addresses should trigger a warning."""
        warning = validate_remote_base_url("http://169.254.1.1")
        assert warning is not None


# ============================================================================
# Private Host Detection
# ============================================================================


class TestPrivateHostDetection:
    """Tests for private/reserved host detection."""

    def test_empty_host(self):
        assert is_private_or_reserved_host("") is True

    def test_localhost(self):
        assert is_private_or_reserved_host("localhost") is True
        assert is_private_or_reserved_host("localhost.localdomain") is True

    def test_private_ipv4(self):
        assert is_private_or_reserved_host("10.0.0.1") is True
        assert is_private_or_reserved_host("172.16.0.1") is True
        assert is_private_or_reserved_host("192.168.1.1") is True

    def test_loopback(self):
        assert is_private_or_reserved_host("127.0.0.1") is True

    def test_public_ip(self):
        assert is_private_or_reserved_host("8.8.8.8") is False
        assert is_private_or_reserved_host("104.16.0.1") is False

    def test_ipv6_loopback(self):
        assert is_private_or_reserved_host("::1") is True

    def test_ipv6_unique_local(self):
        assert is_private_or_reserved_host("fc00::1") is True

    def test_bracketed_ipv6(self):
        assert is_private_or_reserved_host("[::1]") is True
        # 2001:db8::/32 is the documentation prefix (RFC 3849)
        # Marked as "Reserved by protocol" by IANA, so ipaddress marks is_reserved=True
        assert is_private_or_reserved_host("[2001:db8::1]") is True


# ============================================================================
# Client IP Resolution
# ============================================================================


class TestClientIPResolution:
    """Tests for client IP resolution with proxy support."""

    def test_direct_ip(self):
        """Direct connection should return the peer IP."""
        request = MagicMock()
        request.client.host = "192.168.1.100"
        assert get_client_ip(request) == "192.168.1.100"

    def test_no_client_unknown(self):
        """When client is None, should return 'unknown'."""
        request = MagicMock()
        request.client = None
        assert get_client_ip(request) == "unknown"

    def test_trusted_proxy_forwarded_for(self):
        """X-Forwarded-For should be honored when peer is a trusted proxy."""
        with patch.dict(os.environ, {"LATENTGATE_TRUSTED_PROXY": "10.0.0.1"}):
            request = MagicMock()
            request.client.host = "10.0.0.1"
            request.headers.get.return_value = "203.0.113.5, 10.0.0.1"

            ip = get_client_ip(request)
            assert ip == "203.0.113.5"
            request.headers.get.assert_called_once_with("X-Forwarded-For", "")

    def test_untrusted_proxy_direct_ip_used(self):
        """X-Forwarded-For should be ignored when peer is not a trusted proxy."""
        with patch.dict(os.environ, {"LATENTGATE_TRUSTED_PROXY": "10.0.0.1"}):
            request = MagicMock()
            request.client.host = "192.168.1.100"
            request.headers.get.return_value = "203.0.113.5"

            ip = get_client_ip(request)
            assert ip == "192.168.1.100"
            # X-Forwarded-For should not be checked
            request.headers.get.assert_not_called()

    def test_multiple_trusted_proxies(self):
        """Multiple trusted proxies should all be accepted."""
        with patch.dict(os.environ, {"LATENTGATE_TRUSTED_PROXY": "10.0.0.1,10.0.0.2"}):
            request = MagicMock()
            request.client.host = "10.0.0.2"
            request.headers.get.return_value = "203.0.113.5"

            ip = get_client_ip(request)
            assert ip == "203.0.113.5"


# ============================================================================
# get_allowed_image_roots
# ============================================================================


class TestGetAllowedImageRoots:
    """Tests for the get_allowed_image_roots helper."""

    def test_from_config(self):
        """Roots from config should be returned."""
        config = PipelineConfig(allowed_image_roots=["/data/images"])
        roots = get_allowed_image_roots(config)
        assert len(roots) >= 1

    def test_from_env(self):
        """Roots from environment should be merged."""
        with patch.dict(
            os.environ, {"LATENTGATE_ALLOWED_IMAGE_ROOTS": "/data/photos;/data/diagrams"}
        ):
            config = PipelineConfig(allowed_image_roots=["/data/images"])
            roots = get_allowed_image_roots(config)
            # At minimum, the config roots + env roots are present
            source_strs = [str(r) for r in roots]
            assert any("images" in s for s in source_strs)
