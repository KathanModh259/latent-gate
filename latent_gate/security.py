"""
Shared security utilities for API, MCP, and CLI entry points.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from latent_gate.config import PipelineConfig

# ~125k tokens at ~4 chars/token
MAX_TEXT_BODY_CHARS = int(os.getenv("LATENTGATE_MAX_TEXT_CHARS", "500000"))


class PathAccessError(ValueError):
    """Raised when an image path is outside allowed directories."""


def verify_api_key(provided: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time API key comparison."""
    if not expected:
        return True
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def get_allowed_image_roots(config: PipelineConfig) -> list[Path]:
    """Collect resolved allowed image roots from config and environment."""
    roots = list(config.allowed_image_roots or [])
    env_roots = os.getenv("LATENTGATE_ALLOWED_IMAGE_ROOTS", "")
    if env_roots:
        roots.extend(r.strip() for r in env_roots.split(os.pathsep) if r.strip())
    return [Path(root).expanduser().resolve() for root in roots]


def validate_image_path_access(image_path: str, config: PipelineConfig) -> None:
    """
    Restrict image-path access to configured allowed roots.

    Raises PathAccessError when access is denied.
    """
    allowed_roots = get_allowed_image_roots(config)
    if not allowed_roots:
        raise PathAccessError(
            "Image path access disabled. Set LATENTGATE_ALLOWED_IMAGE_ROOTS "
            "to a colon/semicolon-separated list of allowed directories."
        )

    # Sanitize and resolve using os.path to satisfy static analysis
    image_path_str = str(image_path)
    if "\x00" in image_path_str or ".." in image_path_str:
        raise PathAccessError("Path traversal or invalid characters detected in image path.")
        
    expanded = __import__("os").path.expanduser(image_path_str)
    absolute = __import__("os").path.abspath(expanded)
    candidate = Path(absolute).resolve()

    if not candidate.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    for root in allowed_roots:
        try:
            candidate.relative_to(root)
            return
        except ValueError:
            continue

    raise PathAccessError("Image path is outside the configured allowed directories.")


def redact_external_payload(payload: Optional[dict]) -> Optional[dict]:
    """Remove sensitive path details from payload dicts returned to clients."""
    if not payload:
        return payload
    result = dict(payload)
    source = result.get("source_image")
    if source:
        result["source_image"] = Path(str(source)).name
    return result


def redact_query_result(result: dict) -> dict:
    """Redact sensitive fields from pipeline query results."""
    if not isinstance(result, dict):
        return result
    redacted = dict(result)
    payload = redacted.get("payload")
    if isinstance(payload, dict):
        redacted["payload"] = redact_external_payload(payload)
    return redacted


def sanitize_external_error(exc: Exception) -> str:
    """Return a safe error message for external consumers."""
    if isinstance(exc, FileNotFoundError):
        return "The requested file was not found."
    if isinstance(exc, PathAccessError):
        return str(exc)
    if isinstance(exc, PermissionError):
        return "Permission denied."
    if isinstance(exc, TimeoutError):
        return "Request timed out."
    if isinstance(exc, ConnectionError):
        return "Unable to connect to the backend service."
    if isinstance(exc, ValueError):
        return f"Invalid input: {exc}"
    return "An error occurred while processing the request."


def validate_text_length(text: str, field_name: str = "text") -> None:
    """Reject oversized text payloads."""
    if len(text) > MAX_TEXT_BODY_CHARS:
        raise ValueError(
            f"{field_name} exceeds maximum length of {MAX_TEXT_BODY_CHARS:,} characters"
        )


def validate_documents_length(documents: list[str]) -> None:
    """Reject oversized document batches."""
    total = sum(len(doc) for doc in documents)
    if total > MAX_TEXT_BODY_CHARS:
        raise ValueError(
            f"Total document content exceeds maximum length of {MAX_TEXT_BODY_CHARS:,} characters"
        )


def validate_conversation_length(messages: list[dict]) -> None:
    """Reject oversized conversation payloads."""
    total = sum(len(str(msg.get("content", ""))) for msg in messages)
    if total > MAX_TEXT_BODY_CHARS:
        raise ValueError(
            f"Conversation content exceeds maximum length of {MAX_TEXT_BODY_CHARS:,} characters"
        )


def is_private_or_reserved_host(host: str) -> bool:
    """Return True if host resolves to private, loopback, or link-local space."""
    if not host:
        return True
    host = host.strip("[]").lower()
    if host in ("localhost", "localhost.localdomain"):
        return True
    try:
        addr = ipaddress.ip_address(host)
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
        )
    except ValueError:
        return False


def validate_remote_base_url(url: str) -> Optional[str]:
    """
    Validate remote_base_url for SSRF safety.

    Returns a warning string if unsafe, else None.
    Ollama URLs are intentionally not checked here (local by design).
    """
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"remote_base_url must use http or https, got: {parsed.scheme or '(none)'}"
    host = parsed.hostname or ""
    if is_private_or_reserved_host(host):
        return (
            f"remote_base_url points to a private/reserved host ({host}). "
            "Only use this if you trust the endpoint."
        )
    return None


def get_client_ip(request) -> str:
    """
    Resolve client IP for rate limiting.

    X-Forwarded-For is only honored when the direct peer is a trusted proxy.
    """
    direct_ip = request.client.host if request.client else "unknown"
    trusted = os.getenv("LATENTGATE_TRUSTED_PROXY", "")
    if not trusted:
        return direct_ip

    trusted_ips = {ip.strip() for ip in trusted.split(",") if ip.strip()}
    if direct_ip not in trusted_ips:
        return direct_ip

    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return direct_ip
