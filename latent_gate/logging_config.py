"""
Structured Logging Configuration.

Provides JSON-formatted logging with log rotation and correlation IDs.

Features:
  - JSON-formatted log output
  - Log rotation by size or time
  - Correlation IDs for request tracking
  - Configurable log levels per module
  - Performance logging
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict
from contextvars import ContextVar


# ============================================================================
# Correlation ID
# ============================================================================

correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


# ============================================================================
# JSON Formatter
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.
    
    Output format:
    {
        "timestamp": "2024-01-01T12:00:00.000Z",
        "level": "INFO",
        "logger": "latent_gate.pipeline",
        "message": "Processing image",
        "correlation_id": "abc-123",
        "extra": {...}
    }
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            log_data["correlation_id"] = corr_id
        
        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add performance metrics if present
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        if hasattr(record, "tokens"):
            log_data["tokens"] = record.tokens
        
        if hasattr(record, "cost"):
            log_data["cost"] = record.cost
        
        return json.dumps(log_data, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter with color support.
    
    Format: [TIMESTAMP] LEVEL | logger | message
    """
    
    # Colors for different log levels
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            level = f"{color}{record.levelname:8s}{self.RESET}"
        else:
            level = record.levelname
        
        # Shorten logger name
        logger_name = record.name.replace("latent_gate.", "")
        
        message = record.getMessage()
        
        # Add correlation ID if present
        corr_id = correlation_id.get()
        if corr_id:
            message = f"[{corr_id[:8]}] {message}"
        
        # Add duration if present
        if hasattr(record, "duration_ms"):
            message = f"{message} ({record.duration_ms:.1f}ms)"
        
        return f"{timestamp} | {level} | {logger_name} | {message}"


# ============================================================================
# Log Rotation
# ============================================================================

def create_rotating_handler(
    filename: str,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = True,
) -> logging.Handler:
    """
    Create a rotating file handler.
    
    Args:
        filename: Log file path
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        json_format: Use JSON formatting
        
    Returns:
        Configured logging handler
    """
    handler = logging.handlers.RotatingFileHandler(
        filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter(use_colors=False))
    
    return handler


def create_timed_handler(
    filename: str,
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 30,
    json_format: bool = True,
) -> logging.Handler:
    """
    Create a timed rotating file handler.
    
    Args:
        filename: Log file path
        when: Rotation time (midnight, h0, h1, etc.)
        interval: Rotation interval
        backup_count: Number of backup files to keep
        json_format: Use JSON formatting
        
    Returns:
        Configured logging handler
    """
    handler = logging.handlers.TimedRotatingFileHandler(
        filename,
        when=when,
        interval=interval,
        backupCount=backup_count,
    )
    
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(HumanReadableFormatter(use_colors=False))
    
    return handler


# ============================================================================
# Configuration
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    log_rotation: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    module_levels: Optional[Dict[str, str]] = None,
):
    """
    Set up logging configuration.
    
    Args:
        level: Root log level
        log_file: Optional log file path
        json_format: Use JSON formatting
        log_rotation: Enable log rotation
        max_bytes: Maximum log file size for rotation
        backup_count: Number of backup files to keep
        module_levels: Per-module log levels
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(HumanReadableFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        if log_rotation:
            file_handler = create_rotating_handler(
                log_file,
                max_bytes=max_bytes,
                backup_count=backup_count,
                json_format=json_format,
            )
        else:
            file_handler = logging.FileHandler(log_file)
            if json_format:
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(HumanReadableFormatter(use_colors=False))
        
        root_logger.addHandler(file_handler)
    
    # Set per-module levels
    if module_levels:
        for module, module_level in module_levels.items():
            module_logger = logging.getLogger(module)
            module_logger.setLevel(getattr(logging, module_level.upper(), logging.INFO))
    
    # Configure LatentGate modules
    latentgate_modules = [
        "latent_gate.pipeline",
        "latent_gate.local",
        "latent_gate.text",
        "latent_gate.remote",
        "latent_gate.selective",
        "latent_gate.api",
        "latent_gate.video",
        "latent_gate.cost",
    ]
    
    for module in latentgate_modules:
        if module not in (module_levels or {}):
            logging.getLogger(module).setLevel(getattr(logging, level.upper(), logging.INFO))


def setup_from_env():
    """
    Set up logging from environment variables.
    
    Environment variables:
        LATENTGATE_LOG_LEVEL: Root log level
        LATENTGATE_LOG_FILE: Log file path
        LATENTGATE_LOG_JSON: Use JSON format (true/false)
        LATENTGATE_LOG_ROTATION: Enable rotation (true/false)
    """
    level = os.getenv("LATENTGATE_LOG_LEVEL", "INFO")
    log_file = os.getenv("LATENTGATE_LOG_FILE")
    json_format = os.getenv("LATENTGATE_LOG_JSON", "").lower() in ("true", "1", "yes")
    log_rotation = os.getenv("LATENTGATE_LOG_ROTATION", "").lower() in ("true", "1", "yes")
    
    setup_logging(
        level=level,
        log_file=log_file,
        json_format=json_format,
        log_rotation=log_rotation,
    )


# ============================================================================
# Performance Logging
# ============================================================================

class PerformanceLogger:
    """
    Context manager for logging performance metrics.
    
    Usage:
        with PerformanceLogger("image_processing") as perf:
            # Do work
            perf.tokens = 150
            perf.cost = 0.001
    """
    
    def __init__(self, operation: str, logger_name: str = "latent_gate"):
        self.operation = operation
        self.logger = logging.getLogger(logger_name)
        self.tokens: Optional[int] = None
        self.cost: Optional[float] = None
        self.start_time: float = 0
        self.duration_ms: float = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.duration_ms = (time.time() - self.start_time) * 1000
        
        extra = {
            "duration_ms": self.duration_ms,
            "operation": self.operation,
        }
        
        if self.tokens is not None:
            extra["tokens"] = self.tokens
        
        if self.cost is not None:
            extra["cost"] = self.cost
        
        # Create log record with extra data
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"{self.operation} completed",
            args=(),
            exc_info=None,
        )
        record.extra_data = extra
        record.duration_ms = self.duration_ms
        
        if self.tokens is not None:
            record.tokens = self.tokens
        
        if self.cost is not None:
            record.cost = self.cost
        
        self.logger.handle(record)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def set_correlation_id(cid: str):
    """Set the correlation ID for the current context."""
    correlation_id.set(cid)


def clear_correlation_id():
    """Clear the correlation ID."""
    correlation_id.set(None)
