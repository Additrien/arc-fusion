"""
Arc-Fusion Logging Configuration

Centralized logging setup for production-ready structured logging.
Replaces all print statements with proper logging throughout the application.
"""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
from contextvars import ContextVar

# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_obj['request_id'] = request_id
            
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_obj[key] = value
                
        return json.dumps(log_obj, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Build formatted message
        parts = [
            f"{color}{record.levelname:<8}{reset}",
            f"{timestamp}",
            f"{record.name}",
        ]
        
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            parts.append(f"[{request_id[:8]}]")
            
        parts.append(f"{record.getMessage()}")
        
        formatted = " | ".join(parts)
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    enable_file_logging: bool = True,
    log_file_path: str = "logs/arc-fusion.log"
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("console" for development, "json" for production)
        enable_file_logging: Whether to enable file logging
        log_file_path: Path to log file
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if needed
    if enable_file_logging:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Configure formatters
    formatters = {
        'json': {
            '()': JSONFormatter,
        },
        'console': {
            '()': ColoredConsoleFormatter,
        },
        'simple': {
            'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    }
    
    # Configure handlers
    handlers = {
        'console': {
            'class': 'logging.StreamHandler',
            'level': numeric_level,
            'formatter': log_format if log_format in formatters else 'console',
            'stream': sys.stdout,
        }
    }
    
    # Add file handler if enabled
    if enable_file_logging:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': numeric_level,
            'formatter': 'json',  # Always use JSON for file logs
            'filename': log_file_path,
            'maxBytes': 10_000_000,  # 10MB
            'backupCount': 5,
        }
    
    # Configure root loggers
    root_handlers = ['console']
    if enable_file_logging:
        root_handlers.append('file')
        
    # Logging configuration dictionary
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': formatters,
        'handlers': handlers,
        'loggers': {
            # Arc-Fusion specific loggers
            'arc_fusion': {
                'level': numeric_level,
                'handlers': root_handlers,
                'propagate': False,
            },
            'arc_fusion.api': {
                'level': numeric_level,
                'handlers': root_handlers,
                'propagate': False,
            },
            'arc_fusion.document_processor': {
                'level': numeric_level,
                'handlers': root_handlers,
                'propagate': False,
            },
            'arc_fusion.vector_store': {
                'level': numeric_level,
                'handlers': root_handlers,
                'propagate': False,
            },
            'arc_fusion.scripts': {
                'level': numeric_level,
                'handlers': root_handlers,
                'propagate': False,
            },
            # Suppress noisy third-party loggers
            'weaviate': {
                'level': 'WARNING',
                'handlers': root_handlers,
                'propagate': False,
            },
            'urllib3': {
                'level': 'WARNING',
                'handlers': root_handlers,
                'propagate': False,
            },
            'requests': {
                'level': 'WARNING',
                'handlers': root_handlers,
                'propagate': False,
            },
        },
        'root': {
            'level': 'WARNING',  # Only show warnings/errors from other libraries
            'handlers': root_handlers,
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Test logging setup
    logger = get_logger('arc_fusion')
    logger.info(f"Logging initialized", extra={
        'log_level': log_level,
        'log_format': log_format,
        'file_logging': enable_file_logging
    })


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (e.g., 'arc_fusion.api', 'arc_fusion.document_processor')
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID for context-aware logging.
    
    Args:
        request_id: Optional request ID, generates one if not provided
        
    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear the request ID from context."""
    request_id_var.set(None)


# Convenience function to initialize logging from environment
def init_logging_from_env() -> None:
    """Initialize logging using environment variables."""
    setup_logging(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_format=os.getenv("LOG_FORMAT", "console"),  # "console" for dev, "json" for prod
        enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
        log_file_path=os.getenv("LOG_FILE_PATH", "logs/arc-fusion.log")
    )
