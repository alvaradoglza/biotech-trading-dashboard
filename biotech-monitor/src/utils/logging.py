"""Structured logging with context support."""

import logging
import sys
from datetime import datetime
from typing import Any


class ContextLogger:
    """Logger wrapper that supports structured context."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with context key-value pairs."""
        if not kwargs:
            return message
        context_parts = [f"{k}={v}" for k, v in kwargs.items()]
        return f"{message} | {' '.join(context_parts)}"

    def debug(self, message: str, **kwargs: Any) -> None:
        self._logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        self._logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        self._logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        self._logger.error(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        self._logger.exception(self._format_message(message, **kwargs))


def get_logger(name: str) -> ContextLogger:
    """Get a structured logger for the given module name.

    Args:
        name: Module name, typically __name__

    Returns:
        ContextLogger instance with structured logging support
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return ContextLogger(logger)


def setup_file_logging(log_dir: str = "logs") -> None:
    """Setup file logging for the application.

    Args:
        log_dir: Directory to store log files
    """
    import os

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"biopharma_{datetime.now().strftime('%Y%m%d')}.log"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
