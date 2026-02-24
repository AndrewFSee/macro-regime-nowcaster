"""Structured logging configuration using loguru.

Call :func:`setup_logging` once at application startup to configure
log levels, formatting, and optional file output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru logging for the application.

    Args:
        level: Minimum log level (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, …).
        log_file: Optional path to a log file.  If *None*, logs to stderr only.
        rotation: loguru rotation policy for the log file.
        retention: loguru retention policy for the log file.
    """
    logger.remove()  # Remove default handler

    # Console handler with colour
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
            rotation=rotation,
            retention=retention,
            enqueue=True,
        )
