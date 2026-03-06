"""Tests for the logging configuration utility."""

from __future__ import annotations

import shutil
import tempfile

from loguru import logger
import pytest

from src.utils.logging_config import setup_logging


@pytest.fixture
def scratch_dir():
    d = tempfile.mkdtemp(prefix="nowcaster_log_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_setup_logging_does_not_raise():
    """setup_logging should work without error for any valid level."""
    setup_logging(level="DEBUG")
    setup_logging(level="INFO")
    setup_logging(level="WARNING")
    setup_logging(level="ERROR")


def test_setup_logging_with_file(scratch_dir):
    """setup_logging should accept a file path without error."""
    import os
    log_file = os.path.join(scratch_dir, "test.log")
    setup_logging(level="DEBUG", log_file=log_file)
    # Verify logger is usable
    logger.info("test message")
