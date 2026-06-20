"""Tests for utility helpers."""

from __future__ import annotations

import logging

from mdfs.utils import configure_logging


def test_configure_logging_idempotent():
    configure_logging(logging.WARNING)
    root = logging.getLogger()
    n_first = len(root.handlers)
    configure_logging(logging.INFO)  # should reset, not accumulate handlers
    assert len(root.handlers) == n_first == 1
    assert root.level == logging.INFO
