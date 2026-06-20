"""Logging configuration for ``mdfs``.

Library modules use ``logging.getLogger(__name__)`` and never configure handlers.
Applications, scripts, and notebooks call :func:`configure_logging` once.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging idempotently with a compact, timestamped format."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
