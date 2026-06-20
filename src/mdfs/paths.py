"""Centralized filesystem paths and resource locations for ``mdfs``.

Override the default force field by setting the ``MDFS_FFXML`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# ``.../src/mdfs``
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent

# Repository root (one level above ``src/``); only meaningful in a source checkout.
REPO_ROOT: Final[Path] = PACKAGE_ROOT.parent.parent

# Bundled force fields shipped as package data.
FFXML_DIR: Final[Path] = PACKAGE_ROOT / "ffxml"


def _default_ffxml() -> Path:
    env = os.environ.get("MDFS_FFXML")
    if env:
        return Path(env)
    return FFXML_DIR / "amber19" / "protein.ff19SB.xml"


DEFAULT_FFXML: Final[Path] = _default_ffxml()
