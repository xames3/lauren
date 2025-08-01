"""\
Filesystem utility objects
==========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, July 30 2025
Last updated on: Wednesday, July 30 2025

This module provides various utilities that are used throughout the
framework.
"""

from __future__ import annotations

import os

__all__: tuple[str, ...] = (
    "mkdir",
    "touch",
)


def mkdir(path: str) -> str:
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def touch(path: str) -> str:
    """Create a file if it does not exist."""
    if not os.path.exists(path):
        with open(path, "a"):
            os.utime(path, None)
    return path
