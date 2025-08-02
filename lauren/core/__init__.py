"""\
Core
====

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, August 02 2025
Last updated on: Saturday, August 02 2025

This module acts as an entry point for combining various core objects
and configurations used throughout this framework.
"""

from __future__ import annotations

from .config import *
from .error import *


__all__: tuple[str, ...] = config.__all__ + error.__all__
