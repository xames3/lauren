"""\
L.A.U.R.E.N
===========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, August 02 2025
Last updated on: Saturday, August 02 2025

Large AI Utility for Research and Engineering Needs

This package (lauren) is an end-to-end, open-source, enterprise-grade
toolkit for building, customising, deploying, and scaling bespoke
language models (LLM and SLM).

L.A.U.R.E.N is community-driven and engineered for those who wish to
build real, maintainable, and extensible language model powered
applications. It is suitable for both production teams and open-source
contributors who value transparency, reliability, and best practices in
software engineering.

PS: The name L.A.U.R.E.N is a nod to my close friend, Lauren.

Read complete documentation at: https://github.com/xames3/lauren.
"""

from __future__ import annotations

from .core import *
from .utils import *


__all__: tuple[str, ...] = ("__version__",)
__all__ += core.__all__
__all__ += utils.__all__

__version__: str = "31.8.2025"
