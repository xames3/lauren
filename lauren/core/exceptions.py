"""\
Exception
=========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, July 23 2025
Last updated on: Friday, July 25 2025

This module defines custom exceptions for the framework. These exceptions
are used to handle errors related to security, validation, and other
framework-specific issues. They provide a consistent way to manage errors
across the framework, allowing for better error handling and debugging.
"""

from __future__ import annotations

import typing as t

__all__: list[str] = [
    "LaurenException",
    "SecurityError",
    "ValidationError",
]


class LaurenException(Exception):
    """Base exception class for all exceptions in the framework.

    This class serves as the base for all custom exceptions in the
    framework, allowing for consistent error handling and logging.

    :param message: The error message to be displayed.
    """

    def __init__(self, message: str, *args: t.Any) -> None:
        """Initialise the exception with a message and optional args."""
        super().__init__(message, *args)
        self.message = message

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        extra = ""
        if self.args:
            extra += f", {self.args!r})>"
        return f"<{type(self).__name__}(message={self.message!r}{extra})>"


class SecurityError(LaurenException):
    """Exception raised for security-related issues.

    This exception is used to indicate security violations or issues
    that need to be addressed, such as authentication failures or
    access control violations.

    :param message: The error message to be displayed.
    """

    def __init__(self, message: str, *, component: str | None = None) -> None:
        """Initialise the security error with context."""
        super().__init__(message)
        if component:
            self.message = f"{message} (Component: {component!r})"
        else:
            self.message = message
        self.component = component


class ValidationError(LaurenException):
    """Exception raised for validation errors.

    This exception is used when input data or configuration does not
    meet the required criteria, such as missing fields or incorrect
    types.

    :param message: The error message to be displayed.
    """

    def __init__(self, message: str, *, attribute: str | None = None) -> None:
        """Initialise the validation error with context."""
        super().__init__(message)
        if attribute:
            self.message = f"{message} (Attribute: {attribute!r})"
        else:
            self.message = message
        self.attribute = attribute
