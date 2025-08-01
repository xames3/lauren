"""\
Error and warning objects
=========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, July 28 2025
Last updated on: Wednesday, July 30 2025

This module provides various error and warning utilities that are used
throughout the framework.
"""

from __future__ import annotations

__all__: tuple[str, ...] = (
    "ApplicationError",
    "BaseError",
    "ComponentError",
    "ConfigValidationError",
    "ContextError",
    "GuardrailError",
    "InspectionError",
    "PluginError",
    "PolicyError",
    "SecurityError",
    "ValidationError",
)

Error = Exception


class BaseError(Error):
    """Base error class for all exceptions."""


class SecurityError(BaseError):
    """Errors related to security, policies, and inspection objects."""


class ContextError(BaseError):
    """Errors related to failure in context-bound operation."""


class ComponentError(BaseError):
    """Errors related to failure in lifecycle or execution component."""


class ApplicationError(BaseError):
    """Errors related to application runtime errors."""


class PluginError(BaseError):
    """Errors related to plugin runtime errors."""


class InspectionError(SecurityError):
    """Errors related to security inspection or guard check failure."""


class PolicyError(SecurityError):
    """Errors related to policy runtime violation."""


class ValidationError(SecurityError):
    """Errors related to validation check failure."""


class GuardrailError(SecurityError):
    """Errors related to guardrail check failure."""


class ConfigValidationError(ValidationError):
    """Errors related to configuration validation failure."""
