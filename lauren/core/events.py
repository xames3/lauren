from __future__ import annotations

import typing as t
from enum import Enum
from typing import Final

__all__ = [
    "EventCategory",
    "EventSeverity",
    "LIFECYCLE_EVENTS",
    "SECURITY_EVENTS",
    "OPERATION_EVENTS",
    "ERROR_EVENTS",
    "PERFORMANCE_EVENTS",
    "CONFIGURATION_EVENTS",
    "INTEGRATION_EVENTS",
    "EVENTS",
]


class EventCategory(Enum):
    """Event classification for audit trail organisation."""

    LIFECYCLE = "lifecycle"
    SECURITY = "security"
    OPERATION = "operation"
    ERROR = "error"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"


class EventSeverity(Enum):
    """Event severity levels for filtering and alerting."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


LIFECYCLE_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "component_initialising": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component beginning initialisation process",
    },
    "component_initialised": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component successfully initialised",
    },
    "component_created": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component instance created",
    },
    "component_started": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component started and ready for operations",
    },
    "component_stopping": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component beginning shutdown process",
    },
    "component_stopped": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component successfully stopped",
    },
    "application_initialising": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Application starting up",
    },
    "application_created": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Application instance created",
    },
    "application_ready": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Application fully initialised and ready",
    },
    "application_shutdown": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Application shutting down gracefully",
    },
    "initialisation": {
        "category": EventCategory.LIFECYCLE,
        "severity": EventSeverity.INFO,
        "description": "Component or application initialisation",
    },
}

SECURITY_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "policy_evaluation": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.DEBUG,
        "description": "Security policy being evaluated",
    },
    "policy_allowed": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.INFO,
        "description": "Operation permitted by security policy",
    },
    "policy_denied": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.WARNING,
        "description": "Operation denied by security policy",
    },
    "validation_started": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.DEBUG,
        "description": "Component validation beginning",
    },
    "validation_passed": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.INFO,
        "description": "Component passed validation checks",
    },
    "validation_failed": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.ERROR,
        "description": "Component failed validation checks",
    },
    "guardian_protection_triggered": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.WARNING,
        "description": "Guardian prevented unauthorised access",
    },
    "access_denied": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.WARNING,
        "description": "Access attempt rejected",
    },
    "privilege_escalation_attempt": {
        "category": EventCategory.SECURITY,
        "severity": EventSeverity.CRITICAL,
        "description": "Attempted privilege escalation detected",
    },
}

OPERATION_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "operation_started": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.DEBUG,
        "description": "Operation beginning execution",
    },
    "operation_completed": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Operation completed successfully",
    },
    "operation_cancelled": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.WARNING,
        "description": "Operation was cancelled before completion",
    },
    "plugin_discovered": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin found during discovery process",
    },
    "plugin_loaded": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin successfully loaded and validated",
    },
    "plugin_created": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin instance created",
    },
    "plugin_executed": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin execution completed",
    },
    "plugin_activated": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin activated for execution",
    },
    "plugin_deactivated": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin deactivated from execution",
    },
    "plugin_unloaded": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin unloaded from system",
    },
    "plugins_discovered": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin discovery process completed",
    },
    "plugins_loaded": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Plugin loading process completed",
    },
    "context_created": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.DEBUG,
        "description": "Execution context created",
    },
    "context_updated": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.DEBUG,
        "description": "Execution context modified",
    },
    "context_destroyed": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.DEBUG,
        "description": "Execution context cleaned up",
    },
    "context_reset": {
        "category": EventCategory.OPERATION,
        "severity": EventSeverity.INFO,
        "description": "Context reset to initial state",
    },
}

ERROR_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "error_occurred": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Unexpected error during operation",
    },
    "exception_caught": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Exception handled gracefully",
    },
    "validation_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Data validation failed",
    },
    "configuration_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Configuration validation failed",
    },
    "plugin_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Plugin execution encountered error",
    },
    "plugin_load_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Plugin loading failed",
    },
    "plugin_validation_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Plugin validation failed",
    },
    "resource_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Resource allocation or access failed",
    },
    "timeout_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Operation exceeded timeout limit",
    },
    "plugin_load_timeout": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Plugin loading exceeded timeout limit",
    },
    "critical_failure": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.CRITICAL,
        "description": "System-wide critical failure occurred",
    },
    "inspector_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Inspector execution encountered error",
    },
    "module_load_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.ERROR,
        "description": "Module loading failed",
    },
    "plugin_discovery_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Plugin discovery encountered error",
    },
    "plugin_discovery_scan_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Plugin directory scan failed",
    },
    "entry_point_discovery_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Entry point discovery failed",
    },
    "plugin_cleanup_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Plugin cleanup encountered error",
    },
    "memory_check_error": {
        "category": EventCategory.ERROR,
        "severity": EventSeverity.WARNING,
        "description": "Memory usage check failed",
    },
}

PERFORMANCE_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "operation_timing": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.DEBUG,
        "description": "Operation execution timing recorded",
    },
    "memory_usage": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.DEBUG,
        "description": "Memory consumption metrics captured",
    },
    "performance_threshold_exceeded": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.WARNING,
        "description": "Performance metric exceeded threshold",
    },
    "resource_limit_reached": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.WARNING,
        "description": "System resource limit approached",
    },
    "bottleneck_detected": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.WARNING,
        "description": "Performance bottleneck identified",
    },
    "plugin_memory_limit_exceeded": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.WARNING,
        "description": "Plugin memory usage exceeded limit",
    },
    "plugin_scan_depth_exceeded": {
        "category": EventCategory.PERFORMANCE,
        "severity": EventSeverity.WARNING,
        "description": "Plugin directory scan exceeded depth limit",
    },
}

CONFIGURATION_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "configuration_loaded": {
        "category": EventCategory.CONFIGURATION,
        "severity": EventSeverity.INFO,
        "description": "Configuration successfully loaded",
    },
    "configuration_updated": {
        "category": EventCategory.CONFIGURATION,
        "severity": EventSeverity.INFO,
        "description": "Configuration values modified",
    },
    "configuration_validated": {
        "category": EventCategory.CONFIGURATION,
        "severity": EventSeverity.INFO,
        "description": "Configuration passed validation",
    },
    "configuration_error": {
        "category": EventCategory.CONFIGURATION,
        "severity": EventSeverity.ERROR,
        "description": "Configuration validation or loading failed",
    },
    "environment_detected": {
        "category": EventCategory.CONFIGURATION,
        "severity": EventSeverity.INFO,
        "description": "Runtime environment configuration detected",
    },
}

INTEGRATION_EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    "external_service_called": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "External service integration invoked",
    },
    "external_service_response": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Response received from external service",
    },
    "integration_failed": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.ERROR,
        "description": "External service integration failed",
    },
    "api_request": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "API request processed",
    },
    "database_query": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Database query executed",
    },
    "message_published": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Message published to queue or topic",
    },
    "message_consumed": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Message consumed from queue or topic",
    },
    "app_started": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.INFO,
        "description": "Application started event notification",
    },
    "app_stopped": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.INFO,
        "description": "Application stopped event notification",
    },
    "request_processed": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Request processing completed",
    },
    "plugin_execution_started": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Plugin execution started notification",
    },
    "plugin_execution_completed": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.DEBUG,
        "description": "Plugin execution completed notification",
    },
    "component_cleaned_up": {
        "category": EventCategory.INTEGRATION,
        "severity": EventSeverity.INFO,
        "description": "Component cleanup completed notification",
    },
}

EVENTS: Final[dict[str, dict[str, t.Any]]] = {
    **LIFECYCLE_EVENTS,
    **SECURITY_EVENTS,
    **OPERATION_EVENTS,
    **ERROR_EVENTS,
    **PERFORMANCE_EVENTS,
    **CONFIGURATION_EVENTS,
    **INTEGRATION_EVENTS,
}
