"""\
Base Tools
==========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, July 22 2025
Last updated on: Monday, July 28 2025

Base components.

This module provides foundational classes and utilities for building
robust, observable, and secure components as part of this framework.
These building blocks support creating extensible and well-structured
applications with comprehensive audit trails, validations, and
inspection capabilities.
"""

from __future__ import annotations

import time
import types
import typing as t
from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from uuid import uuid4

from lauren.core.events import EVENTS
from lauren.core.events import EventCategory
from lauren.core.events import EventSeverity

__all__: Sequence[str] = [
    "AuditLog",
    "Component",
    "ExecutionContext",
    "Guardian",
    "Inspector",
    "MetricCollector",
    "Observable",
    "Policy",
    "Validator",
]

_AttributeStream = Iterator[tuple[str, t.Any]]
_StateInfoDict = dict[str, t.Any]
_ContextInfoDict = dict[str, t.Any]

# NOTE(xames3): These limits are used to prevent excessive output and
# are not intended to be changed by users. As such, they are defined as
# final constants. Currently, these are set to reasonable defaults to
# prevent performance issues and excessive logging. Maybe in the future
# we can make them configurable using `lauren.core.Config` object.
_SEQUENCE_LIMIT: t.Final[int] = 5
_DICTIONARY_LIMIT: t.Final[int] = 3
_STRING_LIMIT: t.Final[int] = 60

_SEVERITY_LEVEL_MAP: dict[str, int] = {
    EventSeverity.DEBUG: 0,
    EventSeverity.INFO: 1,
    EventSeverity.WARNING: 2,
    EventSeverity.ERROR: 3,
    EventSeverity.CRITICAL: 4,
}


class Observable:
    """Provide observable behaviour for derived classes.

    This class serves as a foundational mixin class for all the derived
    components that need to expose their internal state in a consistent
    and controlled manner. It primarily offers formatting and string
    representation features for derived classes, making it
    straightforward to inspect and debug complex objects.

    This class handles different data types and presents them as proper
    key-value pairs and format their string representation with sensible
    limits for various data types to prevent excessive output.

    .. note::

        This class uses `__slots__` to reduce memory usage and only
        includes the `__weakref__` slot to allow weak references. This
        makes it safe to use as a mixin without introducing significant
        overhead.

    .. warning::

        The introspection method (`__inspect_attrs__`) will only show
        public attributes and properties of the derived class. It does
        not include private attributes or methods by design.
    """

    __slots__: tuple[str] = ("__weakref__",)

    def __inspect_attrs__(self) -> _AttributeStream:
        """Inspect and yield public attributes of the instance.

        This method is useful for debugging and introspection of the
        component's state.

        :yield: An iterator yielding tuples of attribute names and their
            corresponding values.

        .. note::

            Only attributes that do not start with underscore and are
            not None are included in the introspection output.
        """
        slots = getattr(self, "__slots__: tuple[str]", ())
        attrs = slots if slots else getattr(self, "__dict__", {}).keys()
        for attr in attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if not attr.startswith("_") and value is not None:
                    yield attr, value

    def _format(self, value: t.Any) -> str:
        """Format value for string representation.

        This method formats values for string representation, applying
        sensible limits to prevent overly long outputs. Circular
        references are replaced with type indicators, long strings are
        truncated, and large sequences and dictionaries show their type
        and length rather than their full contents.

        :param value: The value to format.
        :return: A formatted string representation of the value.
        """
        if value is self:
            return f"<circular-{type(self).__name__}>"
        elif isinstance(value, str) and len(value) > _STRING_LIMIT:
            return f"{value[:_STRING_LIMIT - 3]}..."
        elif (
            isinstance(value, (list, tuple, set))
            and len(value) > _SEQUENCE_LIMIT
        ):
            return f"{type(value).__name__}({len(value)} items)"
        elif isinstance(value, dict) and len(value) > _DICTIONARY_LIMIT:
            return f"dict({len(value)} items)"
        return repr(value)

    def __repr__(self) -> str:
        """Return a string representation of the instance.

        This method creates a string representation that includes the
        class name and formatted non-private attributes. The
        representation is designed to be concise yet informative, with
        large data structures summarised rather than fully expanded.

        :return: A string representation of the instance.
        """
        attrs = [
            f"{name}={self._format(value)}"
            for name, value in self.__inspect_attrs__()
        ]
        extra = ", ".join(attrs) if attrs else ""
        return f"{type(self).__name__}({extra})"


class Validator(Observable, ABC):
    """Validate components against defined rules and constraints.

    This class provides a mechanism to verify that components meet
    expected requirements before they are used. It is an abstract class
    that defines the interface for different validation strategies.

    Validators can be attached to components and contexts to ensure they
    maintain expected invariants and meet requirements throughout their
    lifecycle. Each validator can observe specific events and be
    activated or deactivated as needed.

    :param name: Name for the validator, defaults to `None`.
    :param description: Optional description of the validator as to what
        this validator checks or validates, defaults to `None`.
    :param strict: Whether to enforce strict validation mode, defaults
        to `False`.

    .. note::

        This is an abstract base class. Concrete validators must
        implement the `__call__` method to perform the actual validation
        logic.

    .. seealso::

        :class:`Component` for information on how validators are
        integrated into component validation workflows.

    .. code-block:: python

        class ConfigValidator(Validator):
           async def __call__(self, target, *, context=None):
               return (
                    hasattr(target, "config")
                    and target.config is not None
                )
    """

    __slots__: tuple[str] = ("_name", "_description", "_strict")

    def __init__(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
    ) -> None:
        """Initialise a validator."""
        self._name = name or type(self).__name__
        self._description = description
        self._strict = strict

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield validator's attributes for introspection."""
        yield "name", self._name
        if self._description:
            yield "description", self._description
        if self._strict:
            yield "strict", self._strict

    @abstractmethod
    async def __call__(
        self,
        target: t.Any,
        *,
        context: _ContextInfoDict | None = None,
    ) -> bool:
        """Validate the target against defined rules.

        This abstract method must be implemented by the concrete
        validator classes to perform actual validation logic. It should
        return `True` if validation passes, otherwise `False`.

        :param target: The target object to validate.
        :param context: Optional context information for validation,
            such as additional data or state that may influence the
            validation process, defaults to `None`.
        :return: `True` if validation passes, otherwise `False`.

        .. note::

            Validation methods should be idempotent and not modify
            the target object. They should focus on checking invariants
            and constraints rather than performing corrections.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    @property
    def name(self) -> str:
        """Get the name of the validator."""
        return self._name

    @property
    def description(self) -> str | None:
        """Get the description of the validator."""
        return self._description

    @property
    def strict(self) -> bool:
        """Check if the validator is in strict mode."""
        return self._strict


class Inspector(Observable, ABC):
    """Monitor and inspect events within a system.

    This class provides a mechanism to observing events that occur
    within components. It allows for monitoring, debugging, and
    recording system behaviour without modifying the components' core
    functionality.

    Inspectors receive notifications about events along with contextual
    data, enabling them to track system state and behaviour over time.
    They can be configured to observe specific events and can be activated
    or deactivated as needed.

    :param name: Name for the inspector, defaults to `None`.
    :param events: Set of event names to observe, defaults to `None`.
    :param active: Whether the inspector is active and should receive
        events, defaults to `True`.

    .. note::

        This is an abstract base class. Concrete inspectors must
        implement the `__call__` method to handle event notifications.

    .. seealso::

        :class:`Component` for information on how inspectors are
        integrated into component event notification workflows.

    .. code-block:: python

        class PerformanceInspector(Inspector):

            def __init__(self):
                super().__init__(
                    name="PerformanceMonitor",
                    events={"operation_started", "operation_completed"},
                )
                self.timings = {}

            async def __call__(self, event, subject, data):
                if event == "operation_started":
                    self.timings[subject] = time.time()
                elif event == "operation_completed":
                    started_at = self.timings.pop(subject, None)
                    if started_at:
                        took = time.time() - started_at
                        print(f"Operation {subject} took {took:.2f}s")
    """

    __slots__: tuple[str] = ("_name", "_events", "_active")

    def __init__(
        self,
        *,
        name: str | None = None,
        events: set[str] | None = None,
        active: bool = True,
    ) -> None:
        """Initialise an inspector with name and events to observe."""
        self._name = name or type(self).__name__
        self._events = events or set()
        self._active = active

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield inspector's attributes for introspection."""
        yield "name", self._name
        yield "active", self._active
        if self._events:
            yield "events", sorted(self._events)

    @abstractmethod
    async def __call__(
        self,
        event: str,
        subject: t.Any,
        data: dict[str, t.Any],
    ) -> None:
        """Process an event with associated subject and data.

        This abstract method must be implemented by concrete inspector
        classes to handle events they are interested in observing. It
        receives the event name, subject (source of the event), and
        associated data.

        :param event: The name of the event being observed.
        :param subject: The subject or source of the event.
        :param data: Additional data associated with the event, such as
            context information or parameters relevant to the event.

        .. note::

            Inspector methods should not raise exceptions as this could
            disrupt the component that generated the event. Consider
            logging errors or handling them gracefully within the
            inspector implementation.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    @property
    def name(self) -> str:
        """Get the name of the inspector."""
        return self._name

    @property
    def events(self) -> set[str]:
        """Get the set of events the inspector observes."""
        return self._events

    @property
    def active(self) -> bool:
        """Check if the inspector is active."""
        return self._active


class Policy(Observable, ABC):
    """Define security policies and access control rules.

    Policy provides a mechanism for enforcing access controls and
    security rules on operations. It determines whether operations
    should be permitted based on the subject, operation, and context.

    Policies can be attached to components to control access to their
    functionality and ensure operations comply with security
    requirements. Multiple policies can be combined to create
    sophisticated access control schemes.

    :param name: Name for the policy, defaults to `None`.
    :param operations: Set of operations this policy governs, defaults
        to `None`.
    :param level: The level of the policy, such as `strict`
        or `permissive`, defaults to `strict`.

    .. note::

        This is an abstract base class. Concrete policies must
        implement the `__call__` method to perform actual policy
        evaluation.

    .. seealso::

        :class:`Component` for information on how policies are
        integrated into component security workflows.

    .. code-block:: python

        class ReadOnlyPolicy(Policy):
            def __init__(self):
                super().__init__(
                    name="ReadOnlyAccess",
                    events={"read", "list"},
                )

            async def __call__(
                self,
                subject,
                operation=None,
                *,
                context=None,
            ):
                return operation in {"read", "list"}
    """

    __slots__: tuple[str] = ("_name", "_operations", "_level")

    def __init__(
        self,
        *,
        name: str | None = None,
        operations: set[str] | None = None,
        level: str = "strict",
    ) -> None:
        """Initialise a policy with a name and operations to govern."""
        self._name = name or type(self).__name__
        self._operations = operations or set()
        self._level = level

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield policy's attributes for introspection."""
        yield "name", self._name
        yield "level", self._level
        if self._operations:
            yield "operations", sorted(self._operations)

    @abstractmethod
    async def __call__(
        self,
        subject: t.Any,
        operation: str | None = None,
        *,
        context: _ContextInfoDict | None = None,
    ) -> bool:
        """Evaluate whether the operation is permitted or denied.

        This abstract method must be implemented by concrete policy
        classes to determine if operations are allowed. It should return
        `True` if the operation is permitted, `False` otherwise.

        :param subject: The entity attempting the operation.
        :param operation: The operation being attempted, defaults
            to `None`.
        :param context: Additional context for policy evaluation,
            defaults to `None`.
        :return: `True` if permitted, `False` otherwise.

        .. note::

            Policy evaluation should be deterministic and not have side
            effects. Consider logging policy decisions for audit
            purposes without affecting the policy outcome.
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    @property
    def name(self) -> str:
        """Get the name of the policy."""
        return self._name

    @property
    def operations(self) -> set[str]:
        """Get the set of operations governed by the policy."""
        return self._operations

    @property
    def level(self) -> str:
        """Get the enforcement level of the policy."""
        return self._level


class MetricCollector(Observable):
    """Collect and track operation metrics for performance tracking.

    Collector tracks metrics about operations, including counts,
    durations, success rates, and other performance indicators. It
    provides methods to record operations and retrieve metric summaries
    for monitoring and analysis purposes.

    The metric collector maintains lightweight statistics without
    requiring external dependencies.

    .. note::

        This class uses `__slots__` for memory efficiency and maintains
        simple statistics suitable for most monitoring needs.

    .. code-block:: python

        collector = MetricCollector()
        collector.record_operation(0.5, success=True)
        collector.record_operation(1.2, success=False)
        print(f"Success rate: {collector.metrics['success_rate']:.2%}")
        print(f"Avg duration: {collector.metrics['avg_duration']:.3f}s")
    """

    __slots__: tuple[str] = ("_data",)

    def __init__(self) -> None:
        """Initialise a new metric collector instance."""
        self._data: _StateInfoDict = {
            "created_at": time.time(),
            "operations": 0,
            "last_operation": None,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "fastest": float("inf"),
            "slowest": 0.0,
            "successes": 0,
            "failures": 0,
            "success_rate": 0.0,
        }

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield collector's attributes for introspection."""
        yield "operations", self._data["operations"]
        if self._data["operations"] > 0:
            yield "success_rate", f"{self._data['success_rate']:.1%}"
            if self._data["avg_duration"] > 0:
                yield "avg_duration", f"{self._data['avg_duration']:.3f}s"

    def __getitem__(self, key: str) -> t.Any:
        """Get a metric value by key."""
        return self._data[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        """Set a metric value by key."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a metric key exists."""
        return key in self._data

    def __len__(self) -> int:
        """Return the number of tracked metrics."""
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        """Iterate over metric names."""
        return iter(self._data)

    def _update_success_rate(self) -> None:
        """Calculate and update the current success rate.

        This internal method recalculates the success rate based on the
        current number of successful and failed operations.
        """
        total = self._data["successes"] + self._data["failures"]
        if total > 0:
            self._data["success_rate"] = self._data["successes"] / total

    def record_operation(
        self,
        duration: float,
        *,
        success: bool = True,
    ) -> None:
        """Record the execution of an operation.

        This method records a single operation execution, tracking its
        duration and whether it was successful. The metrics are
        aggregated internally for later retrieval and analysis.

        :param duration: Time taken by the operation in seconds.
        :param success: Whether the operation was successful, defaults
            to `True`.

        .. note::

            Duration should always be provided in seconds as a float. The
            success rates and timing statistics are automatically
            calculated from the recorded operations.
        """
        self._data["operations"] += 1
        self._data["last_operation"] = time.time()
        self._data["total_duration"] += duration
        if self._data["fastest"] == float("inf"):
            self._data["fastest"] = duration
        else:
            self._data["fastest"] = min(self._data["fastest"], duration)
        self._data["slowest"] = max(self._data["slowest"], duration)
        count = self._data["operations"]
        total = self._data["total_duration"]
        self._data["avg_duration"] = total / count if count > 0 else 0.0
        if success:
            self._data["successes"] += 1
        else:
            self._data["failures"] += 1
        self._update_success_rate()

    @property
    def metrics(self) -> _StateInfoDict:
        """Return a copy of all collected metrics."""
        return self._data.copy()


class AuditLog(Observable):
    """Record and query event logs for auditing and observability.

    This class maintains a chronological record of significant events
    that occur within a system. It provides methods to record events
    with detailed metadata and to query recorded events using various
    filters.

    Each event is stored with a timestamp, category, severity level, and
    custom metadata, allowing for comprehensive analysis and filtering.
    Events are automatically categorised and assigned severity levels
    based on predefined event definitions.

    .. note::

        This class stores events in memory and provides filtering
        capabilities for analysis. For production systems with high
        event volumes, consider implementing persistence or log
        rotation.

    .. seealso::

        :clas:`EventCategory` and :class:`EventSeverity` for available
        categorisation options.

    .. code-block:: python

        audit = AuditLog()
        audit.record_event(
            "user_login",
            component="AuthService",
            category=EventCategory.SECURITY,
            user_id="xames3",
        )
        events = audit.get_events_by_category(EventCategory.SECURITY)
        print(f"Found {len(events)} security events")
    """

    __slots__: tuple[str] = ("_entries",)

    def __init__(self) -> None:
        """Initialise a new audit logging instance."""
        self._entries: list[_StateInfoDict] = []

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield audit loggers's attributes for introspection."""
        yield "entries", len(self._entries)
        if self._entries:
            yield "latest", self._entries[-1].get("event", "unknown")

    def __len__(self) -> int:
        """Return the number of audit entries."""
        return len(self._entries)

    def __bool__(self) -> bool:
        """Check if the audit log contains any entries."""
        return bool(self._entries)

    def __getitem__(
        self, index: int | slice
    ) -> _StateInfoDict | list[_StateInfoDict]:
        """Get audit entries by index or slice."""
        return self._entries[index]

    def __iter__(self) -> Iterator[_StateInfoDict]:
        """Iterate over audit entries."""
        return iter(self._entries)

    def record_event(
        self,
        event: str,
        *,
        component: str,
        category: EventCategory | None = None,
        severity: EventSeverity | None = None,
        **metadata: t.Any,
    ) -> None:
        """Record an audit event.

        This method records a significant event that occurred within the
        system, along with detailed metadata about the event. Events are
        categorised and assigned severity levels to aid in filtering and
        analysis.

        :param event: Type or name of the event.
        :param component: Component that generated the event.
        :param category: The event category for filtering, defaults
            to `None`.
        :param severity: The event severity level, defaults to `None`.
        """
        e = EVENTS.get(event, {})
        entry = {
            "event": event,
            "event_id": str(uuid4())[:8],
            "timestamp": time.time(),
            "component": component,
            "category": category or e.get("category", EventCategory.OPERATION),
            "severity": severity or e.get("severity", EventSeverity.INFO),
            "description": e.get("description", f"Unknown event: {event}"),
            **metadata,
        }
        self._entries.append(entry)

    def get_events(
        self,
        event: str | None = None,
        component: str | None = None,
        category: EventCategory | None = None,
        severity: EventSeverity | None = None,
        limit: int | None = None,
    ) -> list[_StateInfoDict]:
        """Get filtered audit events.

        This method returns events that match the given filters. Events
        can be filtered by events, components, categories, and a minimum
        severity level. Results can be limited to a specific number of
        most recent events.

        :param event: Filter by specific event type, defaults to `None`.
        :param component: Filter by component name, defaults to `None`.
        :param category: Filter by event category, defaults to `None`.
        :param severity: Filter by minimum severity level, defaults
            to `None`.
        :param limit: Maximum number of events to return, defaults
            to `None`.
        :return: A list of matching events (audit entries).

        .. note::

            Filters are combined with `AND` logic; all specific
            conditions must be satisfied for an event to be included in
            the results. The limit applies to the most recent events
            after filtering.
        """
        filtered = self._entries
        if event:
            filtered = [e for e in filtered if e.get("event") == event]
        if component:
            filtered = [c for c in filtered if c.get("component") == component]
        if category:
            filtered = [c for c in filtered if c.get("category") == category]
        if severity:
            level = _SEVERITY_LEVEL_MAP[severity]
            filtered = [
                entry
                for entry in filtered
                if _SEVERITY_LEVEL_MAP.get(entry.get("severity", "info"), 0)
                >= level
            ]
        if limit:
            filtered = filtered[-limit:]
        return filtered.copy()

    def get_events_by_category(
        self,
        category: EventCategory,
        limit: int | None = None,
    ) -> list[_StateInfoDict]:
        """Retrieve events filtered by category.

        This method returns events that belong to the specified category.
        Results can be limited to a specific number of most recent
        events.

        :param category: Event category to filter by.
        :param limit: Maximum number of events to return, defaults
            to `None`.
        :return: A list of matching events.
        """
        return self.get_events(category=category, limit=limit)

    def get_events_by_severity(
        self,
        severity: EventSeverity,
        limit: int | None = None,
    ) -> list[_StateInfoDict]:
        """Retrieve events at or above specified severity level.

        This method returns events that have a severity level equal to
        or higher than the specified minimum severity. Results can be
        limited to a specific number of most recent events.

        :param severity: Minimum severity level to filter by.
        :param limit: Maximum number of events to return, defaults
            to `None`.
        :return: A list of matching events.

        .. warning::

            Critical and error events should be monitored closely as
            they may indicate system issues requiring immediate
            attention.
        """
        return self.get_events(severity=severity, limit=limit)

    def get_event_summary(self) -> dict[str, t.Any]:
        """Return a statistical summary of recorded events.

        This method provides an overview of the events recorded in the
        log, including total count, distribution by category and
        severity, and component breakdown.

        :return: A dictionary containing summary statistics.

        .. note::

            The summary includes total event count and breakdowns by
            category, severity, and component. This is useful for
            understanding system activity patterns and identifying
            components with high event volumes.
        """
        summary = {
            "total_events": len(self._entries),
            "by_category": {},
            "by_severity": {},
            "by_component": {},
        }
        for entry in self._entries:
            category = entry.get("category", "unknown")
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )
            severity = entry.get("severity", "unknown")
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )
            component = entry.get("component", "unknown")
            summary["by_component"][component] = (
                summary["by_component"].get(component, 0) + 1
            )
        return summary

    @property
    def entries(self) -> list[_StateInfoDict]:
        """Get a copy of all the recorded events."""
        return self._entries.copy()


class Guardian(Observable):
    """Enforce security boundaries and protect component integrity.

    This class provides a security mechanism to protect components from
    unauthorised access and modifications. It can enforce various
    security constraints such as preventing post-initialisation changes,
    protecting private attributes, and ensuring strict access controls.

    Each component can have a guardian configured to its specific security
    needs, controlling what operations are permitted on the component.
    The guardian works in conjunction with policies to provide
    comprehensive security enforcement.

    :param disallow_post_init: Whether to prevent changes after
        initialisation, defaults to `True`.
    :param private_access: Whether to allow access to private
        attributes, defaults to `True`.
    :param strict: Whether to enforce strict security policies, defaults
        to `False`.

    .. warning::

        Guardian security checks are performed at runtime and may impact
        performance in high-frequency operations. Consider the trade-off
        between security and performance for your use case.

    .. code-block:: python

        guardian = Guardian(
            disallow_post_init=True,
            private_access=False,
            strict=True,
        )
        allowed, error = guardian(component, "_internal")
        if not allowed:
            print(f"Access denied: {error}")
    """

    __slots__: tuple[str] = ("_disallow_post_init", "_private", "_strict")

    def __init__(
        self,
        disallow_post_init: bool = True,
        private_access: bool = True,
        strict: bool = False,
    ) -> None:
        """Initialise a guardian instance with security to protect."""
        self._disallow_post_init = disallow_post_init
        self._private = private_access
        self._strict = strict

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield the guardian's attributes for introspection."""
        yield "protection_active", self._disallow_post_init
        yield "private_access", self._private
        if self._strict:
            yield "strict", self._strict

    def __eq__(self, other: object) -> bool:
        """Check equality with another `Guardian` instance."""
        if not isinstance(other, Guardian):
            return NotImplemented
        return (
            self._disallow_post_init == other._disallow_post_init
            and self._private == other._private
            and self._strict == other._strict
        )

    def __hash__(self) -> int:
        """Return hash value for the `Guardian` instance."""
        return hash((self._disallow_post_init, self._private, self._strict))

    def __call__(self, obj: t.Any, attribute: str) -> tuple[bool, str | None]:
        """Check if an operation is permitted under current security
        policy.

        This method evaluates whether an operation on an attribute
        should be permitted based on the guardian's configuration and the
        component's initialisation state.

        :param obj: The object being accessed.
        :param attribute: The attribute being accessed.
        :return: Tuple of status and error message. If allowed, the
            error message is `None`.

        .. note::

            The guardian checks for component initialisation state
            through the presence of an `_initialised` attribute.
            Components that do not have this attribute are considered
            uninitialised.
        """
        if not hasattr(obj, "_initialised"):
            return True, None
        if not self._disallow_post_init:
            return True, None
        if attribute.startswith("_") and self._private:
            return True, None
        if self._strict:
            return False, (
                "Strict guardian policy prevents modification of "
                f"{attribute!r} after component initialisation"
            )
        if not attribute.startswith("_"):
            return True, None
        return False, (
            "Guardian policy prevents modification of private "
            f"attribute {attribute!r}"
        )

    @property
    def disallow_post_init(self) -> bool:
        """Check if post-initialisation modifications are prevented."""
        return self._disallow_post_init

    @property
    def private_access(self) -> bool:
        """Check if private attribute access is allowed."""
        return self._private

    @property
    def strict(self) -> bool:
        """Check if strict mode is enabled."""
        return self._strict


class ComponentMeta(type):
    """Metaclass that adds component lifecycle management.

    This metaclass enhances the component class' capability with some
    additional capabilities such as attribute protection, lifecycle
    management, and metadata tracking. It works with `Guardian` class
    to enforce security constraints on component attributes.

    This metaclass automatically creates protective mechanisms for
    component classes without requiring explicit code in each component.
    It provides the foundation for the component security and lifecycle
    management infrastructure.

    :param name: Name of the class being created.
    :param bases: Tuple of base classes.
    :param namespace: Dictionary of class attributes.
    :param kwargs: Additional class creation parameters.
    :return: The newly created component class.

    .. warning::

        Users should not interact with this metaclass directly unless
        absolutely necessary.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> type:
        """Create a new component class with enhanced capabilities."""
        if "__slots__" not in namespace and not any(
            hasattr(base, "__slots__") for base in bases
        ):
            namespace["__slots__"] = ()
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if (
            not hasattr(cls, "__setattr__")
            or cls.__setattr__ is object.__setattr__
        ):
            cls.__setattr__ = mcs._protected_setattr
        return cls

    @staticmethod
    def _protected_setattr(obj: t.Any, name: str, value: t.Any) -> None:
        """Protected attribute setter that respects guardian policies.

        This method is automatically assigned to component classes to
        enforce guardian-based security policies on attribute access.

        :param obj: The object whose attribute is being set.
        :param name: The name of the attribute being set.
        :param value: The value being assigned.
        :raises AttributeError: If the guardian denies the operation.
        """
        guardian = getattr(obj, "_guardian", None)
        if guardian is not None:
            allowed, error = guardian.check(obj, name)
            if not allowed:
                raise AttributeError(error)
        object.__setattr__(obj, name, value)


class Component(Observable, metaclass=ComponentMeta):
    """Base class for all mananged components in the system.

    This class provides a comprehensive foundation for building robust,
    observable, and secure system elements. It integrates validation,
    inspection, security policies, and audit logging into a cohesive
    whole.

    Components have well-defined lifecycles with initialisation,
    activation, deactivation, and cleanup phases. They maintain their
    own metrics, audit logs, and security boundaries through guardian
    protection.

    :param name: Human-readable component name, defaults to `None`.
    :param type_: Component type identifier, defaults to `None`.
    :param config: Component configuration dictionary, defaults to `None`.
    :param validators: List of validators for this component, defaults
        to `None`.
    :param inspectors: List of inspectors for this component, defaults
        to `None`.
    :param policies: List of security policies for this component,
        defaults to `None`.
    :param guardian: Security guardian for this component, defaults
        to `None`.

    .. see-also::

        :class:`Guardian`, :class:`Validator`, :class:`Inspector`, and
        :class:`Policy` for the security and monitoring components.

    .. code-block:: python

        class DatabaseComponent(Component):
            def __init__(self, host: str):
                super().__init__(
                    name="Database",
                    type_="storage",
                    config={"connection": host},
                )

        async with DatabaseComponent("sqlite:///app.db") as db:
            await db.validate()
    """

    __slots__: tuple[str] = (
        "_id",
        "_name",
        "_type",
        "_config",
        "_guardian",
        "_metrics",
        "_audit",
        "_validators",
        "_inspectors",
        "_policies",
        "_initialised",
    )

    def __init__(
        self,
        *,
        name: str | None = None,
        type_: str | None = None,
        config: _StateInfoDict | None = None,
        validators: list[Validator] | None = None,
        inspectors: list[Inspector] | None = None,
        policies: list[Policy] | None = None,
        guardian: Guardian | None = None,
    ) -> None:
        """Initialise a component instance."""
        self._id = str(uuid4())[:8]
        self._name = name or type(self).__name__
        self._type = type_ or "component"
        self._config = config or {}
        self._guardian = guardian or Guardian()
        self._metrics = MetricCollector()
        self._audit = AuditLog()
        self._validators = list(validators or [])
        self._inspectors = list(inspectors or [])
        self._policies = list(policies or [])
        object.__setattr__(self, "_initialised", True)

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield the component's attributes for introspection."""
        yield "id", self._id
        yield "name", self._name
        yield "type", self._type
        if self._config:
            if hasattr(self._config, "keys"):
                yield "config_keys", list(self._config.keys())
            elif hasattr(self._config, "__dict__"):
                yield "config_keys", list(self._config.__dict__.keys())
            else:
                yield "config_type", type(self._config).__name__

    def __iter__(self) -> Iterator[tuple[str, t.Any]]:
        """Iterate over component's key properties."""
        yield "id", self._id
        yield "name", self._name
        yield "type", self._type
        yield "config", self._config

    def __setattr__(self, name: str, value: t.Any) -> None:
        """Set component attribute with guardian protection.

        This method overrides the default attribute setting behaviour to
        enforce guardian-based security policies on attribute
        modifications.

        :param name: The attribute name to set.
        :param value: The value to assign.
        :raises AttributeError: If the guardian denies the operation.
        """
        if not hasattr(self, "_initialised"):
            object.__setattr__(self, name, value)
            return
        guardian = getattr(self, "_guardian", None)
        if guardian is not None:
            allowed, error = guardian(self, name)
            if not allowed:
                raise AttributeError(error)
        object.__setattr__(self, name, value)

    async def validate(
        self,
        *,
        context: _ContextInfoDict | None = None,
    ) -> bool:
        """Run all validators on the components.

        This method executes all validators registered with this
        component to ensure it meets all required constraints and
        invariants. The validation process is timed and recorded in the
        component's metrics.

        :param context: Optional context data for validation, defaults
            to `None`.
        :return: `True` if all validations pass, `False` otherwise.

        .. note::

            Validation stops at the first failure, with all subsequent
            validators skipped. If strict validation is required,
            consider using a specific validator that enforces this
            policy.
        """
        started = time.time()
        passed = True
        try:
            for validator in self._validators:
                successful = await validator(self, context=context)
                if not successful:
                    passed = False
                    self._audit.record_event(
                        "validation_failed",
                        component=self._name,
                        validator=validator.name,
                    )
            return passed
        finally:
            took = time.time() - started
            self._metrics.record_operation(took, success=passed)

    async def notify_inspectors(
        self,
        event: str,
        data: dict[str, t.Any] | None = None,
    ) -> None:
        """Notify all active inspectors about an event.

        This method sends event notifications to all active inspectors
        registered with this component. Errors in individual inspectors
        are caught and logged without affecting other inspectors.

        :param event: The event name or type.
        :param data: Additional data associated with the event, defaults
            to `None`.

        .. note::

            Errors in individual inspectors are caught and logged, but
            do not prevent other inspectors from being notified.
        """
        _data = data or {}
        for inspector in self._inspectors:
            try:
                await inspector(event, self, _data)
            except Exception as __error__:
                self._audit.record_event(
                    "inspector_error",
                    component=self._name,
                    inspector=inspector.name,
                    error=str(__error__),
                )

    async def check_policies(
        self,
        operation: str | None = None,
        *,
        context: _ContextInfoDict | None = None,
    ) -> bool:
        """Check if an operation is permittted by all policies.

        This method evaluates the operations against all policies
        registered with this component. Policy evaluation stops at the
        first denial.

        :param operation: The operation to check, defaults to `None`.
        :param context: Optional context data for policy evaluation,
            defaults to `None`.
        :return: `True` if the operation is permitted by all applicable
            policies, `False` otherwise.

        .. note::

            Policy evaluation stops at the first denial, with all
            subsequent policies skipped. This ensures that a single
            restrictive policy can effectively block operations.
        """
        for policy in self._policies:
            allowed = await policy(self, operation, context=context)
            if not allowed:
                self._audit.record_event(
                    "policy_violation",
                    component=self._name,
                    policy=policy.name,
                    operation=operation,
                )
                return False
        return True

    async def initialise(self) -> None:
        """Initialise and prepare the component for use.

        This method performs any necessary setup and initialisation for
        the component. Subclasses should override this method to
        implement specific initialisation logic while calling the parent
        implementation.
        """
        await self.notify_inspectors("component_initialised")

    async def activate(self) -> None:
        """Activate the component for operation.

        This method brings the component into an active state where it
        can perform its intended functions. Subclasses should override
        this method to implement specific activation logic.
        """
        await self.notify_inspectors("component_activated")

    async def deactivate(self) -> None:
        """Deactivate the component gracefully.

        This method transitions the component from an active state to an
        inactive state. Subclasses should override this method to
        implement specific deactivation logic.
        """
        await self.notify_inspectors("component_deactivated")

    async def cleanup(self) -> None:
        """Perform final cleanup of resources.

        This method releases any resources held by the component and
        performs final cleanup operations. Subclasses should override
        this method to implement specific cleanup logic.
        """
        await self.notify_inspectors("component_cleaned_up")

    async def __aenter__(self) -> "Component":
        """Async context manager entry.

        Automatically initialises and activates the component when
        entering an async context manager block.

        :return: The component instance ready for use.
        """
        await self.initialise()
        await self.activate()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit.

        Automatically deactivates and cleans up the component when
        exiting an async context manager block, even if exceptions
        occur.

        :param exc_type: Exception type if an exception occurred,
            defaults to `None`.
        :param exc_val: Exception value if an exception occurred,
            defaults to `None`.
        :param exc_tb: Exception traceback if an exception occurred,
            defaults to `None`.
        """
        try:
            await self.deactivate()
        finally:
            await self.cleanup()

    def add_validator(self, validator: Validator) -> None:
        """Add a validator to the component.

        This method registers a new validator with the component.
        Validators ensure that the component meets required constraints
        and invariants.

        :param validator: The validator to add.
        :raises TypeError: If the provided object is not a Validator.
        """
        if not isinstance(validator, Validator):
            raise TypeError("Expected Validator instance")
        if validator not in self._validators:
            self._validators.append(validator)
            self._audit.record_event(
                "validator_added",
                component=self._name,
                validator=validator.name,
            )

    def add_inspector(self, inspector: Inspector) -> None:
        """Add an inspector to the component.

        This method registers a new inspector with the component.
        Inspectors monitor events and behaviours within the component.

        :param inspector: The inspector to add.
        :raises TypeError: If the provided object is not an Inspector.
        """
        if not isinstance(inspector, Inspector):
            raise TypeError("Expected Inspector instance")
        if inspector not in self._inspectors:
            self._inspectors.append(inspector)
            self._audit.record_event(
                "inspector_added",
                component=self._name,
                inspector=inspector.name,
            )

    def add_policy(self, policy: Policy) -> None:
        """Add a security policy to the component.

        This method registers a new security policy with the component.
        Policies control what operations are permitted on the component.

        :param policy: The policy to add.
        :raises TypeError: If the provided object is not a Policy.
        """
        if not isinstance(policy, Policy):
            raise TypeError("Expected Policy instance")
        if policy not in self._policies:
            self._policies.append(policy)
            self._audit.record_event(
                "policy_added",
                component=self._name,
                policy=policy.name,
            )

    @asynccontextmanager
    async def lifecycle(self):
        """Manage the complete component lifecycle as a context manager.

        This context manager automatically handles the component
        lifecycle, ensuring proper initialisation, activation,
        deactivation, and cleanup even if exceptions occur during usage.

        :yield: The component instance ready for use.

        .. note::

            This context manager ensures proper cleanup even if
            exceptions occur during the component's operation, making it
            the recommended way to use components in most scenarios.
        """
        try:
            await self.initialise()
            await self.activate()
            yield self
        finally:
            try:
                await self.deactivate()
            finally:
                await self.cleanup()

    @property
    def id(self) -> str:
        """Get the unique identifier of the component."""
        return self._id

    @property
    def name(self) -> str:
        """Get the human-readable name of the component."""
        return self._name

    @property
    def type(self) -> str:
        """Get the type identifier of the component"""
        return self._type

    @property
    def config(self) -> _StateInfoDict:
        """Get a copy of the component's configuration."""
        return self._config.copy()

    @property
    def metrics(self) -> MetricCollector:
        """Get the metrics for the component."""
        return self._metrics

    @property
    def audit(self) -> AuditLog:
        """Get the audit log for the component."""
        return self._audit

    @property
    def guardian(self) -> Guardian:
        """Get the security guardian for the component."""
        return self._guardian


class ExecutionContext(Observable):
    """Store contextual information for operations and requests.

    This class provides a container for storing and managing contextual
    information related to operations, requests, and other activities
    within the system. It maintains metadata such as timestamps, user
    and session identifiers, and tracing information.

    Contexts are inherently immutable once created, but new derived
    contexts can be created with updated information while maintaining
    references to the original context's data and hierarchy.

    :param name: Human-readable context name.
    :param type_: Context type identifier.
    :param user_id: Identifier of the user associated with this context,
        defaults to `None`.
    :param session_id: Identifier of the session associated with this
        context, defaults to `None`.
    :param trace_id: Distributed tracing identifier, defaults to `None`.
    :param parent_id: Identifier of the parent context, defaults
        to `None`.
    :param metadata: Additional metadata for this context, defaults
        to `None`.

    .. note::

        `ExecutionContext` supports dictionary-like access to its
        properties and metadata, making it convenient to use in various
        scenarios.

    .. see-also::

        :meth:`extend` and :meth:`fork` for creating derived contexts.

    .. code-block:: python

        context = ExecutionContext(
            name="user_request",
            type_="http_request",
            user_id="xames3",
            session_id="session-2345623746725",
        )
        sub_context = context.extend(
            operation="database_query", query="SELECT * FROM users"
        )
        parallel_context = context.fork(
            name="background_task",
            type_="async_task",
            task_id="task-7892374563274",
        )
    """

    __slots__: tuple[str] = (
        "_name",
        "_type",
        "_id",
        "_timestamp",
        "_user_id",
        "_session_id",
        "_trace_id",
        "_parent_id",
        "_metadata",
    )

    def __init__(
        self,
        name: str,
        type_: str,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_id: str | None = None,
        metadata: _StateInfoDict | None = None,
    ) -> None:
        """Initialise an execution context instance."""
        self._name = name
        self._type = type_
        self._user_id = user_id
        self._session_id = session_id
        self._trace_id = trace_id
        self._parent_id = parent_id
        self._metadata = metadata or {}
        self._id = str(uuid4())[:8]
        self._timestamp = time.time()

    def __inspect_attrs__(self) -> _AttributeStream:
        """Yield the context's attributes for introspection."""
        yield "name", self._name
        yield "type", self._type
        yield "id", self._id
        if self._user_id:
            yield "user", self._user_id
        if self._session_id:
            yield "session", (
                self._session_id[:8] + "..."
                if len(self._session_id) > 8
                else self._session_id
            )

    def __getitem__(self, key: str) -> t.Any:
        """Get a context property or metadata value by key.

        This method provides dictionary-like access to context
        properties and metadata. Standard properties are returned first,
        followed by custom metadata.

        :param key: The property or metadata key to retrieve.
        :return: The property or metadata value.
        """
        if key == "name":
            return self._name
        elif key == "type":
            return self._type
        elif key == "id":
            return self._id
        elif key == "timestamp":
            return self._timestamp
        elif key == "user_id":
            return self._user_id
        elif key == "session_id":
            return self._session_id
        elif key == "trace_id":
            return self._trace_id
        elif key == "parent_id":
            return self._parent_id
        return self._metadata[key]

    def __setitem__(self, key: str, value: t.Any) -> None:
        """Set a metadata value by key."""
        self._metadata[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if a property or metadata key exists."""
        return (
            key
            in {
                "name",
                "type",
                "id",
                "timestamp",
                "user_id",
                "session_id",
                "trace_id",
                "parent_id",
            }
            or key in self._metadata
        )

    def __iter__(self) -> Iterator[tuple[str, t.Any]]:
        """Iterate over context's key properties."""
        yield "name", self._name
        yield "id", self._id
        yield "type", self._type
        yield "timestamp", self._timestamp
        if self._user_id:
            yield "user_id", self._user_id
        if self._session_id:
            yield "session_id", self._session_id
        if self._trace_id:
            yield "trace_id", self._trace_id
        if self._parent_id:
            yield "parent_id", self._parent_id
        yield from self._metadata.items()

    def extend(self, **metadata: t.Any) -> "ExecutionContext":
        """Extend the context with additional data.

        This method creates a new context that inherits properties from
        this context, with specified properties overridden or added. The
        original context remains unchanged.

        :return: A new `ExecutionContext` instance with the combined
            properties.
        """
        return ExecutionContext(
            name=self._name,
            type_=self._type,
            user_id=self._user_id,
            session_id=self._session_id,
            trace_id=self._trace_id,
            parent_id=self._id,
            metadata={**self._metadata, **metadata},
        )

    def fork(
        self,
        name: str,
        type_: str,
        **metadata: t.Any,
    ) -> "ExecutionContext":
        """Fork a new context instance.

        This method creates a new context that inherits properties from
        this context, with specified properties (name and type)
        overridden or added. The original context remains unchanged.

        :return: A new `ExecutionContext` instance with the combined
            properties.
        """
        return ExecutionContext(
            name=name,
            type_=type_,
            user_id=self._user_id,
            session_id=self._session_id,
            trace_id=self._trace_id,
            parent_id=self._id,
            metadata=metadata,
        )

    @property
    def name(self) -> str:
        """Get the human-readable name of the context."""
        return self._name

    @property
    def type(self) -> str:
        """Get the type identifier of the context."""
        return self._type

    @property
    def timestamp(self) -> float:
        """Get the creation timestamp of the context."""
        return self._timestamp

    @property
    def metadata(self) -> _StateInfoDict:
        """Get a copy of the context's metadata."""
        return self._metadata.copy()

    @property
    def id(self) -> str:
        """Get the unique identifier of the context."""
        return self._id

    @property
    def user_id(self) -> str | None:
        """Get the user identifier associated with the context."""
        return self._user_id

    @property
    def session_id(self) -> str | None:
        """Get the session identifier associated with the context."""
        return self._session_id

    @property
    def trace_id(self) -> str | None:
        """Get the distributed tracing identifier for the context."""
        return self._trace_id

    @property
    def parent_id(self) -> str | None:
        """Get the identifier of the parent context."""
        return self._parent_id
