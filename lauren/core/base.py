"""\
Base
====

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, July 22 2025
Last updated on: Saturday, July 26 2025

This module provides the foundational base classes and utilities for
the framework, including the `BaseComponent` class with enterprise
capabilities, validation, security policies, and immutability
enforcement. It also includes the `BaseContext` class for representing
execution contexts, and various utility classes for metrics collection,
audit trails, and immutability guards.

This module is designed to be used as a base for building components
within the framework, providing a consistent interface and behaviour
across different components. It ensures that components can be
validated, observed, and secured according to the framework's
requirements, while also allowing for extensibility and customisation
through the use of validators, observers, and security policies.
"""

from __future__ import annotations

import asyncio
import time
import typing as t
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from uuid import uuid4

from lauren.core.exceptions import ValidationError

__all__: list[str] = [
    "AuditTrail",
    "BaseComponent",
    "BaseComponentMeta",
    "BaseContext",
    "ImmutabilityGuard",
    "MetricsCollector",
    "Observer",
    "SecurityPolicy",
    "Validator",
]


class Validator(ABC):
    """Validator for component validation."""

    def __repr__(self) -> str:
        """Return a string representation of the validator."""
        return f"<{type(self).__name__}()>"

    @abstractmethod
    async def __call__(
        self,
        component: t.Any,
        context: BaseContext | None = None,
    ) -> bool:
        """Validate a component instance.

        This method should implement the validation logic for a
        component, ensuring it meets the required criteria for execution
        within the framework.

        :param component: The component to validate.
        :param context: Optional execution context.
        :return: `True` if validation passes, `False` otherwise.
        """
        raise NotImplementedError


class Observer(ABC):
    """Observer for the component lifecycle events."""

    def __repr__(self) -> str:
        """Return a string representation of the observer."""
        return f"<{type(self).__name__}()>"

    @abstractmethod
    async def __call__(
        self,
        event: str,
        component: t.Any,
        data: dict[str, t.Any],
    ) -> None:
        """Handle a component lifecycle event.

        This method should implement the logic for handling events
        triggered by components, allowing observers to react to
        lifecycle changes.

        :param event: The name of the event.
        :param component: The component that triggered the event.
        :param data: Additional event data.
        """
        raise NotImplementedError


class SecurityPolicy(ABC):
    """Security policy for component operations."""

    def __repr__(self) -> str:
        """Return a string representation of the security policy."""
        return f"<{type(self).__name__}()>"

    @abstractmethod
    async def __call__(
        self,
        component: t.Any,
        operation: str | None = None,
    ) -> bool:
        """Evaluate security policy for a component operation.

        This method should implement the logic for evaluating whether a
        component operation is allowed based on the security policy.

        :param component: The component being evaluated.
        :param operation: Optional operation being performed.
        :return: `True` if operation is allowed, `False` otherwise.
        """
        raise NotImplementedError


class MetricsCollector:
    """Metrics collection with performance tracking."""

    def __init__(self) -> None:
        """Initialise metrics collector."""
        self.data: dict[str, t.Any] = {
            "created_at": time.time(),
            "exec_count": 0,
            "last_exec": None,
            "total_exec_time": 0.0,
            "avg_exec_time": 0.0,
            "min_exec_time": float("inf"),
            "max_exec_time": 0.0,
            "error_count": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "tokens_processed": 0,
            "tokens_per_second": 0.0,
            "memory_usage_mb": 0.0,
        }

    def __repr__(self) -> str:
        """Return a string representation of the metrics collector."""
        return f"<{type(self).__name__}(metrics={self.data})>"

    def record(self, exec_time: float) -> None:
        """Record execution metrics.

        This method updates the internal metrics state with the latest
        execution time, incrementing the execution count and updating
        the total execution time. It also calculates the average
        execution time based on the number of executions recorded.

        :param exec_time: The time taken for the last execution.
        """
        self.data["exec_count"] += 1
        self.data["last_exec"] = time.time()
        self.data["total_exec_time"] += exec_time
        self.data["min_exec_time"] = min(self.data["min_exec_time"], exec_time)
        self.data["max_exec_time"] = max(self.data["max_exec_time"], exec_time)
        count = self.data["exec_count"]
        total = self.data["total_exec_time"]
        self.data["avg_exec_time"] = total / count if count > 0 else 0.0

    def record_success(self) -> None:
        """Record a successful operation."""
        self.data["success_count"] += 1
        self._update_success_rate()

    def record_error(self) -> None:
        """Record a failed operation."""
        self.data["error_count"] += 1
        self._update_success_rate()

    def record_tokens(self, tokens: int, processing_time: float) -> None:
        """Record token processing metrics for LLM operations.

        :param tokens: Number of tokens processed.
        :param processing_time: Time taken to process tokens.
        """
        self.data["tokens_processed"] += tokens
        if processing_time > 0:
            self.data["tokens_per_second"] = tokens / processing_time

    def record_memory_usage(self, memory: float) -> None:
        """Record memory usage metrics.

        :param memory: Memory usage in megabytes.
        """
        self.data["memory_usage_mb"] = memory

    def _update_success_rate(self) -> None:
        """Update success rate calculation."""
        total_ops = self.data["success_count"] + self.data["error_count"]
        if total_ops > 0:
            self.data["success_rate"] = self.data["success_count"] / total_ops

    def show(self) -> dict[str, t.Any]:
        """Return current metrics snapshot."""
        return self.data.copy()


class AuditTrail:
    """Audit trail for compliance and debugging."""

    def __init__(self) -> None:
        """Initialise an empty audit trail."""
        self.data: list[dict[str, t.Any]] = []

    def __repr__(self) -> str:
        """Return a string representation of the audit trail."""
        return f"<{type(self).__name__}(audit={self.data})>"

    def record(self, event: str, component: str, **data: t.Any) -> None:
        """Record an audit event.

        This method adds a new event to the audit trail, capturing the
        event type, timestamp, component involved, and any additional
        data provided. This allows for comprehensive tracking of
        component lifecycle events and operations.

        :param event: The type of the event (e.g., `plugin_loaded`).
        :param component: Component (name) involved in the event.
        """
        entry = {
            "type": event,
            "timestamp": time.time(),
            "component": component,
            **data,
        }
        self.data.append(entry)

    def show(self) -> list[dict[str, t.Any]]:
        """Return audit trail entries."""
        return self.data.copy()


class ImmutabilityGuard:
    """Immutability enforcement with configurable policies."""

    def __init__(
        self,
        enforce_post_init: bool = True,
        allow_private: bool = True,
        strict: bool = False,
    ) -> None:
        """Initialise immutability guard with policy."""
        self.enforce_post_init = enforce_post_init
        self.allow_private = allow_private
        self.strict = strict

    def check(
        self,
        obj: t.Any,
        name: str,
    ) -> tuple[bool, str | None]:
        """Check if attribute modification is allowed.

        This method checks whether an attribute modification is allowed
        based on the immutability policies configured. It returns a tuple
        indicating whether the modification is allowed and an error
        message if it is not.

        :param obj: The object being modified.
        :param name: The attribute name.
        :return: Tuple of (boolean of allowed or not, error_message).
        """
        if not hasattr(obj, "_initialised"):
            return True, None
        if not self.enforce_post_init:
            return True, None
        if name.startswith("_") and self.allow_private:
            return True, None
        if not name.startswith("_") and not self.strict:
            return True, None
        if self.strict and not name.startswith("_"):
            return False, (
                f"Cannot modify attribute {name!r} directly. "
                "Component instances are immutable after initialisation"
            )
        return False, (
            f"Cannot modify private attribute {name!r} directly. "
            "Use appropriate methods"
        )


@dataclass(frozen=True)
class BaseContext:
    """Immutable execution context for the framework components.

    This context follows the value object pattern, providing thread-safe
    state representation with comprehensive metadata support. This
    allows components to operate with a consistent view of their
    execution environment without risking state mutation during
    processing.

    :var name: Context name, typically the component name.
    :var type: Context type, usually the component class name.
    :var metadata: Additional metadata, defaults to empty dict.
    :var timestamp: Context creation time, defaults to current time.
    :var ctx_id: Unique identifier for tracing, defaults to a new UUID.
    :var user_id: Optional user identifier for multi-tenant operations.
    :var session_id: Optional session identifier for conversation tracking.
    :var trace_id: Optional trace identifier for distributed tracing.
    :var parent_id: Optional parent context for nested operations.
    :raises ValueError: If name or type is not a non-empty string.
    :raises TypeError: If metadata is not a dict.
    """

    name: str
    type: str
    metadata: dict[str, t.Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    ctx_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str | None = field(default=None)
    session_id: str | None = field(default=None)
    trace_id: str | None = field(default=None)
    parent_id: str | None = field(default=None)

    def __post_init__(self) -> None:
        """Validate context after initialisation.

        This method ensures that the context is properly configured with
        valid name and type, and that metadata is a dictionary. This
        guarantees that the context can be used safely across different
        components without risking type errors or state inconsistencies.
        """
        checks = {self.name, self.type}
        for check in checks:
            if not check or not isinstance(check, str):
                raise ValueError(f"{check} must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")

    def extend(self, **metadata: t.Any) -> BaseContext:
        """Create new context with additional metadata.

        This method allows creating a new context instance with the same
        name and type, but with additional metadata. This is useful for
        passing context-specific information without modifying the
        original context, thus preserving immutability and thread
        safety.

        :return: A new `BaseContext` instance with the updated metadata.
        """
        return replace(self, metadata={**self.metadata, **metadata})

    def fork(
        self,
        name: str,
        type_: str,
        **metadata: t.Any,
    ) -> BaseContext:
        """Fork a child context for nested operations.

        This method creates a new context that inherits `user_id`,
        `session_id`, and `trace_id` from the parent context, making it
        suitable for nested operations within the same execution flow.

        :param name: Name for the child context.
        :param type_: Type for the child context.
        :param metadata: Additional metadata for the child context.
        :return: A new child BaseContext instance.
        """
        return BaseContext(
            name=name,
            type=type_,
            metadata={**self.metadata, **metadata},
            user_id=self.user_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            parent_id=self.ctx_id,
        )


class BaseComponentMeta(ABCMeta):
    """Elegant metaclass for framework components."""

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> type:
        """Create component class with contract validation.

        This method creates a new component class, validating that it
        adheres to the required contract by checking for properties,
        methods, and validators.

        :param name: Name of the class being created.
        :param bases: Base classes for the new class.
        :param namespace: Namespace dictionary containing class
            attributes.
        :return: The newly created component class.
        """
        properties = kwargs.pop("properties", set())
        methods = kwargs.pop("methods", set())
        validators = kwargs.pop("validators", [])
        class_ = super().__new__(cls, name, bases, namespace)
        if name == "BaseComponent":
            return class_
        if any(
            issubclass(base, BaseComponent) for base in bases if base != class_
        ):
            cls._validate_contract(class_, properties, methods)
            cls._register_validators(class_, validators)
        return class_

    @staticmethod
    def _validate_contract(
        cls: type,
        properties: set[str],
        methods: set[str],
    ) -> None:
        """Validate component contract implementation.

        This method checks that the component class implements the
        required properties and methods as defined in the contract. This
        ensures that the component adheres to the expected interface
        and can be used safely within the framework.
        """
        for property_ in properties:
            if hasattr(cls, property_):
                if not isinstance(getattr(cls, property_), property):
                    raise ValidationError(
                        f"{property_!r} must be a property",
                        attribute=property_,
                    )
        for method in methods:
            if hasattr(cls, method):
                if not callable(getattr(cls, method)):
                    raise ValidationError(
                        f"{method!r} must be a callable",
                        attribute=method,
                    )

    @staticmethod
    def _register_validators(cls: type, validators: list[Validator]) -> None:
        """Register validators with the component class.

        This method adds the provided validators to the component class,
        allowing it to be validated against the specified criteria during
        execution. This is essential for ensuring that components can be
        validated against custom rules defined by the framework or
        users.

        :param cls: The component class to register validators with.
        :param validators: List of validators to register.
        """
        if not hasattr(cls, "_validators"):
            cls._validators = []
        cls._validators.extend(validators)


class BaseComponent(metaclass=BaseComponentMeta):
    """Foundational component with enterprise capabilities."""

    def __init__(
        self,
        validators: list[Validator] | None = None,
        observers: list[Observer] | None = None,
        security_policies: list[SecurityPolicy] | None = None,
        immutability_guard: ImmutabilityGuard | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialise component with injected dependencies."""
        self._context: BaseContext | None = None
        self._validators = validators or []
        self._observers = observers or []
        self._security_policies = security_policies or []
        self._immutability_guard = immutability_guard or ImmutabilityGuard()
        self._metrics_collector = MetricsCollector()
        self._audit_trail = AuditTrail()
        self._audit_trail.record(
            "component_created",
            type(self).__name__,
            timestamp=time.time(),
        )
        self._initialised = True
        # NOTE(xames3): Defer notifying observers until after
        # initialisation is complete. This avoids triggering observer
        # logic while the component is still being set up, which could
        # lead to inconsistent state or errors if attributes are not
        # yet ready.
        if self._is_running_asynchronously():
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._notify_observers(
                        "created", {"timestamp": time.time()}
                    )
                )
            except RuntimeError:
                # If no event loop is running, we cannot notify
                # observers.
                pass

    def __repr__(self) -> str:
        """Provide informative string representation."""
        cls = type(self).__name__
        if hasattr(self, "name"):
            name = self._get_name()
            return f"<{cls}(name={name!r})>"
        return f"<{cls}()>"

    def __setattr__(self, name: str, value: t.Any) -> None:
        """Enforce immutability through injected guard.

        This method overrides the default attribute setting to ensure
        that modifications to attributes are checked against the
        immutability guard. If the guard allows the modification, it
        proceeds; otherwise, it raises an error with a descriptive
        message.

        :param name: The name of the attribute to set.
        :param value: The value to set the attribute to.
        :raises AttributeError: If the modification is not allowed by
            the immutability guard.
        :raises TypeError: If the immutability guard is not set up
            correctly.
        :raises ValueError: If the name is not a valid attribute name.
        """
        if not hasattr(self, "_immutability_guard"):
            super().__setattr__(name, value)
            return
        allowed, error = self._immutability_guard.check(self, name)
        if allowed:
            super().__setattr__(name, value)
        else:
            raise AttributeError(error)

    @property
    def context(self) -> BaseContext | None:
        """Access execution context."""
        return self._context

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Access performance metrics."""
        return self._metrics_collector.show()

    @property
    def audit_trail(self) -> list[dict[str, t.Any]]:
        """Access audit trail."""
        return self._audit_trail.show()

    def add_validator(self, validator: Validator) -> None:
        """Add a validator.

        This method allows adding a new validator to the component,
        enabling custom validation logic to be applied during the
        component's lifecycle. This is useful for extending the
        validation capabilities of the component without modifying its
        core logic.

        :param validator: The validator to add.
        :raises TypeError: If the validator does not implement the
            `Validator` interface.
        """
        if not isinstance(validator, Validator):
            raise TypeError("validator must implement Validator interface")
        self._validators.append(validator)

    def add_observer(self, observer: Observer) -> None:
        """Add an observer.

        This method allows adding an observer to the component, enabling
        it to react to lifecycle events. Observers can be used to
        implement custom logic that should be executed when certain
        events occur, such as state changes or lifecycle transitions.

        :param observer: The observer to add.
        :raises TypeError: If the observer does not implement the
            `Observer` interface.
        """
        if not isinstance(observer, Observer):
            raise TypeError("observer must implement Observer interface")
        self._observers.append(observer)

    def add_security_policy(self, policy: SecurityPolicy) -> None:
        """Add security policy.

        This method allows adding a security policy to the component,
        enabling it to enforce security rules during operations.
        Security policies can be used to implement custom security
        checks that must be satisfied before certain operations can be
        performed.

        :param policy: The security policy to add.
        :raises TypeError: If the policy does not implement the
            `SecurityPolicy` interface.
        """
        if not isinstance(policy, SecurityPolicy):
            raise TypeError("policy must implement SecurityPolicy interface")
        self._security_policies.append(policy)

    async def validate(self, context: BaseContext | None = None) -> bool:
        """Validate component using all registered validators.

        This method runs all registered validators against the
        component, ensuring that it meets the required criteria for
        execution within the framework.

        :param context: Optional execution context for validation.
        :return: `True` if validation passes, `False` otherwise.

        .. note::

            Validation failures are logged to the audit trail for
            debugging purposes.
        """
        for validator in self._validators:
            try:
                if not await validator(self, context):
                    self._audit_trail.record(
                        "validation_failed",
                        type(self).__name__,
                        validator=type(validator).__name__,
                    )
                    return False
            except Exception as error:
                self._audit_trail.record(
                    "validation_error",
                    type(self).__name__,
                    validator=type(validator).__name__,
                    error=str(error),
                )
                return False
        return True

    async def evaluate(self, operation: str | None = None) -> bool:
        """Evaluate security using all registered policies.

        This method checks whether the component's operations are
        allowed based on the registered security policies.

        :param operation: Optional operation being performed.
        :return: `True` if operation is allowed, `False` otherwise.

        .. note::

            Security policy failures are logged to the audit trail for
            compliance and debugging purposes.
        """
        for policy in self._security_policies:
            try:
                if not await policy(self, operation):
                    self._audit_trail.record(
                        "security_denied",
                        type(self).__name__,
                        policy=type(policy).__name__,
                        operation=operation,
                    )
                    return False
            except Exception as error:
                self._audit_trail.record(
                    "security_error",
                    type(self).__name__,
                    policy=type(policy).__name__,
                    operation=operation,
                    error=str(error),
                )
                return False
        return True

    async def _notify_observers(
        self, event: str, data: dict[str, t.Any]
    ) -> None:
        """Notify all observers of an event.

        This method triggers all registered observers, passing the event
        name and any additional data. Observers can implement custom
        logic that should be executed when the event occurs, allowing
        for flexible and extensible event handling.

        :param event: The name of the event to notify observers about.
        :param data: Additional data to pass to observers.

        .. note::

            Observer failures are logged but do not prevent other
            observers from executing.
        """
        if not self._observers:
            return
        tasks: list[asyncio.Task] = []
        for observer in self._observers:
            try:
                task = asyncio.create_task(observer(event, self, data))
                tasks.append(task)
            except Exception as error:
                self._audit_trail.record(
                    "observer_error",
                    type(self).__name__,
                    observer=type(observer).__name__,
                    event=event,
                    error=str(error),
                )
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for index, result in enumerate(results):
                if isinstance(result, Exception):
                    observer = self._observers[index]
                    self._audit_trail.record(
                        "observer_failed",
                        type(self).__name__,
                        observer=type(observer).__name__,
                        event=event,
                        error=str(result),
                    )

    def _create_context(self, **metadata: t.Any) -> BaseContext:
        """Create execution context for operations.

        This method creates a new context instance with the component's
        name and type, along with any additional metadata provided. This
        allows components to operate with a consistent view of their
        execution environment without risking state mutation during
        processing.

        :param metadata: Additional metadata to include in the context.
        :return: A new `BaseContext` instance with the updated metadata.
        """
        return BaseContext(self._get_name(), type(self).__name__, metadata)

    def _record_audit_trail(self, event: str, **data: t.Any) -> None:
        """Record audit event.

        This method adds a new event to the audit trail, capturing the
        event type, timestamp, component involved, and any additional
        data provided. This allows for comprehensive tracking of
        component lifecycle events and operations.

        :param event: The type of the event (e.g., `plugin_loaded`).
        :param data: Additional event data.
        """
        self._audit_trail.record(event, type(self).__name__, **data)

    def _record_execution_metrics(self, execution_time: float) -> None:
        """Record execution metrics with success tracking."""
        self._metrics_collector.record(execution_time)
        self._metrics_collector.record_success()

    def _is_running_asynchronously(self) -> bool:
        """Check if the application currently in an async context."""
        try:
            return asyncio.get_running_loop() is not None
        except RuntimeError:
            return False

    def _get_name(self) -> str:
        """Get component name."""
        if hasattr(self, "name"):
            name = getattr(self, "name")
            result = name() if callable(name) else str(name)
            return result if result and result.strip() else type(self).__name__
        return type(self).__name__
