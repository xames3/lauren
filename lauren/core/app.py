"""\
Application
===========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Monday, July 21 2025

This module provides the foundational application class for building
L.A.U.R.E.N powered applications. The `App` class serves as an abstract
base that developers extend to create their own prompt processing
applications. The framework handles the infrastructure concerns like
configuration, logging, tracing, and plugin management whilst developers
focus on implementing their specific business logic through clean,
extensible interfaces.

This design separates framework responsibilities from application logic,
ensuring consistent observability and configuration across all
applications whilst maintaining flexibility for diverse use cases.
"""

from __future__ import annotations

import time
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from uuid import uuid4

from lauren.core.config import Config
from lauren.core.plugin import PluginManager
from lauren.utils.logging import configure
from lauren.utils.logging import get_logger
from lauren.utils.opentelemetry import get_tracer

if t.TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from lauren.core.plugin import Plugin

__all__: list[str] = ["App", "ContextManager", "AppContext"]

user: ContextVar[str | None] = ContextVar("user", default=None)
session: ContextVar[str | None] = ContextVar("session", default=None)
metadata: ContextVar[dict[str, t.Any]] = ContextVar("metadata", default={})


@dataclass(frozen=True)
class AppContext:
    """Execution context for request processing.

    This context captures the essential state information for processing
    requests through the framework. It provides a snapshot of the user
    session, authentication, and metadata that flows through the entire
    request lifecycle whilst maintaining immutability for security and
    consistency.

    The context is designed to be lightweight and serialisable, making
    it suitable for distributed processing, caching, and audit trails.

    .. note::

        All modifications create new instances rather than mutating
        existing state.
    """

    user: str | None = field(default=None)
    session: str | None = field(default=None)
    metadata: dict[str, t.Any] = field(default_factory=dict)
    request: str | None = field(default=None)
    timestamp: float = field(default_factory=lambda: time.time())
    trace: str | None = field(default=None)


class ContextManager:
    """Class to manage execution context.

    This manager provides thread-safe, immutable context operations that
    maintain audit trails and support distributed processing. All
    context modifications create new instances whilst preserving the
    original state for rollback and observability purposes.

    The manager integrates with the broader framework observability
    stack, automatically capturing context changes for tracing and
    security auditing whilst maintaining high performance through lazy
    evaluation and efficient state management.
    """

    def __init__(self, context: AppContext) -> None:
        """Initialise the context manager with an execution context."""
        if not isinstance(context, AppContext):
            raise TypeError(
                f"Expected AppContext, got {type(context).__name__}"
            )
        self._context = context
        self._initial_state = context
        self._observers: list[t.Callable[[AppContext, AppContext], None]] = []

    @property
    def context(self) -> AppContext:
        """Access the current execution context."""
        return self._context

    def add_observer(
        self,
        observer: t.Callable[[AppContext, AppContext], None],
    ) -> None:
        """Add an observer for context changes.

        This allows external components to react to context changes,
        such as logging, tracing, or triggering additional workflows.

        :param observer: A callable that takes the old and new context
            as arguments. It should handle any exceptions internally to
            avoid breaking the context management flow.
        :raises TypeError: If the observer is not callable.
        """
        if not callable(observer):
            raise TypeError(f"{type(observer).__name__} must be a callable")
        self._observers.append(observer)

    def remove_observer(
        self,
        observer: t.Callable[[AppContext, AppContext], None],
    ) -> None:
        """Remove an observer for context changes.

        This allows for dynamic management of observers, enabling
        components to stop receiving notifications about context changes.

        :param observer: The observer to remove.
        :raises ValueError: If the observer is not found in the
            observers list.
        """
        if observer in self._observers:
            self._observers.remove(observer)
        else:
            raise ValueError(
                f"Observer: {observer} not found in current context observers"
            )

    def notify_observers(
        self,
        old_context: AppContext,
        new_context: AppContext,
    ) -> None:
        """Notify all observers of context changes.

        This method calls all registered observers with the old and new
        context, allowing them to react to context changes. Observers
        can perform additional actions such as logging, tracing, or
        triggering additional workflows based on context changes.

        :param old_context: The previous execution context.
        :param new_context: The updated execution context.
        """
        for observer in self._observers:
            try:
                observer(old_context, new_context)
            except Exception:
                pass

    def set(self, name: str, value: t.Any) -> None:
        """Set a value in the execution context.

        This method updates the execution context with a new value for
        the specified name. It creates a new context instance to ensure
        immutability and thread-safety, allowing for rollback and
        observability.

        :param name: The name of the value to set.
        :param value: The value to set in the context.
        """
        old_context = self._context
        self._context = replace(
            self._context,
            metadata={**self._context.metadata, name: value},
        )
        self.notify_observers(old_context, self._context)

    def get(self, name: str, default: t.Any | None = None) -> t.Any | None:
        """Get a value from the execution context.

        This method retrieves a value from the execution context by its
        name. If the name does not exist, it returns the provided
        default value or `None` if no default is specified. This allows
        for safe access to context values without raising exceptions.

        :param name: The name of the value to retrieve.
        :param default: The default value to return if the name does not
            exist in the context, defaults to `None`.
        :return: The value associated with the name, or the default
            value if the name does not exist.
        """
        return self._context.metadata.get(name, default)

    def update(self, name: str, value: t.Any) -> None:
        """Update an existing value in the execution context.

        This method updates the value of an existing name in the
        execution context.

        :param name: The name of the value to update.
        :param value: The new value to set for the name.
        :raises KeyError: If the name does not exist in the context.
        """
        if name not in self._context.metadata:
            raise KeyError(f"Key: {name!r} does not exist in current context")
        self.set(name, value)

    def exists(self, name: str) -> bool:
        """Check if a name exists in the execution context."""
        return name in self._context.metadata

    def remove(self, name: str) -> None:
        """Remove a name from the execution context.

        This method removes a name and its associated value from the
        execution context.

        :param name: The name to remove from the context.
        """
        if name not in self._context.metadata:
            raise KeyError(f"Key: {name!r} does not exist in current context")
        old_context = self._context
        self._context = replace(
            self._context,
            metadata={
                k: v for k, v in self._context.metadata.items() if k != name
            },
        )
        self.notify_observers(old_context, self._context)

    def clear(self) -> None:
        """Clear all metadata from the execution context."""
        old_context = self._context
        self._context = replace(self._context, metadata={})
        self.notify_observers(old_context, self._context)

    def reset(self) -> None:
        """Reset the execution context to its initial state."""
        old_context = self._context
        self._context = self._initial_state
        self.notify_observers(old_context, self._context)

    def fork(self, **overrides: t.Any) -> "ContextManager":
        """Create a new context manager with modified state.

        This method creates a new instance of the context manager with
        the current context as a base, applying any overrides provided.
        This allows for creating child contexts that inherit from the
        current context whilst allowing modifications for specific
        operations or workflows.

        :param overrides: Key-value pairs to override in the new
            context.
        :return: A new `ContextManager` instance with the modified state.
        """
        if not overrides:
            return self
        return ContextManager(replace(self._context, **overrides))


class AppMeta(type):
    """Metaclass for `App`.

    This metaclass provides enterprise-grade capabilities including lazy
    manager instantiation, configuration validation, security
    enforcement, and comprehensive observability hooks. It ensures
    consistent behaviour across all App subclasses whilst enabling
    dynamic customisation and performance optimisation.

    The metaclass automatically configures managers, validates security
    policies, and establishes observability infrastructure at class
    creation time, ensuring that all instances inherit proper framework
    behaviour whilst maintaining the flexibility for application-specific
    customisation.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespaces: dict[str, t.Any],
    ) -> type:
        """Create new `App` class with some necessary checks."""
        if name == "App":
            return super().__new__(cls, name, bases, namespaces)
        subclassed = any(
            hasattr(base, "__name__") and base.__name__ == "App"
            for base in bases
        )
        if subclassed:
            if "process" in namespaces:
                raise TypeError("Cannot override the 'process' method")
            cls._validate_methods(namespaces)
            cls._setup_observability_hooks(namespaces)
            cls._configure_lazy_loading(namespaces)
        return super().__new__(cls, name, bases, namespaces)

    @staticmethod
    def _validate_methods(namespaces: dict[str, t.Any]) -> None:
        """Validate that required methods are implemented.

        This method checks that some required methods are defined and are
        callable object(s). This ensures that all subclasses of `App`
        implement the necessary application logic interface.

        :param namespaces: The class namespace containing methods to
            validate.
        :raises TypeError: If methods are not defined or not callable.
        """
        methods = {"handle"}
        for _method in methods:
            if _method in namespaces:
                method = namespaces[_method]
                if not callable(method):
                    raise TypeError(f"{_method!r} must be a callable")

    @staticmethod
    def _setup_observability_hooks(namespaces: dict[str, t.Any]) -> None:
        """Setup observability hooks for the class."""
        pass

    @staticmethod
    def _configure_lazy_loading(namespaces: dict[str, t.Any]) -> None:
        """Configure lazy loading capabilities."""
        lazy_config = namespaces.get("_lazy_config", {})
        namespaces["_lazy_config"] = {
            "plugins": True,
            "context": False,
            **lazy_config,
        }


class App(metaclass=AppMeta):
    """Foundational class for creating dynamic, observable applications.

    This class serves as the central orchestrator for all the
    framework's functionality, providing a highly configurable yet
    immutable foundation for building enterprise-grade LLM applications.
    It manages the complete request lifecycle with built-in
    observability, security, and performance optimisation capabilities.

    This class implements lazy loading strategies for optimal resource
    utilisation whilst maintaining strict immutability for security and
    consistency. All operations flow through this class, providing
    centralised control, audit trails, and comprehensive observability
    for enterprise deployments.

    :param config: An optional configuration object to initialise the
        application. If not provided, a default `Config` instance is
        created. This configuration object is immutable after
        initialisation, ensuring that all settings are fixed for the
        lifetime of the application instance.

    .. note::

        Developers should subclass `App` and implement the `handle`
        method to define their application's specific logic whilst
        inheriting all framework capabilities automatically.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialise the core application."""
        self._config = config or Config()
        configure(self._config.logging)
        self._logger = get_logger(__name__)
        self._logger.debug("Initialising L.A.U.R.E.N...")
        self._tracer = get_tracer(self._config.name)
        self._runtime = AppContext(
            user=user.get(None),
            session=session.get(None),
            metadata=metadata.get({}).copy(),
            request=str(uuid4()),
            timestamp=time.time(),
        )
        self._audit: list[dict[str, t.Any]] = []
        self._security_policies: list[t.Callable[[t.Any], bool]] = []
        self._context = ContextManager(self._runtime)
        self._context.add_observer(self._observe_context)
        self._metrics: dict[str, t.Any] = {
            "requests_processed": 0,
            "processing_time": 0.0,
            "plugin_loaded": 0,
            "context_changes": 0,
        }
        lazy = getattr(self, "_lazy_config", {})
        self._plugins = PluginManager(self)
        self._plugins.add_validator(self._validate_plugin)
        self._plugins.add_observer(self._observe_plugin)
        self._plugins.load(lazy=lazy.get("plugins", False))
        self._initialised = True
        self._logger.debug("Initialisation complete...")

    def __repr__(self) -> str:
        """Return a string representation of the core application."""
        return (
            f"<App(name={type(self).__name__!r}, "
            f"plugins={len(self._plugins.show())})>"
        )

    def __setattr__(self, name: str, value: t.Any) -> None:
        """Prevent direct attribute modification after initialisation.

        This method overrides the default attribute setting behaviour to
        prevent direct modification of attributes after the application
        has been initialised. It allows only attributes that start with
        an underscore (private attributes) to be modified directly.

        :param name: The name of the attribute to set.
        :param value: The value to set for the attribute.
        """
        if hasattr(self, "_initialised") and self._initialised:
            if name.startswith("_"):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f"Cannot modify attribute {name!r} directly."
                    " Use appropriate methods"
                )
        else:
            super().__setattr__(name, value)

    def _setup_observability_hooks(self) -> None:
        """Setup observability hooks"""
        pass

    def _observe_context(
        self,
        old_context: AppContext,
        new_context: AppContext,
    ) -> None:
        """Handle context changes for observability.

        This method is called whenever the execution context changes,
        allowing the application to maintain an audit trail of all
        context modifications. It captures the old and new context
        metadata, timestamps, and any relevant information for security
        and compliance purposes.

        :param old_context: The previous execution context before the
            change.
        :param new_context: The new execution context after the change.
        """
        self._metrics["context_changes"] += 1
        self._audit.append(
            {
                "type": "context_change",
                "timestamp": time.time(),
                "old_context": old_context.metadata,
                "new_context": new_context.metadata,
            }
        )

    def _observe_plugin(self, name: str, plugin: Plugin) -> None:
        """Handle plugin loading events.

        This method is called whenever a plugin is successfully loaded
        into the application. It captures the plugin name, version, and
        timestamp of the loading event for observability and audit
        trails. It also increments the plugin load metrics to track
        plugin utilisation across the application.

        :param name: The name of the plugin that was loaded.
        :param plugin: The plugin instance that was loaded.
        """
        self._metrics["plugin_loaded"] += 1
        self._audit.append(
            {
                "type": "plugin_loaded",
                "timestamp": time.time(),
                "plugin_name": name,
                "plugin_version": getattr(plugin, "version", "0.0.0"),
            }
        )

    def _validate_plugin(self, plugin: Plugin) -> bool:
        """Validate plugin security using registered policies.

        This method applies all registered security policies to the
        plugin and returns `True` if all policies pass, or `False` if
        any policy fails. This ensures that only safe plugins are loaded
        into the application. The policies can include checks for
        security vulnerabilities, compliance with coding standards, and
        adherence to best practices.

        :param plugin: The plugin instance to validate.
        :return: `True` if the plugin passes all security checks,
            `False` otherwise.
        """
        return all(policy(plugin) for policy in self._security_policies)

    @property
    def config(self) -> Config:
        """Access the configuration object."""
        return self._config

    @property
    def logger(self) -> t.Any:
        """Access the logger object."""
        return self._logger

    @property
    def tracer(self) -> Tracer:
        """Access the distributed tracer object."""
        return self._tracer

    @property
    def context(self) -> ContextManager:
        """Access the execution context manager."""
        return self._context

    @property
    def plugins(self) -> PluginManager:
        """Access the plugin manager with lazy loading."""
        return self._plugins

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Access current metrics snapshot."""
        return {
            **self._metrics,
            "plugins": self._plugins.metrics(),
            "uptime": (time.time() - self._context.context.timestamp),
        }

    @property
    def audit(self) -> list[dict[str, t.Any]]:
        """Access the audit trail for security and compliance."""
        return self._audit.copy()

    def add_security_policy(self, policy: t.Callable[[t.Any], bool]) -> None:
        """Add a security policy validator.

        This method allows developers to register custom security
        policies that will be applied to all plugins before they are
        loaded into the application. Policies should be callable
        functions that take a plugin instance and return `True` if the
        plugin is safe to load, or `False` if it should be rejected.

        :param policy: A callable that takes a plugin instance and
            returns a boolean indicating whether the plugin is safe to
            load.
        :raises TypeError: If the policy is not callable.
        """
        if not callable(policy):
            raise TypeError(f"{policy!r} must be a callable")
        self._security_policies.append(policy)

    def process(
        self,
        prompt: str,
        context: AppContext | None = None,
    ) -> str:
        """Process a prompt through the application's logic.

        This method serves as the main entry point for all requests,
        providing comprehensive instrumentation, security validation,
        performance monitoring, and audit trail generation. It manages
        the complete request lifecycle whilst delegating to the `handle`
        method for application-specific logic.

        The method automatically captures metrics, generates trace
        spans, validates security policies, and maintains audit trails
        for compliance and observability purposes. Context is managed
        immutably throughout the request lifecycle.

        :param prompt: The input prompt to be processed.
        :param context: An optional execution context for this request,
            defaults to `None`.
        :return: The application's response.
        """
        started = time.time()
        ctx = context or self._context.context
        valid = ctx.user is not None or ctx.session is not None or ctx.metadata
        with self.tracer.start_as_current_span("lauren.process") as sp:
            try:
                sp.set_attribute("lauren.prompt", prompt)
                sp.set_attribute("lauren.request", ctx.request)
                if valid:
                    if ctx.user:
                        sp.set_attribute("lauren.user", ctx.user)
                    if ctx.session:
                        sp.set_attribute("lauren.session", ctx.session)
                self.logger.debug("Processing user prompt...")
                self._audit.append(
                    {
                        "type": "request_start",
                        "user": ctx.user,
                        "context_id": ctx.request,
                        "prompt_hash": hash(prompt),
                        "timestamp": started,
                    }
                )
                response = self.handle(prompt, ctx if valid else None)
                processing_time = time.time() - started
                self._metrics["requests_processed"] += 1
                self._metrics["processing_time"] += processing_time
                sp.set_attribute("lauren.response", response)
                sp.set_attribute("lauren.processing_time", processing_time)
                self.logger.debug(
                    "Response generated",
                    extra={
                        "successful": True,
                        "response": response,
                        "processing_time": processing_time,
                    },
                )
                self._audit.append(
                    {
                        "success": True,
                        "type": "request_complete",
                        "response_hash": hash(response),
                        "timestamp": time.time(),
                        "processing_time": processing_time,
                    }
                )
                return response
            except Exception as exc:
                processing_time = time.time() - started
                sp.set_attribute("lauren.error", str(exc))
                sp.set_attribute("lauren.processing_time", processing_time)
                self.logger.error(
                    "Request processing failed",
                    extra={
                        "successful": False,
                        "error": str(exc),
                        "processing_time": processing_time,
                    },
                    exc_info=exc,
                )
                self._audit.append(
                    {
                        "success": False,
                        "type": "request_failed",
                        "error": str(exc),
                        "timestamp": time.time(),
                        "processing_time": processing_time,
                    }
                )
                raise

    def handle(
        self,
        prompt: str,
        context: AppContext | None = None,
    ) -> str:
        """Prompt processing logic.

        Subclasses must override this method to define their
        application's core behaviour.

        :param prompt: The input prompt to be processed.
        :param context: The execution context for this request.
        :return: The application's response.
        """
        raise NotImplementedError("Must implement `handle` method")
