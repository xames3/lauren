"""\
Plugin
======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, July 20 2025
Last updated on: Monday, July 21 2025

This module provides the foundational plugin architecture for
L.A.U.R.E.N, enabling modular, extensible, and enterprise-grade plugin
development. The `Plugin` class serves as an abstract base that
developers extend to create their own specialised plugins.

The plugin architecture emphasises immutability, security, and
observability whilst maintaining developer productivity through clear
abstractions and comprehensive lifecycle management.

This design separates plugin logic from framework responsibilities,
ensuring consistent behaviour and security across all plugins whilst
maintaining flexibility for diverse plugin implementations and use cases.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import time
import typing as t
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field

import pkg_resources

import lauren.plugins
from lauren.core.config import Config

if t.TYPE_CHECKING:
    from lauren.core.app import App

__all__: list[str] = [
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PluginMeta",
    "SecurityError",
]


@dataclass(frozen=True)
class PluginContext:
    """Execution context for plugin operations.

    This context captures the essential state information for plugin
    operations within the framework. It provides a snapshot of the
    plugin's runtime environment, configuration, and metadata that
    flows through the entire plugin lifecycle whilst maintaining
    immutability for security and consistency.

    The context is designed to be lightweight and serialisable, making
    it suitable for distributed plugin execution, caching, and audit
    trails whilst ensuring thread-safety and immutability.

    .. note::

        The `PluginContext` is immutable after creation, ensuring that
        all operations on the context maintain a consistent state
        across the plugin's lifecycle.
    """

    name: str
    version: str
    context: t.Any = field(default=None)
    config: Config | None = field(default=None)
    metadata: dict[str, t.Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    trace: str | None = field(default=None)


class PluginMeta(ABCMeta):
    """Metaclass for `Plugin`.

    This metaclass ensures consistent behaviour across all plugin
    implementations whilst enabling dynamic customisation and
    performance optimisation. It automatically validates plugin
    contracts, establishes security boundaries, configures observability
    infrastructure, and enforces immutability patterns at class creation
    time.

    This ensures that all plugin instances inherit proper framework
    behaviour whilst maintaining flexibility for plugin-specific
    customisation.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespaces: dict[str, t.Any],
    ) -> type:
        """Create new `Plugin` class with some necessary checks."""
        if name == "Plugin":
            return super().__new__(cls, name, bases, namespaces)
        subclassed = any(
            hasattr(base, "__name__") and base.__name__ == "Plugin"
            for base in bases
        )
        if subclassed:
            cls._validate_properties(namespaces)
            cls._validate_methods(namespaces)
            cls._setup_security_boundaries(namespaces)
            cls._setup_observability_hooks(namespaces)
            cls._enforce_immutability(namespaces)
        return super().__new__(cls, name, bases, namespaces)

    @staticmethod
    def _validate_properties(namespaces: dict[str, t.Any]) -> None:
        """Validate that required properties are implemented.

        This method ensures that plugin subclasses implement the
        essential properties required by the framework's plugin
        contract. All plugins must provide a unique name and version
        for proper identification and lifecycle management.

        :param namespaces: The class namespace containing properties
            to validate.
        :raises TypeError: If required properties are not properly
            implemented.
        """
        properties = {"name", "version"}
        for _property in properties:
            if _property in namespaces:
                descriptor = namespaces[_property]
                if not isinstance(descriptor, property):
                    raise TypeError(f"{_property!r} must be a property")

    @staticmethod
    def _validate_methods(namespaces: dict[str, t.Any]) -> None:
        """Validate plugin lifecycle method signatures.

        This method ensures that lifecycle methods follow the correct
        signatures and conventions expected by the framework. It
        validates that methods are callable and have appropriate
        parameter expectations.

        :param namespaces: The class namespace containing methods to
            validate.
        :raises TypeError: If methods are not defined or not callable.
        """
        methods = {"on_load", "on_configure", "on_shutdown", "on_validate"}
        for _method in methods:
            if _method in namespaces:
                method = namespaces[_method]
                if not callable(method):
                    raise TypeError(f"{_method!r} must be a callable")

    @staticmethod
    def _setup_security_boundaries(namespaces: dict[str, t.Any]) -> None:
        """Setup security boundaries for plugin execution.

        This method establishes security constraints and validation
        hooks that ensure plugins operate within secure boundaries. It
        configures default security policies and validation mechanisms
        that are applied to all plugin operations.

        :param namespaces: The class namespace to configure with
            security boundaries.
        """
        if "_security_policies" not in namespaces:
            namespaces["_security_policies"] = []
        if "_validation_hooks" not in namespaces:
            namespaces["_validation_hooks"] = []

    @staticmethod
    def _setup_observability_hooks(namespaces: dict[str, t.Any]) -> None:
        """Configure observability hooks for the plugin.

        This method sets up observability hooks, metrics collection,
        and audit trail capabilities that enable comprehensive
        monitoring and troubleshooting of plugin operations.

        :param namespaces: The class namespace to configure with
            observability hooks.
        """
        if "_observability_hooks" not in namespaces:
            namespaces["_observability_hooks"] = {
                "metrics_enabled": True,
                "tracing_enabled": True,
                "audit_enabled": True,
            }

    @staticmethod
    def _enforce_immutability(namespaces: dict[str, t.Any]) -> None:
        """Enforce immutability for plugin instances.

        This method configures immutability enforcement that prevents
        unauthorised modification of plugin state after initialisation.
        It ensures plugin instances maintain consistent state throughout
        their lifecycle.

        :param namespaces: The class namespace to configure with
            immutability enforcement.
        """
        if "_immutability_post_init" not in namespaces:
            namespaces["_immutability_post_init"] = True


class Plugin(metaclass=PluginMeta):
    """Foundational class for creating secure, observable plugins.

    This class serves as the abstract foundation for all L.A.U.R.E.N
    plugins, providing enterprise-grade capabilities including security
    validation, lifecycle management, observability hooks, and
    immutability enforcement. It manages the complete plugin lifecycle
    with built-in audit trails, performance monitoring, and security
    boundaries.

    The plugin architecture implements strict immutability patterns for
    security and consistency whilst maintaining flexibility for diverse
    plugin implementations. All operations flow through this class,
    providing centralised control, comprehensive observability, and
    security validation for enterprise deployments.

    Developers should subclass `Plugin` and implement the required
    abstract properties whilst optionally overriding lifecycle methods
    to define their plugin's specific behaviour. The framework
    automatically handles infrastructure concerns, allowing developers
    to focus on plugin logic.

    .. note::

        All plugin instances are immutable after initialisation and
        operate within strict security boundaries enforced by the
        framework's security infrastructure.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initialise the plugin with configuration and security setup."""
        self._config = config
        self._audit: list[dict[str, t.Any]] = []
        self._security_policies: list[t.Callable[[t.Any], bool]] = []
        self._validation_hooks: list[t.Callable[[], bool]] = []
        self._context: PluginContext | None = None
        self._metrics: dict[str, t.Any] = {
            "load_time": 0.0,
            "configure_time": 0.0,
            "execution_count": 0,
            "last_execution": None,
        }
        self._observers: list[t.Callable[[str, t.Any], None]] = []
        self._setup_default_policies()
        self._initialised = True

    def __repr__(self) -> str:
        """Return a string representation of the plugin."""
        try:
            return (
                f"<{type(self).__name__}(name={self.name!r}, "
                f"version={self.version!r})>"
            )
        except NotImplementedError:
            return f"<{type(self).__name__}(incomplete)>"

    def __setattr__(self, name: str, value: t.Any) -> None:
        """Prevent direct attribute modification after initialisation.

        This method enforces immutability by preventing direct
        modification of plugin attributes after initialisation. It
        allows only private attributes to be modified directly whilst
        requiring controlled modification through appropriate methods.

        :param name: The name of the attribute to set.
        :param value: The value to set for the attribute.
        :raises AttributeError: If attempting to modify a public
            attribute after initialisation.
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

    def _setup_default_policies(self) -> None:
        """Setup default security policies for the plugin."""

        def validate_plugin_contract() -> bool:
            """Validate that basic contract requirements are met."""
            try:
                return bool(self.name and self.version)
            except (NotImplementedError, AttributeError):
                return False

        self._validation_hooks.append(validate_plugin_contract)

    def _validate_plugin(self) -> bool:
        """Validate plugin security using registered policies.

        This method applies all registered security policies and
        validation hooks to ensure the plugin operates within security
        boundaries. It returns `True` if all validations pass, or
        `False` if any validation fails.

        :return: `True` if the plugin passes all security validations,
            `False` otherwise.
        """
        try:
            policies = [policy(self) for policy in self._security_policies]
            hooks = [hook() for hook in self._validation_hooks]
            return all(policies + hooks)
        except Exception:
            return False

    def _record_event(self, event: str, **kwargs: t.Any) -> None:
        """Record an audit event for compliance and security monitoring.

        This method captures audit events with timestamps and relevant
        metadata for security monitoring, compliance reporting, and
        troubleshooting purposes.

        :param event: The type of audit event being recorded.
        """
        record = {
            "type": event,
            "name": getattr(self, "name", "0.0.0"),
            "version": getattr(self, "version", "0.0.0"),
            **kwargs,
            "timestamp": time.time(),
        }
        self._audit.append(record)

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the plugin.

        This property must return a unique identifier for the plugin
        that distinguishes it from all other plugins in the system.
        The name should be stable across plugin versions and should
        follow naming conventions for consistency.

        :return: A unique string identifier for the plugin.
        """
        raise NotImplementedError("Must implement the 'name' property")

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the plugin.

        This property must return a version string. The version is used
        for dependency resolution, compatibility checking, and plugin
        lifecycle management.

        :return: A version string for the plugin.
        """
        raise NotImplementedError("Must implement the 'version' property")

    @property
    def config(self) -> Config | None:
        """Access the configuration object."""
        return self._config

    @property
    def context(self) -> PluginContext | None:
        """Access the execution context."""
        return self._context

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Access current metrics snapshot."""
        return self._metrics.copy()

    @property
    def audit(self) -> list[dict[str, t.Any]]:
        """Access the audit trail for security and compliance."""
        return self._audit.copy()

    def add_observer(self, observer: t.Callable[[str, t.Any], None]) -> None:
        """Add an observer for plugin lifecycle events.

        This method allows external components to react to plugin
        lifecycle events, such as loading, configuration, or execution.
        Observers can perform additional actions based on plugin state
        changes.

        :param observer: A callable that takes event name and data as
            arguments.
        :raises TypeError: If the observer is not callable.
        """
        if not callable(observer):
            raise TypeError(f"{observer!r} must be a callable")
        self._observers.append(observer)

    def notify_observers(self, event: str, data: t.Any) -> None:
        """Notify observers of plugin events.

        This method calls all registered observers with the event name
        and associated data, allowing them to react to plugin lifecycle
        events.

        :param event: The name of the event that occurred.
        :param data: The data associated with the event.
        """
        for observer in self._observers:
            try:
                observer(event, data)
            except Exception:
                pass

    def add_security_policy(self, policy: t.Callable[[t.Any], bool]) -> None:
        """Add a security policy validator.

        This method allows registration of custom security policies
        that are applied during plugin operations. Policies should
        return `True` if the operation is permitted, or `False` if it
        should be rejected.

        :param policy: A callable that validates plugin operations.
        :raises TypeError: If the policy is not callable.
        """
        if not callable(policy):
            raise TypeError(f"{policy!r} must be a callable")
        self._security_policies.append(policy)

    def on_load(self, context: t.Any) -> None:
        """Method invoked when the plugin is loaded into the framework.

        This lifecycle method is invoked when the plugin is being loaded
        by the plugin manager. It provides an opportunity for the plugin
        to perform initialisation tasks, validate its environment, and
        prepare for operation.

        :param context: The execution context for the loading operation.
        """
        started = time.time()
        if not self._validate_plugin():
            raise SecurityError(
                f"Plugin: {self.name!r} failed security validation"
            )
        self._context = PluginContext(
            name=self.name,
            version=self.version,
            context=context,
            timestamp=started,
        )
        load_time = time.time() - started
        self._metrics["load_time"] = load_time
        self._record_event("plugin_loaded", load_time=load_time)
        self.notify_observers("load", {"context": context})

    def on_configure(self, config: Config) -> None:
        """Method invoked when the application is configuring the plugin.

        This lifecycle method is invoked during the configuration phase,
        allowing the plugin to access framework config and adjust its
        behaviour accordingly. The plugin should validate configuration
        values and prepare its internal state.

        :param config: The framework configuration object containing
            settings that may affect plugin behaviour.
        """
        started = time.time()
        self._config = config
        if self._context:
            self._context = PluginContext(
                name=self._context.name,
                version=self._context.version,
                context=self._context.context,
                config=config,
                metadata=self._context.metadata,
                timestamp=self._context.timestamp,
                trace=self._context.trace,
            )
        configure_time = time.time() - started
        self._metrics["configure_time"] = configure_time
        self._record_event("plugin_configured", configure_time=configure_time)
        self.notify_observers("configure", {"config": config})

    def on_shutdown(self) -> None:
        """Method invoked when the application is shutting down.

        This lifecycle method is invoked during application shutdown,
        allowing the plugin to perform cleanup tasks, release resources,
        and ensure graceful termination of any ongoing operations.
        """
        self._record_event("plugin_shutdown")
        self.notify_observers("shutdown", {})

    def on_validate(self) -> bool:
        """Method invoked to validate the plugin's current state.

        This lifecycle method allows the plugin to perform
        self-validation checks and report whether it is in a valid state
        for operation. This is useful for health checks and diagnostic
        purposes.

        :return: `True` if the plugin is in a valid state, `False`
            otherwise.
        """
        return self._validate_plugin()


class SecurityError(Exception):
    """Exception raised when plugin security validation fails."""

    pass


class PluginManager:
    """Class to manage plugins.

    This manager provides enterprise-grade plugin management with lazy
    loading, security validation, dependency resolution, and
    comprehensive observability. It maintains plugin isolation whilst
    enabling efficient resource utilisation through on-demand loading
    strategies.

    The manager integrates deeply with framework's security and
    observability infrastructure, providing audit trails, performance
    metrics, and security scanning for all plugin operations. It
    supports hot-reload capabilities for development environments whilst
    maintaining strict security controls for production deployments.

    The manager is designed to be extensible, allowing developers to
    register custom plugins, security validators, and load observers
    that integrate seamlessly with the framework's lifecycle. It
    automatically discovers plugins from the filesystem, supports lazy
    loading patterns, and provides a consistent interface for plugin
    registration and management.

    :param app: The core application instance to manage plugins for.
    """

    def __init__(self, app: App) -> None:
        self._app = app
        self._class: dict[str, str] = {}
        self._plugins: dict[str, Plugin] = {}
        self._builtins: set[str] = set()
        self._installed: set[str] = set()
        self._lazy: dict[str, t.Callable[[], Plugin]] = {}
        self._metadata: dict[str, dict[str, t.Any]] = {}
        self._validators: list[t.Callable[[Plugin], bool]] = []
        self._observers: list[t.Callable[[str, Plugin], None]] = []
        self._discovered: bool = False

    @property
    def app(self) -> "App":
        """Access the core application instance."""
        return self._app

    def add_validator(self, validator: t.Callable[[Plugin], bool]) -> None:
        """Add a security validator for plugin loading.

        This allows custom security checks to be applied to plugins
        before they are loaded into the application. `Validators` should
        return `True` if the plugin is safe to load, or `False` if it
        should be rejected.

        :param validator: A callable that takes a `Plugin` instance and
            returns a boolean indicating whether the plugin is safe to
            load.
        """
        self._validators.append(validator)

    def add_observer(self, observer: t.Callable[[str, Plugin], None]) -> None:
        """Add an observer for plugin loading events.

        This allows external components to react to plugin loading
        events, such as logging, auditing, or triggering additional
        workflows. Observers can perform additional actions based on
        plugin loading.

        :param observer: A callable that takes the plugin name and
            plugin instance as arguments.
        """
        self._observers.append(observer)

    def validate(self, plugin: Plugin) -> bool:
        """Validate plugin security using registered validators.

        This method applies all registered security validators to the
        plugin and returns `True` if all validators pass, or `False` if
        any validator fails. This ensures that only safe plugins are
        loaded into the application.

        :param plugin: The plugin instance to validate.
        :return: `True` if the plugin passes all security checks,
            `False` otherwise.
        """
        return all(validator(plugin) for validator in self._validators)

    def notify_observers(self, name: str, plugin: Plugin) -> None:
        """Notify observers of plugin loading.

        This method calls all registered observers with the plugin name
        and plugin instance, allowing them to react to plugin loading
        events. Observers can perform additional actions such as
        logging, auditing, or triggering additional workflows based on
        plugin loading.

        :param name: The name of the plugin being loaded.
        :param plugin: The plugin instance being loaded.
        """
        for observer in self._observers:
            try:
                observer(name, plugin)
            except Exception:
                pass

    def discover_plugins(self) -> dict[str, Plugin]:
        """Discover all available plugins.

        This method scans the plugin directory for all available
        plugins, dynamically importing them and registering them in the
        plugin manager. Discovery only runs once and is cached to avoid
        repeated filesystem operations.

        :return: A dictionary mapping plugin names to their instances.
        """
        if self._discovered:
            return {name: self._lazy[name]() for name in self._lazy}
        plugins: dict[str, list[Plugin, str]] = {}
        self._app.logger.debug("Discovering built-in plugins...")
        for package in pkgutil.iter_modules(
            lauren.plugins.__path__,
            f"{lauren.plugins.__name__}.",
        ):
            try:
                module = importlib.import_module(package.name)
                self.discover_plugins_from_lauren(module, plugins)
            except Exception as exc:
                self._app.logger.error(
                    f"Failed to discover built-in plugin: {package.name!r}",
                    exc_info=exc,
                )
        self.discover_plugins_from_entrypoints(plugins)
        self.discover_plugins_from_custom_sources(plugins)
        self.discover_plugins_from_namespace_packages(plugins)
        self._discovered = True
        self._app.logger.debug(
            f"Plugin discovery complete. Found {len(plugins)} plugins "
            f"({len(self._builtins)} built-in and "
            f"{len(self._installed)} installed)"
        )
        return plugins

    def discover_plugins_from_entrypoints(
        self,
        plugins: dict[str, list[Plugin, str]],
    ) -> None:
        """Discover installed plugins.

        This method scans for plugins registered via entry points or
        other sources, adding them to the plugin manager. It ensures
        that installed plugins are properly registered and can be
        loaded dynamically.

        :param discovered: A dictionary of already discovered plugins
            to avoid duplicates.
        """
        self._app.logger.debug("Discovering plugins from entry points...")
        try:
            for entry in pkg_resources.iter_entry_points("lauren.plugins"):
                try:
                    attribute = entry.load()
                    if (
                        isinstance(attribute, type)
                        and issubclass(attribute, Plugin)
                        and attribute is not Plugin
                    ):

                        def _create_plugin(cls=attribute):
                            return cls()

                        self._lazy[attribute.__name__] = _create_plugin
                        plugin = _create_plugin()
                        if plugin.name not in plugins:
                            plugins[plugin.name] = [plugin, "entry_point"]
                            self._class[plugin.name] = attribute.__name__
                            self._metadata[plugin.name] = {
                                "module": attribute.__module__,
                                "class": attribute.__name__,
                                "version": getattr(plugin, "version", "0.0.0"),
                                "builtin": False,
                                "source": "entry_point",
                                "entry_point": entry.name,
                            }
                            self._installed.add(plugin.name)
                except Exception as exc:
                    self._app.logger.error(
                        "Failed to load plugin from entry point: "
                        f"{entry.name!r}",
                        exc_info=exc,
                    )
        except ImportError:
            pass

    def discover_plugins_from_custom_sources(
        self,
        plugins: dict[str, list[Plugin, str]],
    ) -> None:
        """Discover plugins from custom directories.

        This method looks for plugins in directories specified by
        environment variables such as `LAUREN_PLUGIN_PATH`. It allows
        users to add custom plugin directories without modifying the
        core framework.

        :param plugins: A dictionary of already discovered plugins
            to avoid duplicates.
        """
        paths: list[str] = []
        self._app.logger.debug("Discovering plugins from directories...")
        if "LAUREN_PLUGIN_PATH" in os.environ:
            paths.extend(os.environ["LAUREN_PLUGIN_PATH"].split(os.pathsep))
        if "LAUREN_PLUGINS_DIR" in os.environ:
            paths.append(os.environ["LAUREN_PLUGINS_DIR"])
        for path in paths:
            if not os.path.isdir(path):
                continue
            try:
                if path not in sys.path:
                    sys.path.insert(0, path)
                for item in os.listdir(path):
                    if item.endswith(".py") and not item.startswith("_"):
                        name = item[:-3]
                        try:
                            module = importlib.import_module(name)
                            self.discover_plugins_in_module(
                                module, plugins, source="module"
                            )
                        except Exception as exc:
                            self._app.logger.error(
                                "Failed to discover plugins from "
                                f"custom module: {name!r}",
                                exc_info=exc,
                            )
            except Exception as exc:
                self._app.logger.error(
                    f"Failed to scan custom plugin directory: {path!r}",
                    exc_info=exc,
                )

    def discover_plugins_from_namespace_packages(
        self,
        plugins: dict[str, list[Plugin, str]],
    ) -> None:
        """Discover plugins from namespace packages.

        This method scans for plugins in namespace packages that follow
        the convention `lauren_plugins.*`. This allows third-party
        packages to provide plugins without being part of the core
        framework.

        :param plugins: A dictionary of already discovered plugins
            to avoid duplicates.
        """
        self._app.logger.debug(
            "Discovering plugins from namespace packages..."
        )
        try:
            try:
                import lauren_plugins  # type: ignore[import]

                for _, name, _ in pkgutil.iter_modules(
                    lauren_plugins.__path__,
                    "lauren_plugins.",
                ):
                    try:
                        module = importlib.import_module(name)
                        self.discover_plugins_in_module(
                            module, plugins, source="namespace"
                        )
                    except Exception as exc:
                        self._app.logger.error(
                            "Failed to discover plugins from namespace "
                            f"package: {name!r}",
                            exc_info=exc,
                        )
            except ImportError:
                pass
        except Exception as exc:
            self._app.logger.error(
                "Failed to discover namespace packages",
                exc_info=exc,
            )

    def discover_plugins_in_module(
        self,
        module: t.Any,
        plugins: dict[str, list[Plugin, str]],
        source: str = "unknown",
    ) -> None:
        """Discover plugins within a specific module.

        This helper method scans a module for plugin classes and adds
        them to the discovered plugins dictionary.

        :param module: The module to scan for plugin classes.
        :param discovered: A dictionary of already discovered plugins.
        :param source: The source of the discovery for metadata.
        """
        for name in dir(module):
            attribute = getattr(module, name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, Plugin)
                and attribute is not Plugin
            ):

                def _create_plugin(cls=attribute):
                    return cls()

                self._lazy[attribute.__name__] = _create_plugin
                plugin = _create_plugin()
                if plugin.name not in plugins:
                    plugins[plugin.name] = [plugin, source]
                    self._class[plugin.name] = attribute.__name__
                    self._metadata[plugin.name] = {
                        "module": module.__name__,
                        "class": attribute.__name__,
                        "version": getattr(plugin, "version", "0.0.0"),
                        "builtin": False,
                        "source": source,
                    }
                    self._installed.add(plugin.name)

    def discover_plugins_from_lauren(
        self,
        module: t.Any,
        plugins: dict[str, list[Plugin, str]],
    ) -> None:
        """Discover built-in plugins.

        This helper method scans a module for built-in plugin classes and
        adds them to the discovered plugins dictionary.

        :param module: The module to scan for plugin classes.
        :param plugins: A dictionary of already discovered plugins.
        """
        for name in dir(module):
            attribute = getattr(module, name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, Plugin)
                and attribute is not Plugin
            ):

                def _create_plugin(cls=attribute):
                    return cls()

                self._lazy[attribute.__name__] = _create_plugin
                plugin = _create_plugin()
                plugins[plugin.name] = [plugin, "built-in"]
                self._class[plugin.name] = attribute.__name__
                self._metadata[plugin.name] = {
                    "module": module.__name__,
                    "class": attribute.__name__,
                    "version": getattr(plugin, "version", "0.0.0"),
                    "builtin": True,
                    "source": "built-in",
                }
                self._builtins.add(plugin.name)

    def load(self, name: str | None = None, lazy: bool = True) -> None:
        """Load discovered plugins.

        This method loads plugins either by name or all discovered by
        the plugin manager. If `lazy` is `True`, it will load plugins
        on-demand, allowing for efficient resource utilisation. If `lazy`
        is `False`, it will load all discovered plugins immediately.

        :param name: The name of the plugin to load. If `None`, all
            discovered plugins will be loaded.
        :param lazy: Whether to load plugins lazily or immediately,
            defaults to `True`.
        """
        if name:
            self.load_plugin(name, lazy)
        else:
            self.load_plugins(lazy)

    def load_plugin(self, name: str, lazy: bool = True) -> None:
        """Load a specific plugin.

        This method loads a plugin by its name, applying all registered
        security validators to ensure the plugin is safe to load. If the
        plugin passes validation, it is added to the plugin manager and
        observers are notified of the loading event. If the plugin fails
        validation, it raises a `ValueError`.

        :param name: The name of the plugin to load.
        :param lazy: Whether to load the plugin lazily or immediately,
            defaults to `True`.
        :raises ValueError: If the plugin is already loaded or fails
            security validation.
        """
        if name in self._plugins:
            return
        class_name = self._class.get(name, name)
        if lazy and class_name in self._lazy:
            plugin = self._lazy[class_name]()
        elif not self._discovered:
            discovered = self.discover_plugins()
            if name not in discovered:
                raise ValueError(f"Plugin: {name!r} not found")
            plugin = discovered[name]
        elif class_name in self._lazy:
            plugin = self._lazy[class_name]()
        else:
            raise ValueError(f"Plugin: {name!r} not found")
        if not self.validate(plugin):
            raise ValueError(f"Plugin: {name!r} failed security validation")
        self._plugins[name] = plugin
        self.notify_observers(name, plugin)
        if hasattr(plugin, "on_load"):
            plugin.on_load(self._app.context.context)
        self._app.logger.debug(
            f"Successfully loaded plugin: {name}",
            extra={"plugin.name": name, "plugin.lazy": lazy},
        )

    def load_plugins(self, lazy: bool = True) -> None:
        """Load all discovered plugins.

        This method loads all plugins discovered by the plugin manager,
        applying all registered security validators to ensure each
        plugin is safe to load. It logs the loading process and notifies
        observers of the loading events. If `lazy` is `True`, it will
        load plugins on-demand, allowing for efficient resource
        utilisation. If `lazy` is `False`, it will load all discovered
        plugins immediately.

        :param lazy: Whether to load plugins lazily or immediately,
            defaults to `True`.
        :raises RuntimeError: If the core application is not
            initialised.
        """
        with self._app.tracer.start_as_current_span("core.load_plugins") as sp:
            self._app.logger.debug("Starting plugin discovery and loading...")
            sp.set_attribute("lauren.plugins.lazy", lazy)
            loaded = failed = 0
            if lazy:
                self.discover_plugins()
                stats = self.statistics
                sp.set_attribute(
                    "lauren.plugins.discovered", stats["plugins_discovered"]
                )
                self._app.logger.debug(
                    "Lazy plugin loading enabled: "
                    f"{stats['plugins_discovered']} plugin(s) available "
                    f"for on-demand loading ({stats['builtin_discovered']} "
                    f"built-ins, {stats['installed_discovered']} installed)"
                )
            else:
                discovered = self.discover_plugins()
                for name, (plugin, _) in discovered.items():
                    try:
                        if self.validate(plugin):
                            self._plugins[name] = plugin
                            self.notify_observers(name, plugin)
                            if hasattr(plugin, "on_load"):
                                plugin.on_load(self._app.context.context)
                            self._app.logger.debug(
                                f"Successfully loaded plugin: {name!r}"
                            )
                            loaded += 1
                        else:
                            self._app.logger.warning(
                                f"Plugin: {name!r} failed security validation"
                            )
                            failed += 1
                    except Exception as exc:
                        self._app.logger.error(
                            f"Failed to load plugin {name!r}",
                            exc_info=exc,
                        )
                        failed += 1
            sp.set_attribute("lauren.plugins.loaded", loaded)
            sp.set_attribute("lauren.plugins.failed", failed)
            stats = self.statistics
            self._app.logger.debug(
                "Plugin loading complete. "
                f"{stats['plugins_loaded']} plugins active "
                f"({stats['builtin_loaded']} built-in and "
                f"{stats['installed_loaded']} installed), "
                f"{failed} failed to load"
            )

    def register_plugin(
        self,
        attribute: type[Plugin],
        source: str = "dynamic",
    ) -> None:
        """Register a plugin class dynamically.

        This method allows plugins to be registered at runtime without
        requiring them to be in the lauren.plugins package or registered
        via entry points.

        :param attribute: The plugin class to register.
        :param source: The source of the plugin registration, defaults
            to `dynamic`. This can be used for logging or auditing
            purposes.
        :raises TypeError: If the attribute is not a subclass of
            `Plugin`.
        """
        if not (
            isinstance(attribute, type)
            and issubclass(attribute, Plugin)
            and attribute is not Plugin
        ):
            raise TypeError("attribute must be a subclass of Plugin")

        def _create_plugin(cls=attribute):
            return cls()

        self._lazy[attribute.__name__] = _create_plugin
        plugin = _create_plugin()
        self._class[plugin.name] = attribute.__name__
        self._metadata[plugin.name] = {
            "module": attribute.__module__,
            "class": attribute.__name__,
            "version": getattr(plugin, "version", "0.0.0"),
            "builtin": False,
            "source": source,
        }
        self._installed.add(plugin.name)
        self._app.logger.debug(
            f"Registered new plugin: {plugin.name!r} from source: {source}"
        )

    def get(self, name: str) -> Plugin:
        """Get a plugin, loading it lazily if necessary.

        This method retrieves a plugin by its name, loading it lazily if
        it has not been loaded yet.

        :param name: The name of the plugin to retrieve.
        :return: The plugin instance.
        :raises ValueError: If the plugin is not found or fails to load.
        """
        if name in self._plugins:
            return self._plugins[name]
        class_name = self._class.get(name, name)
        if class_name in self._lazy:
            self.load_plugin(name, lazy=True)
            return self._plugins[name]
        if not self._discovered:
            self.discover_plugins()
            class_name = self._class.get(name, name)
            if class_name in self._lazy:
                self.load_plugin(name, lazy=True)
                return self._plugins[name]
        raise ValueError(f"Plugin: {name!r} not found or failed to load")

    def unload(self, name: str) -> None:
        """Unload a specific plugin with cleanup.

        This method unloads a plugin by its name, removing it from the
        plugin manager and performing any necessary cleanup.

        :param name: The name of the plugin to unload.
        :raises ValueError: If the plugin is not loaded.
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin: {name!r} is not loaded")
        plugin = self._plugins[name]
        if hasattr(plugin, "on_unload"):
            plugin.on_unload()
        del self._plugins[name]
        self._app.logger.debug(f"Successfully unloaded plugin: {name!r}")

    def reload(self, name: str) -> None:
        """Reload a specific plugin with hot-reload support.

        This method unloads and then reloads a plugin by its name,
        allowing for hot-reloading of plugin code without restarting the
        application.

        :param name: The name of the plugin to reload.
        """
        if name in self._plugins:
            self.unload(name)
        self.load(name, lazy=False)

    def show(self) -> dict[str, Plugin]:
        """Show all loaded plugins."""
        return self._plugins.copy()

    @property
    def builtins(self) -> dict[str, Plugin]:
        """Show only builtin plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if name in self._builtins
        }

    @property
    def installed(self) -> dict[str, Plugin]:
        """Show only installed plugins."""
        return {
            name: plugin
            for name, plugin in self._plugins.items()
            if name in self._installed
        }

    @property
    def available(self) -> dict[str, dict[str, t.Any]]:
        """Show all available plugins with metadata."""
        if not self._discovered:
            self.discover_plugins()
        return self._metadata.copy()

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Get plugin manager metrics."""
        if not self._discovered:
            self.discover_plugins()
        return {
            "loaded": len(self._plugins),
            "available": len(self._lazy),
            "builtin": len(self._builtins),
            "installed": len(self._installed),
            "validators": len(self._validators),
            "observers": len(self._observers),
        }

    @property
    def statistics(self) -> dict[str, int]:
        """Get statistics about plugin state.

        This method provides detailed statistics about all plugins in
        the system, including discovered, loaded, built-in, and
        dynamically registered plugins.

        :return: A dictionary containing plugin statistics.
        """
        builtins = 0
        installed = 0
        for plugin in self._plugins:
            if plugin in self._builtins:
                builtins += 1
            else:
                installed += 1
        return {
            "plugins_discovered": len(self._lazy) + len(self._plugins),
            "plugins_loaded": len(self._plugins),
            "builtin_discovered": len(self._builtins),
            "installed_discovered": len(self._installed),
            "builtin_loaded": builtins,
            "installed_loaded": installed,
            "lazily_available": len(self._lazy),
        }
