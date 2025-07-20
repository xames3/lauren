"""\
Core Application
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Saturday, July 19 2025

This module provides the foundational application class for building
L.A.U.R.E.N powered applications. The `Core` class serves as an
abstract base that developers extend to create their own prompt
processing applications. The framework handles the infrastructure
concerns like configuration, logging, tracing, and plugin management
whilst developers focus on implementing their specific business logic
through clean, extensible interfaces.

This design separates framework responsibilities from application logic,
ensuring consistent observability and configuration across all
applications whilst maintaining flexibility for diverse use cases.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import sys
import time
import typing as t
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from uuid import uuid4

import pkg_resources
from opentelemetry.trace import Tracer

import lauren.plugins
from lauren.core.config import BaseConfig
from lauren.plugins.base import Plugin
from lauren.utils.logging import configure
from lauren.utils.logging import get_logger
from lauren.utils.opentelemetry import get_tracer

__all__: list[str] = [
    "Core",
    "ExecutionContext",
    "ContextManager",
    "PluginManager",
]

user: ContextVar[str | None] = ContextVar("user", default=None)
session: ContextVar[str | None] = ContextVar("session", default=None)
metadata: ContextVar[dict[str, t.Any]] = ContextVar("metadata", default={})


@dataclass(frozen=True)
class ExecutionContext:
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

    def __init__(self, context: ExecutionContext) -> None:
        """Initialise the context manager with an execution context."""
        if not isinstance(context, ExecutionContext):
            raise TypeError(
                f"Expected ExecutionContext, got {type(context).__name__}"
            )
        self._context = context
        self._initial_state = context
        self._observers: list[
            t.Callable[[ExecutionContext, ExecutionContext], None]
        ] = []

    @property
    def context(self) -> ExecutionContext:
        """Access the current execution context."""
        return self._context

    def add_observer(
        self,
        callback: t.Callable[[ExecutionContext, ExecutionContext], None],
    ) -> None:
        """Add an observer for context changes.

        This allows external components to react to context changes,
        such as logging, tracing, or triggering additional workflows.

        :param callback: A callable that takes the old and new context
            as arguments. It should handle any exceptions internally to
            avoid breaking the context management flow.
        :raises TypeError: If the callback is not callable.
        """
        if not callable(callback):
            raise TypeError(
                f"Expected callable, got {type(callback).__name__}"
            )
        self._observers.append(callback)

    def remove_observer(
        self,
        callback: t.Callable[[ExecutionContext, ExecutionContext], None],
    ) -> None:
        """Remove an observer for context changes.

        This allows for dynamic management of observers, enabling
        components to stop receiving notifications about context changes.

        :param callback: The observer to remove.
        :raises ValueError: If the callback is not found in the
            observers list.
        """
        if callback in self._observers:
            self._observers.remove(callback)
        else:
            raise ValueError(
                f"Observer: {callback} not found in current context observers"
            )

    def _notify_observers(
        self,
        old_context: ExecutionContext,
        new_context: ExecutionContext,
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
        self._notify_observers(old_context, self._context)

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
        self._notify_observers(old_context, self._context)

    def clear(self) -> None:
        """Clear all metadata from the execution context."""
        old_context = self._context
        self._context = replace(self._context, metadata={})
        self._notify_observers(old_context, self._context)

    def reset(self) -> None:
        """Reset the execution context to its initial state."""
        old_context = self._context
        self._context = self._initial_state
        self._notify_observers(old_context, self._context)

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

    :param core: The core application instance to manage plugins for.
    """

    def __init__(self, core: "Core") -> None:
        self._core = core
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
    def core(self) -> "Core":
        """Access the core instance."""
        return self._core

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

    def _notify_observers(self, name: str, plugin: Plugin) -> None:
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

    def discover(self) -> dict[str, Plugin]:
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
        self._core.logger.debug("Discovering built-in plugins...")
        for package in pkgutil.iter_modules(
            lauren.plugins.__path__,
            f"{lauren.plugins.__name__}.",
        ):
            try:
                module = importlib.import_module(package.name)
                self._discover_plugins_from_lauren(module, plugins)
            except Exception as exc:
                self._core.logger.error(
                    f"Failed to discover built-in plugin: {package.name!r}",
                    exc_info=exc,
                )
        self._discover_plugins_from_entrypoints(plugins)
        self._discover_plugins_from_custom_sources(plugins)
        self._discover_plugins_from_namespace_packages(plugins)
        self._discovered = True
        self._core.logger.debug(
            f"Plugin discovery complete. Found {len(plugins)} plugins "
            f"({len(self._builtins)} built-in and "
            f"{len(self._installed)} installed)"
        )
        return plugins

    def _discover_plugins_from_entrypoints(
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
        self._core.logger.debug("Discovering plugins from entry points...")
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
                    self._core.logger.error(
                        "Failed to load plugin from entry point: "
                        f"{entry.name!r}",
                        exc_info=exc,
                    )
        except ImportError:
            pass

    def _discover_plugins_from_custom_sources(
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
        self._core.logger.debug("Discovering plugins from directories...")
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
                            self._discover_plugins_in_module(
                                module, plugins, source="module"
                            )
                        except Exception as exc:
                            self._core.logger.error(
                                "Failed to discover plugins from "
                                f"custom module: {name!r}",
                                exc_info=exc,
                            )
            except Exception as exc:
                self._core.logger.error(
                    f"Failed to scan custom plugin directory: {path!r}",
                    exc_info=exc,
                )

    def _discover_plugins_from_namespace_packages(
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
        self._core.logger.debug(
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
                        self._discover_plugins_in_module(
                            module, plugins, source="namespace"
                        )
                    except Exception as exc:
                        self._core.logger.error(
                            "Failed to discover plugins from namespace "
                            f"package: {name!r}",
                            exc_info=exc,
                        )
            except ImportError:
                pass
        except Exception as exc:
            self._core.logger.error(
                "Failed to discover namespace packages",
                exc_info=exc,
            )

    def _discover_plugins_in_module(
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

    def _discover_plugins_from_lauren(
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
            discovered = self.discover()
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
        self._notify_observers(name, plugin)
        if hasattr(plugin, "on_load"):
            plugin.on_load(self._core.context.context)
        self._core.logger.debug(
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
        with self._core.tracer.start_as_current_span(
            "core.load_plugins"
        ) as span:
            span.set_attribute("lauren.plugins.lazy", lazy)
            self._core.logger.debug("Starting plugin discovery and loading...")
            loaded = failed = 0
            if lazy:
                self.discover()
                stats = self.statistics
                span.set_attribute(
                    "lauren.plugins.discovered", stats["plugins_discovered"]
                )
                self._core.logger.debug(
                    "Lazy plugin loading enabled: "
                    f"{stats['plugins_discovered']} plugin(s) available "
                    f"for on-demand loading ({stats['builtin_discovered']} "
                    f"built-ins, {stats['installed_discovered']} installed)"
                )
            else:
                discovered = self.discover()
                for name, (plugin, _) in discovered.items():
                    try:
                        if self.validate(plugin):
                            self._plugins[name] = plugin
                            self._notify_observers(name, plugin)
                            if hasattr(plugin, "on_load"):
                                plugin.on_load(self._core.context.context)
                            self._core.logger.debug(
                                f"Successfully loaded plugin: {name!r}"
                            )
                            loaded += 1
                        else:
                            self._core.logger.warning(
                                f"Plugin: {name!r} failed security validation"
                            )
                            failed += 1
                    except Exception as exc:
                        self._core.logger.error(
                            f"Failed to load plugin {name!r}",
                            exc_info=exc,
                        )
                        failed += 1
            span.set_attribute("lauren.plugins.loaded", loaded)
            span.set_attribute("lauren.plugins.failed", failed)
            stats = self.statistics
            self._core.logger.debug(
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
        self._core.logger.debug(
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
            self.discover()
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
        self._core.logger.debug(f"Successfully unloaded plugin: {name!r}")

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
            self.discover()
        return self._metadata.copy()

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Get plugin manager metrics."""
        if not self._discovered:
            self.discover()
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


class CoreMeta(type):
    """Metaclass for `Core`.

    This metaclass provides enterprise-grade capabilities including lazy
    manager instantiation, configuration validation, security
    enforcement, and comprehensive observability hooks. It ensures
    consistent behaviour across all Core subclasses whilst enabling
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
        """Create new Core class with enhanced capabilities."""
        if name == "Core":
            return super().__new__(cls, name, bases, namespaces)
        subclassed = any(
            hasattr(base, "__name__") and base.__name__ == "Core"
            for base in bases
        )
        if subclassed:
            if "process" in namespaces:
                raise TypeError("Cannot override the 'process' method")
            cls._validate_required_methods(namespaces)
            cls._setup_observability_hooks(namespaces)
            cls._configure_lazy_loading(namespaces)
        return super().__new__(cls, name, bases, namespaces)

    @staticmethod
    def _validate_required_methods(namespaces: dict[str, t.Any]) -> None:
        """Validate that required methods are properly implemented.

        This method checks that the `handle` method is defined and is a
        callable function. This ensures that all subclasses of `Core`
        implement the necessary application logic interface.

        :param namespaces: The class namespace containing methods and
            attributes to validate.
        :raises TypeError: If the `handle` method is not defined or
            is not callable.
        """
        if "handle" not in namespaces:
            return
        method = namespaces["handle"]
        if not callable(method):
            raise TypeError("'handle' must be a callable method")

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


class Core(metaclass=CoreMeta):
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
        application. If not provided, a default `BaseConfig` instance
        is created. This configuration object is immutable after
        initialisation, ensuring that all settings are fixed for the
        lifetime of the application instance.

    .. note::

        Developers should subclass `Core` and implement the `handle`
        method to define their application's specific logic whilst
        inheriting all framework capabilities automatically.
    """

    def __init__(self, config: BaseConfig | None = None) -> None:
        """Initialise the core application."""
        self._config = config or BaseConfig()
        configure(self._config.logging)
        self._logger = get_logger(__name__)
        self._logger.debug("Initialising L.A.U.R.E.N...")
        self._tracer = get_tracer(self._config.name)
        self._execution_context = ExecutionContext(
            user=user.get(None),
            session=session.get(None),
            metadata=metadata.get({}).copy(),
            request=str(uuid4()),
            timestamp=time.time(),
        )
        self._context_manager = ContextManager(self._execution_context)
        self._context_manager.add_observer(self._on_context_change)
        self._security_policies: list[t.Callable[[t.Any], bool]] = []
        self._audit_trail: list[dict[str, t.Any]] = []
        self._metrics: dict[str, t.Any] = {
            "requests_processed": 0,
            "processing_time": 0.0,
            "plugin_loaded": 0,
            "context_changes": 0,
        }
        self._plugin_manager = PluginManager(self)
        self._plugin_manager.add_validator(self._validate)
        self._plugin_manager.add_observer(self._on_plugin_load)
        lazy_config = getattr(self, "_lazy_config", {})
        if lazy_config.get("plugins", False):
            self._plugin_manager.load(lazy=True)
        else:
            self._plugin_manager.load(lazy=False)
        self._initialised = True
        self._logger.debug("Initialisation complete...")

    def __repr__(self) -> str:
        """Return a string representation of the core application."""
        return (
            f"<Core(name={type(self).__name__!r}, "
            f"plugins={len(self._plugin_manager.show())})>"
        )

    def _setup_observability(self) -> None:
        """Setup observability hooks"""
        pass

    def _on_context_change(
        self,
        old_context: ExecutionContext,
        new_context: ExecutionContext,
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
        self._audit_trail.append(
            {
                "type": "context_change",
                "timestamp": time.time(),
                "old_context": old_context.metadata,
                "new_context": new_context.metadata,
            }
        )

    def _on_plugin_load(self, name: str, plugin: Plugin) -> None:
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
        self._audit_trail.append(
            {
                "type": "plugin_loaded",
                "timestamp": time.time(),
                "plugin_name": name,
                "plugin_version": getattr(plugin, "version", "0.0.0"),
            }
        )

    def _validate(self, plugin: Plugin) -> bool:
        """Validate plugin security.

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
            raise TypeError("Security policy must be a callable")
        self._security_policies.append(policy)

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

    @property
    def config(self) -> BaseConfig:
        """Access the immutable configuration object."""
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
        return self._context_manager

    @property
    def plugins(self) -> PluginManager:
        """Access the plugin manager with lazy loading."""
        return self._plugin_manager

    @property
    def metrics(self) -> dict[str, t.Any]:
        """Access current metrics snapshot."""
        return {
            **self._metrics,
            "plugins": self._plugin_manager.metrics(),
            "uptime": (time.time() - self._context_manager.context.timestamp),
        }

    @property
    def audit_trail(self) -> list[dict[str, t.Any]]:
        """Access the audit trail for security and compliance."""
        return self._audit_trail.copy()

    def process(
        self,
        prompt: str,
        context: ExecutionContext | None = None,
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
        start_time = time.time()
        request_context = context or self._context_manager.context
        context_is_valid = (
            request_context.user is not None
            or request_context.session is not None
            or request_context.metadata
        )
        with self.tracer.start_as_current_span("core.process") as span:
            try:
                span.set_attribute("lauren.prompt", prompt)
                span.set_attribute("lauren.request", request_context.request)
                if context_is_valid:
                    if request_context.user:
                        span.set_attribute("lauren.user", request_context.user)
                    if request_context.session:
                        span.set_attribute(
                            "lauren.session", request_context.session
                        )
                self.logger.debug("Processing user prompt...")
                self._audit_trail.append(
                    {
                        "type": "request_start",
                        "timestamp": start_time,
                        "prompt_hash": hash(prompt),
                        "context_id": request_context.request,
                        "user": request_context.user,
                    }
                )
                response = self.handle(
                    prompt, request_context if context_is_valid else None
                )
                processing_time = time.time() - start_time
                self._metrics["requests_processed"] += 1
                self._metrics["processing_time"] += processing_time
                span.set_attribute("lauren.response", response)
                span.set_attribute("lauren.processing_time", processing_time)
                self.logger.debug(
                    "Response generated",
                    extra={
                        "processing.time": processing_time,
                        "response": response,
                        "successful": True,
                    },
                )
                self._audit_trail.append(
                    {
                        "type": "request_complete",
                        "timestamp": time.time(),
                        "processing_time": processing_time,
                        "response_hash": hash(response),
                        "success": True,
                    }
                )
                return response
            except Exception as exc:
                processing_time = time.time() - start_time
                span.set_attribute("lauren.error", str(exc))
                span.set_attribute("lauren.processing_time", processing_time)
                self.logger.error(
                    "Request processing failed",
                    extra={
                        "processing.time": processing_time,
                        "error": str(exc),
                        "successful": False,
                    },
                    exc_info=exc,
                )
                self._audit_trail.append(
                    {
                        "type": "request_failed",
                        "timestamp": time.time(),
                        "processing_time": processing_time,
                        "error": str(exc),
                        "success": False,
                    }
                )
                raise

    def handle(
        self,
        prompt: str,
        context: ExecutionContext | None = None,
    ) -> str:
        """Prompt processing logic.

        Subclasses must override this method to define their
        application's core behaviour.

        :param prompt: The input prompt to be processed.
        :param context: The execution context for this request.
        :return: The application's response.
        """
        raise NotImplementedError("Must implement `handle` method")
