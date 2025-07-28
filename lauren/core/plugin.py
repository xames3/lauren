from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
import pkgutil
import sys
import time
import typing as t
from abc import abstractmethod
from typing import Any
from typing import Final
from uuid import uuid4

try:
    import pkg_resources as pkgs
except ImportError:
    pkgs = None

import lauren.plugins
from lauren.core.base import Component
from lauren.core.base import Context
from lauren.core.base import Guardian
from lauren.core.base import Inspector
from lauren.core.base import Policy
from lauren.core.base import Validator
from lauren.core.events import EventCategory
from lauren.core.events import EventSeverity
from lauren.core.exceptions import SecurityError

if t.TYPE_CHECKING:
    from lauren.core.app import Application

__all__ = [
    "Plugin",
    "PluginContext",
    "PluginManager",
    "PluginValidator",
    "PluginInspector",
    "PluginSecurityPolicy",
]

_PLUGIN_LOAD_TIMEOUT: Final[float] = 30.0
_MAX_PLUGIN_MEMORY: Final[int] = 100 * 1024 * 1024  # 100MB
_PLUGIN_SCAN_DEPTH: Final[int] = 10


class PluginContext(Context):
    """Enhanced execution context for plugin operations.

    This context extends the base L.A.U.R.E.N context with plugin-specific
    metadata whilst maintaining immutability and comprehensive tracing
    capabilities. It captures the plugin's execution environment,
    configuration, and operational state.
    """

    __slots__ = (
        "_id",
        "_version",
        "_config",
        "_phase",
    )

    def __init__(
        self,
        name: str,
        type_: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        version: str | None = None,
        config: dict[str, Any] | None = None,
        execution_phase: str | None = None,
    ) -> None:
        """Initialise plugin context.

        :param name: Context name
        :param type_: Type of context
        :param ctx_id: Unique context identifier
        :param user_id: User identifier
        :param session_id: Session identifier
        :param trace_id: Distributed tracing identifier
        :param parent_id: Parent context identifier
        :param metadata: Additional context metadata
        :param version: Plugin version
        :param config: Plugin configuration
        :param execution_phase: Current execution phase
        """
        super().__init__(
            name=name,
            type_=type_,
            user_id=user_id,
            session_id=session_id,
            trace_id=trace_id,
            parent_id=parent_id,
            metadata=metadata,
        )
        self._id = str(uuid4())[:8]
        self._version = version
        self._config = config or {}
        self._phase = execution_phase

    def __introspect__(self) -> t.Iterator[tuple[str, Any]]:
        """Show plugin context information."""
        yield from super().__introspect__()
        if self._id:
            yield "plugin", self._id
        if self._version:
            yield "version", self._version
        if self._execution_phase:
            yield "phase", self._execution_phase

    @property
    def id(self) -> str | None:
        """Get plugin ID."""
        return self._id

    @property
    def version(self) -> str | None:
        """Get plugin version."""
        return self._version

    @property
    def config(self) -> dict[str, Any]:
        """Access plugin configuration."""
        return self._config

    @property
    def execution_phase(self) -> str | None:
        """Get execution phase."""
        return self._execution_phase


class PluginValidator(Validator):
    """Comprehensive validator for plugin contract compliance."""

    def __init__(self, strict: bool = True) -> None:
        super().__init__(
            name="plugin_validator",
            description="Validates plugin contract implementation and security requirements",
            strict=strict,
        )

    async def __call__(
        self,
        target: Any,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Validate plugin implementation against L.A.U.R.E.N contracts.

        :param target: Plugin instance to validate
        :param context: Optional execution context
        :return: True if plugin meets all requirements
        """
        if not (hasattr(target, "id") and target.id):
            return False
        if not (hasattr(target, "name") and target.name):
            return False
        if not (hasattr(target, "version") and target.version):
            return False
        # Validate lifecycle methods exist
        methods = ["execute", "process", "cleanup"]
        for method in methods:
            if not hasattr(target, method) or not callable(
                getattr(target, method)
            ):
                return False
        # In strict mode, validate additional security requirements
        if self.strict:
            # Check for proper component inheritance
            if not isinstance(target, Component):
                return False
            # In strict mode, config should be a dict if present
            if hasattr(target, "config") and not isinstance(
                target.config, dict
            ):
                return False
        return True


class PluginInspector(Inspector):
    """Inspector for comprehensive plugin lifecycle monitoring."""

    __slots__ = ("_manager",)

    def __init__(self, manager: "PluginManager") -> None:
        super().__init__(
            name="plugin_inspector",
            events={
                "plugin_loaded",
                "plugin_activated",
                "plugin_deactivated",
                "plugin_error",
                "plugin_execution_started",
                "plugin_execution_completed",
            },
        )
        self._manager = manager

    @property
    def manager(self) -> "PluginManager":
        """Get plugin manager."""
        return self._manager

    async def __call__(
        self,
        event: str,
        subject: Any,
        data: dict[str, Any],
    ) -> None:
        """React to plugin lifecycle events with comprehensive logging.

        :param event: Name of the event
        :param subject: Plugin that triggered the event
        :param data: Additional event information
        """
        # Get application logger if available
        app = getattr(self.manager, "_app", None)
        logger = getattr(app, "_logger", None) if app else None
        if logger:
            logger.info(
                f"Plugin event: {event}",
                extra={
                    "id": getattr(subject, "id", "unknown"),
                    "data": data,
                    "timestamp": time.time(),
                },
            )
        # Record in plugin manager's audit trail
        self.manager.audit.record_event(
            event,
            component=f"plugin:{getattr(subject, 'id', 'unknown')}",
            **data,
        )


class PluginSecurityPolicy(Policy):
    """Comprehensive security policy for plugin operations."""

    def __init__(
        self,
        strict: bool = True,
    ) -> None:
        operations = {
            "discover",
            "load",
            "initialise",
            "activate",
            "execute",
            "deactivate",
            "cleanup",
            "configure",
        }

        super().__init__(
            name="plugin_security",
            operations=operations,
            level="strict" if strict else "permissive",
        )

    async def __call__(
        self,
        subject: t.Any,
        operation: str | None = None,
        *,
        context: t.Dict[str, t.Any] | None = None,
    ) -> bool:
        """Evaluate plugin operation permissions with security checks.

        :param subject: Plugin requesting the operation
        :param operation: Operation being attempted
        :param context: Optional execution context
        :return: True if operation is permitted
        """
        if not operation:
            return True
        # Check if operation is in allowed set
        if operation not in self.operations:
            return False
        # Additional security checks for critical operations
        if operation in {"load", "initialise"} and hasattr(subject, "id"):
            # Validate plugin ID format (basic security check)
            id = getattr(subject, "id", "")
            if not id or not isinstance(id, str) or len(id) < 3:
                return False
        # In strict mode, require proper component inheritance
        if self.level == "strict":
            if not isinstance(subject, Component):
                return False
        return True


class Plugin(Component):
    """Abstract base class for L.A.U.R.E.N plugins with enterprise capabilities.

    This class provides the foundation for building secure, observable plugins
    with comprehensive lifecycle management, security enforcement, and detailed
    audit trails. Every plugin inherits L.A.U.R.E.N's core capabilities while
    maintaining clear separation between framework infrastructure and plugin logic.

    The plugin architecture follows L.A.U.R.E.N principles:
    - Security-first design with mandatory validation
    - Comprehensive observability and metrics collection
    - Immutable configuration with controlled state management
    - Clear contracts with explicit interfaces
    - Enterprise-grade error handling and recovery
    """

    __slots__ = (
        "id",
        "name",
        "version",
        "description",
        "config",
        "type",
        "_plugin_context",
        "_execution_count",
    )

    def __init__(
        self,
        name: str,
        version: str,
        description: str | None = None,
        config: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialise L.A.U.R.E.N plugin with comprehensive framework integration.

        :param _id: Unique identifier for the plugin
        :param _name: Human-readable plugin name
        :param version: Plugin version string
        :param description: Optional plugin description
        :param config: Plugin-specific configuration
        :param kwargs: Additional component configuration
        """
        # Set plugin-specific attributes first
        self.id = str(uuid4())[:8]  # Generate a short unique ID
        self.name = name
        self.version = version
        self.description = description or f"{name} Plugin"
        self.config = config or {}
        self._execution_count = 0
        self.type = kwargs.pop("type", "generic")
        # Plugin context
        self._plugin_context = PluginContext(
            name=f"plugin_{self.id}",
            type_="plugin_execution",
            version=version,
            config=self.config,
            execution_phase="initialised",
        )
        # Initialise component with L.A.U.R.E.N framework
        super().__init__(
            name=name,
            type_="plugin",
            config=config or {},
            validators=[PluginValidator()],
            policies=[PluginSecurityPolicy()],
            guardian=Guardian(
                disallow_post_init=True,
                private_access=True,
                strict=False,
            ),
            **kwargs,
        )
        # Record plugin creation
        self.audit.record_event(
            "plugin_created",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
            plugin_id=self.id,
            version=version,
            description=description,
        )

    def __introspect__(self) -> t.Iterator[tuple[str, t.Any]]:
        """Show plugin-specific introspection data."""
        yield from super().__introspect__()
        yield "id", self.id
        yield "version", self.version
        yield "executions", self._execution_count

    def __eq__(self, other: object) -> bool:
        """Compare plugins based on name and version."""
        if not isinstance(other, Plugin):
            return NotImplemented
        return self.name == other.name and self.version == other.version

    def __hash__(self) -> int:
        """Hash based on name and version for set/dict usage."""
        return hash((self.name, self.version))

    async def load(self) -> None:
        """Load plugin with validation and security checks."""
        # Check security policies
        if not await self.check_policies("load"):
            raise SecurityError(
                f"Plugin {self.id} load denied by security policy"
            )
        # Validate plugin
        if not await self.validate():
            raise ValueError(f"Plugin {self.id} validation failed")
        # Update context
        object.__setattr__(
            self,
            "_plugin_context",
            self._plugin_context.extend(execution_phase="loading"),
        )
        # Notify inspectors
        await self.notify_inspectors(
            "plugin_loaded",
            {"id": self.id, "version": self.version},
        )
        self.audit.record_event(
            "plugin_loaded",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
        )

    async def activate(self) -> None:
        """Activate plugin for execution."""
        if not await self.check_policies("activate"):
            raise SecurityError(f"Plugin {self.id} activation denied")
        # Update context
        object.__setattr__(
            self,
            "_plugin_context",
            self._plugin_context.extend(execution_phase="active"),
        )
        # Call plugin-specific activation
        await self.on_activate()
        # Notify inspectors
        await self.notify_inspectors("plugin_activated", {"id": self.id})
        self.audit.record_event(
            "plugin_activated",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
        )

    async def execute(
        self,
        data: t.Any,
        *,
        context: dict[str, t.Any] | None = None,
    ) -> t.Any:
        """Execute plugin with comprehensive monitoring and error handling.

        :param data: Data to process
        :param context: Optional execution context
        :return: Processed output
        """
        if not await self.check_policies("execute"):
            raise SecurityError(f"Plugin {self.id} execution denied")
        started_at = time.time()
        self._execution_count += 1
        try:
            # Update execution context
            context = self._plugin_context.extend(
                execution_phase="executing",
                execution_count=self._execution_count,
                input_type=type(data).__name__,
            )
            object.__setattr__(self, "_plugin_context", context)
            # Notify execution started
            await self.notify_inspectors(
                "plugin_execution_started",
                {
                    "id": self.id,
                    "execution_count": self._execution_count,
                    "input_type": type(data).__name__,
                },
            )
            # Execute plugin logic
            result = await self.process(data, context=context)
            # Record successful execution
            duration = time.time() - started_at
            self.metrics.record_operation(duration, success=True)
            # Notify execution completed
            await self.notify_inspectors(
                "plugin_execution_completed",
                {
                    "id": self.id,
                    "execution_count": self._execution_count,
                    "duration": duration,
                    "success": True,
                },
            )
            self.audit.record_event(
                "plugin_executed",
                component=self.name,
                category=EventCategory.OPERATION,
                severity=EventSeverity.INFO,
                duration=duration,
                success=True,
            )
            return result
        except Exception as error:
            duration = time.time() - started_at
            self.metrics.record_operation(duration, success=False)
            # Notify error occurred
            await self.notify_inspectors(
                "plugin_error",
                {
                    "id": self.id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "duration": duration,
                },
            )
            self.audit.record_event(
                "plugin_error",
                component=self.name,
                category=EventCategory.ERROR,
                severity=EventSeverity.ERROR,
                error_type=type(error).__name__,
                error_message=str(error),
                duration=duration,
            )
            raise

    @abstractmethod
    async def process(
        self,
        data: t.Any,
        *,
        context: dict[str, t.Any] | None = None,
    ) -> t.Any:
        """Process input data (to be implemented by subclasses).

        :param data: Data to process
        :param context: Optional execution context
        :return: Processed output
        """
        raise NotImplementedError("Subclasses must implement process() method")

    async def deactivate(self) -> None:
        """Deactivate plugin."""
        if not await self.check_policies("deactivate"):
            raise SecurityError(f"Plugin {self.id} deactivation denied")
        # Update context
        object.__setattr__(
            self,
            "_plugin_context",
            self._plugin_context.extend(execution_phase="deactivating"),
        )
        # Call plugin-specific deactivation
        await self.on_deactivate()
        # Notify inspectors
        await self.notify_inspectors("plugin_deactivated", {"id": self.id})
        self.audit.record_event(
            "plugin_deactivated",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
        )

    # Lifecycle hooks for subclasses to override
    async def on_activate(self) -> None:
        """Called when plugin is activated (override in subclasses)."""
        pass

    async def on_deactivate(self) -> None:
        """Called when plugin is deactivated (override in subclasses)."""
        pass

    @property
    def context(self) -> PluginContext:
        """Get current plugin execution context."""
        return self._plugin_context


class PluginManager(Component):
    """Enterprise plugin management with L.A.U.R.E.N architecture.

    This manager provides comprehensive plugin lifecycle management with
    security enforcement, observability, and performance monitoring.
    It orchestrates plugin discovery, loading, execution, and cleanup
    whilst maintaining enterprise-grade audit trails and security boundaries.
    """

    __slots__ = (
        "_app",
        "_config",
        "_plugins",
        "_active_plugins",
        "_discovery_paths",
    )

    def __init__(self, application: "Application") -> None:
        """Initialise plugin manager with application context.

        :param application: Parent application instance
        """
        super().__init__(
            name="PluginManager",
            type_="infrastructure",
            config={"auto_discovery": True, "strict_validation": True},
            inspectors=[PluginInspector(self)],
            policies=[PluginSecurityPolicy(strict=True)],
        )
        self._app = application
        self._plugins: dict[str, Plugin] = {}
        self._active_plugins: set[str] = set()
        self._config = getattr(application.config, "plugins", {})
        # Set up discovery paths
        self._discovery_paths = [lauren.plugins.__path__[0]]
        # Add custom paths from environment variable
        paths = os.environ.get("LAUREN_PLUGIN_PATH", "")
        if paths:
            for path in paths.split(os.pathsep):
                if path and os.path.exists(path):
                    self._discovery_paths.append(path)
        # Add paths from configuration
        paths = getattr(application.config, "plugin_paths", [])
        self._discovery_paths.extend(paths)

    def __introspect__(self) -> t.Iterator[tuple[str, t.Any]]:
        """Show plugin manager state."""
        yield from super().__introspect__()
        yield "total_plugins", len(self._plugins)
        yield "active_plugins", len(self._active_plugins)

    async def initialise(self) -> None:
        """Initialise plugin system with discovery and loading."""
        await self.discover_plugins()
        await self.load_plugins()

    async def discover_plugins(self) -> None:
        """Discover available plugins from multiple sources."""
        if not await self.check_policies("discover"):
            raise SecurityError("Plugin discovery denied by security policy")
        discovered = 0
        # 1. Discover from configured paths (built-in and custom directories)
        discovered += await self._discover_from_paths()
        # 2. Discover from entry points
        discovered += await self._discover_from_entry_points()
        await self.notify_inspectors(
            "plugins_discovered",
            {"count": discovered, "paths": self._discovery_paths},
        )

    async def _discover_from_paths(self) -> int:
        """Discover plugins from filesystem paths with depth protection."""
        discovered = 0
        for path in self._discovery_paths:
            if os.path.exists(path):
                discovered += await self._scan_directory(path, 0)
        return discovered

    async def _scan_directory(self, path: str, current_depth: int) -> int:
        """Recursively scan directory with depth limit protection."""
        if current_depth >= _PLUGIN_SCAN_DEPTH:
            self.audit.record_event(
                "plugin_scan_depth_exceeded",
                component=self.name,
                category=EventCategory.PERFORMANCE,
                severity=EventSeverity.WARNING,
                path=path,
                max_depth=_PLUGIN_SCAN_DEPTH,
                current_depth=current_depth,
            )
            return 0
        discovered = 0
        try:
            for _, name, ispkg in pkgutil.iter_modules([path]):
                if ispkg:
                    discovered += 1
                    self.audit.record_event(
                        "plugin_discovered",
                        component=self.name,
                        category=EventCategory.OPERATION,
                        severity=EventSeverity.INFO,
                        plugin_name=name,
                        discovery_path=path,
                        plugin_source="filesystem",
                        scan_depth=current_depth,
                    )
                    # Optionally scan subdirectories (if needed)
                    sub_path = os.path.join(path, name)
                    if (
                        os.path.isdir(sub_path)
                        and current_depth < _PLUGIN_SCAN_DEPTH - 1
                    ):
                        discovered += await self._scan_directory(
                            sub_path, current_depth + 1
                        )
        except Exception as error:
            self.audit.record_event(
                "plugin_discovery_scan_error",
                component=self.name,
                path=path,
                error=str(error),
                scan_depth=current_depth,
            )
        return discovered

    async def _discover_from_entry_points(self) -> int:
        """Discover plugins from setuptools entry points."""
        discovered = 0
        if pkgs is None:
            return discovered
        try:
            # Look for plugins in the 'lauren.plugins' entry point group
            for entry_point in pkgs.iter_entry_points("lauren.plugins"):
                discovered += 1
                self.audit.record_event(
                    "plugin_discovered",
                    component=self.name,
                    category=EventCategory.OPERATION,
                    severity=EventSeverity.INFO,
                    plugin_name=entry_point.name,
                    plugin_class=f"{entry_point.module_name}:{entry_point.attrs[0]}",
                    plugin_source="entry_point",
                )
        except Exception as error:
            self.audit.record_event(
                "plugin_discovery_error",
                component=self.name,
                error=str(error),
                plugin_source="entry_point",
            )
        return discovered

    async def load_plugins(self) -> None:
        """Load discovered plugins with validation and resource monitoring."""
        if not await self.check_policies("load"):
            raise SecurityError("Plugin loading denied by security policy")
        loaded = 0
        initial_usage = self._check_memory_usage()
        # 1. Load plugins from filesystem paths
        loaded += await self._load_from_paths()
        # 2. Load plugins from entry points
        loaded += await self._load_from_entry_points()
        # Check final memory state
        memory_usage = self._check_memory_usage()
        self.audit.record_event(
            "plugins_loaded",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
            loaded_count=loaded,
            total_plugins=len(self._plugins),
            initial_memory_usage=initial_usage.get("total_plugin_memory", 0),
            final_memory_usage=memory_usage.get("total_plugin_memory", 0),
            memory_within_limits=memory_usage.get("within_limits", True),
        )

    async def _load_from_paths(self) -> int:
        """Load plugins from filesystem paths with timeout protection."""
        loaded = 0
        for path in self._discovery_paths:
            if not os.path.exists(path):
                continue
            for finder, name, ispkg in pkgutil.iter_modules([path]):
                if not ispkg:
                    continue
                try:
                    # Load plugin with timeout protection
                    loaded += await asyncio.wait_for(
                        self._load_single_plugin_from_path(finder, name, path),
                        timeout=_PLUGIN_LOAD_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self.audit.record_event(
                        "plugin_load_timeout",
                        component=self.name,
                        module_name=name,
                        timeout=_PLUGIN_LOAD_TIMEOUT,
                        plugin_source="filesystem",
                        path=path,
                    )
                except Exception as error:
                    self.audit.record_event(
                        "module_load_error",
                        component=self.name,
                        module_name=name,
                        error=str(error),
                        plugin_source="filesystem",
                        path=path,
                    )
        return loaded

    async def _load_single_plugin_from_path(
        self,
        finder: t.Any,
        name: str,
        path: str,
    ) -> int:
        """Load a single plugin from filesystem path."""
        loaded = 0
        _name = name
        if path != lauren.plugins.__path__[0]:
            # For custom paths, create unique module names
            _name = f"lauren_custom_plugins.{name}"
        spec = finder.find_spec(name)
        if spec is None:
            return 0
        module = importlib.util.module_from_spec(spec)
        if module is None:
            return 0
        # Add to sys.modules with our custom name
        sys.modules[_name] = module
        spec.loader.exec_module(module)
        # Look for Plugin classes in the module
        plugins = self._find_plugin_classes(module)
        for _plugin in plugins:
            try:
                # Try new-style instantiation first (constructor-based)
                try:
                    # For new plugins that require name and version in constructor
                    name = (
                        _plugin.__name__.lower()
                        .replace("plugin", "")
                        .replace("lauren", "")
                        .strip("_")
                    )
                    plugin = _plugin(
                        name=name,
                        version=getattr(_plugin, "__version__", "1.0.0"),
                    )
                except TypeError:
                    plugin = _plugin()
                # Validate the plugin
                if await self._validate_plugin(plugin):
                    self._plugins[plugin.name] = plugin
                    loaded += 1
                    self.audit.record_event(
                        "plugin_loaded",
                        component=self.name,
                        plugin_name=plugin.name,
                        plugin_id=plugin.id,
                        plugin_version=plugin.version,
                        plugin_source="filesystem",
                        path=path,
                    )
            except Exception as error:
                self.audit.record_event(
                    "plugin_load_error",
                    component=self.name,
                    _plugin=_plugin.__name__,
                    error=str(error),
                    plugin_source="filesystem",
                    path=path,
                )

        return loaded

    async def _load_from_entry_points(self) -> int:
        """Load plugins from setuptools entry points with timeout protection."""
        loaded = 0
        if pkgs is None:
            return loaded
        try:
            for entry_point in pkgs.iter_entry_points("lauren.plugins"):
                try:
                    # Load plugin with timeout protection
                    loaded += await asyncio.wait_for(
                        self._load_single_plugin_from_entry_point(entry_point),
                        timeout=_PLUGIN_LOAD_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self.audit.record_event(
                        "plugin_load_timeout",
                        component=self.name,
                        plugin_name=entry_point.name,
                        timeout=_PLUGIN_LOAD_TIMEOUT,
                        plugin_source="entry_point",
                    )
                except Exception as error:
                    self.audit.record_event(
                        "plugin_load_error",
                        component=self.name,
                        plugin_name=entry_point.name,
                        error=str(error),
                        plugin_source="entry_point",
                    )
        except Exception as error:
            self.audit.record_event(
                "entry_point_discovery_error",
                component=self.name,
                error=str(error),
                plugin_source="entry_point",
            )
        return loaded

    async def _load_single_plugin_from_entry_point(
        self, entry_point: t.Any
    ) -> int:
        """Load a single plugin from entry point."""
        # Load the plugin class from entry point
        plugin = entry_point.load()
        # Verify it's a Plugin subclass
        if not (isinstance(plugin, type) and issubclass(plugin, Plugin)):
            self.audit.record_event(
                "plugin_load_error",
                component=self.name,
                plugin_name=entry_point.name,
                error="Entry point does not refer to a Plugin subclass",
                plugin_source="entry_point",
            )
            return 0

        # Instantiate the plugin (entry point plugins handle their own initialization)
        _plugin = plugin()
        # Validate the _plugin
        if await self._validate_plugin(_plugin):
            self._plugins[_plugin.name] = _plugin
            self.audit.record_event(
                "plugin_loaded",
                component=self.name,
                plugin_name=_plugin.name,
                plugin_id=_plugin.id,
                plugin_version=_plugin.version,
                plugin_source="entry_point",
                entry_point=str(entry_point),
            )
            return 1
        return 0

    def _find_plugin_classes(self, module: t.Any) -> list[type[Plugin]]:
        """Find Plugin classes in a module."""
        plugins: list[type[Plugin]] = []
        for attrs in dir(module):
            if attrs.startswith("_"):
                continue
            attr = getattr(module, attrs)
            # Check if it's a class that inherits from Plugin
            if (
                isinstance(attr, type)
                and issubclass(attr, Plugin)
                and attr is not Plugin
            ):
                plugins.append(attr)
        return plugins

    def _check_memory_usage(self) -> dict[str, t.Any]:
        """Check current memory usage."""
        try:
            # Basic memory check using sys.getsizeof for loaded plugins
            total_plugin_memory = sum(
                sys.getsizeof(plugin) for plugin in self._plugins.values()
            )
            memory_info = {
                "total_plugin_memory": total_plugin_memory,
                "plugin_count": len(self._plugins),
                "memory_limit": _MAX_PLUGIN_MEMORY,
                "memory_usage_ratio": total_plugin_memory / _MAX_PLUGIN_MEMORY,
                "within_limits": total_plugin_memory <= _MAX_PLUGIN_MEMORY,
            }
            if not memory_info["within_limits"]:
                self.audit.record_event(
                    "plugin_memory_limit_exceeded",
                    component=self.name,
                    current_usage=total_plugin_memory,
                    memory_limit=_MAX_PLUGIN_MEMORY,
                    plugin_count=len(self._plugins),
                )
            return memory_info
        except Exception as error:
            self.audit.record_event(
                "memory_check_error",
                component=self.name,
                error=str(error),
            )
            return {"error": str(error)}

    async def _validate_plugin(self, plugin: Plugin) -> bool:
        """Validate a plugin instance."""
        try:
            # Use the plugin's own validation
            return await plugin.validate()
        except Exception as error:
            self.audit.record_event(
                "plugin_validation_error",
                component=self.name,
                plugin_name=getattr(plugin, "name", "unknown"),
                error=str(error),
            )
            return False

    async def get_plugin(self, name: str) -> Plugin | None:
        """Get plugin by name.

        :param name: Plugin nameentifier
        :return: Plugin instance or None
        """
        return self._plugins.get(name)

    async def activate_plugin(self, name: str) -> None:
        """Activate a specific plugin.

        :param name: Plugin to activate
        """
        plugin = self._plugins.get(name)
        if not plugin:
            raise ValueError(f"Plugin {name!r} not found")
        await plugin.activate()
        self._active_plugins.add(name)

    async def deactivate_plugin(self, name: str) -> None:
        """Deactivate a specific plugin.

        :param name: Plugin to deactivate
        """
        plugin = self._plugins.get(name)
        if not plugin:
            raise ValueError(f"Plugin {name!r} not found")

        await plugin.deactivate()
        self._active_plugins.discard(name)

    async def execute_plugin(
        self,
        name: str,
        data: t.Any,
        *,
        context: dict[str, t.Any] | None = None,
    ) -> t.Any:
        """Execute a specific plugin.

        :param name: Plugin to execute
        :param data: Data to process
        :param context: Optional execution context
        :return: Processed output
        """
        plugin = self._plugins.get(name)
        if not plugin:
            raise ValueError(f"Plugin {name!r} not found")
        if name not in self._active_plugins:
            raise ValueError(f"Plugin {name!r} is not active")
        return await plugin.execute(data, context=context)

    async def register_plugin(
        self, plugin: Plugin, source: str = "manual"
    ) -> None:
        """Register a plugin manually.

        :param plugin: Plugin instance to register
        :param source: Source description for auditing
        """
        if not isinstance(plugin, Plugin):
            raise TypeError(
                f"Expected Plugin instance, got {type(plugin).__name__}"
            )

        if not await self._validate_plugin(plugin):
            raise ValueError(f"Plugin {plugin.name} validation failed")

        self._plugins[plugin.name] = plugin

        self.audit.record_event(
            "plugin_registered",
            component=self.name,
            plugin_name=plugin.name,
            plugin_id=plugin.id,
            plugin_version=plugin.version,
            plugin_source=source,
        )

    def get_memory_stats(self) -> dict[str, t.Any]:
        """Get current plugin system memory statistics.

        :return: Dictionary containing memory usage information
        """
        return self._check_memory_usage()

    def list_plugins(self) -> dict[str, dict[str, t.Any]]:
        """List all available plugins with their metadata.

        :return: Dictionary mapping plugin names to metadata
        """
        return {
            name: {
                "id": plugin.id,
                "version": plugin.version,
                "description": plugin.description,
                "type": plugin.type,
                "active": name in self._active_plugins,
                "executions": plugin._execution_count,
            }
            for name, plugin in self._plugins.items()
        }

    async def cleanup(self) -> None:
        """Clean up plugin system."""
        # Deactivate all active plugins
        for plugin in list(self._active_plugins):
            try:
                await self.deactivate_plugin(plugin)
            except Exception as error:
                self.audit.record_event(
                    "plugin_cleanup_error",
                    component=self.name,
                    name=plugin,
                    error=str(error),
                )
        # Clean up plugin instances
        for plugin in self._plugins.values():
            try:
                await plugin.cleanup()
            except Exception as error:
                self.audit.record_event(
                    "plugin_cleanup_error",
                    component=self.name,
                    name=plugin.name,
                    error=str(error),
                )
        await super().cleanup()
