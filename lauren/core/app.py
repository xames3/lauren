from __future__ import annotations

import time
import types
import typing as t
from contextvars import ContextVar
from uuid import uuid4

from lauren.core.base import Component
from lauren.core.base import Context
from lauren.core.base import Inspector
from lauren.core.base import Policy
from lauren.core.base import StateDict
from lauren.core.base import Validator
from lauren.core.config import Config
from lauren.core.events import EventCategory
from lauren.core.events import EventSeverity
from lauren.core.exceptions import ValidationError
from lauren.core.plugin import PluginManager
from lauren.utils.logging import configure
from lauren.utils.logging import get_logger
from lauren.utils.opentelemetry import get_tracer

if t.TYPE_CHECKING:
    from opentelemetry.trace import Tracer

    from lauren.core.plugin import Plugin

__all__ = ["Application", "ApplicationContext", "ContextManager"]

_REQUEST_TIMEOUT: t.Final[float] = 30.0
_MAX_CONTEXT_DEPTH: t.Final[int] = 100
_CONTEXT_CACHE_SIZE: t.Final[int] = 1000

# Application-wide context variables
_user_id: ContextVar[str | None] = ContextVar("user_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_metadata: ContextVar[dict[str, t.Any]] = ContextVar("metadata", default={})


class ApplicationContext(Context):
    """Enhanced execution context for application-specific operations.

    This context extends the base L.A.U.R.E.N context with application-level
    fields whilst maintaining immutability and comprehensive tracing
    capabilities. It provides the foundation for request tracking,
    user session management, and distributed tracing.
    """

    __slots__ = ("_request", "_operation")

    def __init__(
        self,
        name: str,
        type_: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_id: str | None = None,
        metadata: StateDict | None = None,
        request: str | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialise application context.

        :param name: Context name
        :param type_: Type of context
        :param ctx_id: Unique context identifier
        :param user_id: User identifier
        :param session_id: Session identifier
        :param trace_id: Distributed tracing identifier
        :param parent_id: Parent context identifier
        :param metadata: Additional context metadata
        :param request: HTTP request identifier
        :param operation: Type of operation being performed
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
        self._request = request
        self._operation = operation

    def ___introspect____(self) -> t.Iterator[tuple[str, t.Any]]:
        """Show application context information."""
        yield from super().__introspect__()
        if self._request:
            yield "request", self._request
        if self._operation:
            yield "operation", self._operation

    @property
    def request(self) -> str | None:
        """Get request ID."""
        return self._request

    @property
    def operation(self) -> str | None:
        """Get operation type."""
        return self._operation


class ContextManager(Component):
    """Context management with L.A.U.R.E.N observability and security.

    This manager provides thread-safe context operations whilst maintaining
    comprehensive audit trails and supporting enterprise security policies.
    All context changes are tracked and validated through the L.A.U.R.E.N
    policy framework.
    """

    __slots__ = ("_current", "_initial")

    def __init__(self, context: ApplicationContext) -> None:
        """Initialise context manager with application context.

        :param context: Initial application context
        """
        super().__init__(
            name="ContextManager",
            type_="infrastructure",
            config={"context_validation": True},
        )
        if not isinstance(context, ApplicationContext):
            raise TypeError(
                f"Expected ApplicationContext, got {type(context).__name__}"
            )
        object.__setattr__(self, "_current", context)
        object.__setattr__(self, "_initial", context)

    def __introspect__(self) -> t.Iterator[tuple[str, t.Any]]:
        """Show context manager state."""
        yield from super().__introspect__()
        yield "current", self._current.name
        yield "has_changed", self._current != self._initial

    async def update(self, **updates: t.Any) -> ApplicationContext:
        """Update context with validation and audit logging.

        :param updates: Context field updates
        :return: New context instance
        """
        # Check policies before updating
        if not await self.check_policies("context_update"):
            raise PermissionError("Context update denied by security policy")
        # Create new context with updates
        merged = {
            **self._current.metadata,
            **updates.pop("metadata", {}),
        }
        context = ApplicationContext(
            name=updates.get("name", self._current.name),
            type_=updates.get("type_", self._current.type),
            user_id=updates.get("user_id", self._current.user_id),
            session_id=updates.get("session_id", self._current.session_id),
            trace_id=self._current.trace_id,
            parent_id=self._current.id,
            metadata=merged,
            request=updates.get("request", self._current.request),
            operation=updates.get("operation", self._current.operation),
        )
        # Notify inspectors of context change
        await self.notify_inspectors(
            "context_updated",
            {
                "old_context": dict(self._current),
                "new_context": dict(context),
                "changes": list(updates.keys()),
            },
        )
        # Update current context
        object.__setattr__(self, "_current", context)
        return context

    def get(self) -> ApplicationContext:
        """Get current context."""
        return self._current

    def reset(self) -> None:
        """Reset context to initial state."""
        object.__setattr__(self, "_current", self._initial)
        self._audit.record_event(
            "context_reset",
            component=self.name,
            category=EventCategory.OPERATION,
            severity=EventSeverity.INFO,
        )

    @property
    def current(self) -> ApplicationContext:
        """Get current context."""
        return self._current

    @property
    def initial(self) -> ApplicationContext:
        """Get initial context."""
        return self._initial


class ApplicationValidator(Validator):
    """Validator for application component integrity."""

    def __init__(self) -> None:
        super().__init__(
            name="application_validator",
            description="Validates application configuration and state",
        )

    async def __call__(
        self,
        target: t.Any,
        *,
        context: dict[str, t.Any] | None = None,
    ) -> bool:
        """Validate application configuration and setup."""
        if not hasattr(target, "_config"):
            return False
        if not hasattr(target, "_context_manager"):
            return False
        # Validate critical configuration
        config = getattr(target, "_config")
        if not config or not hasattr(config, "logging"):
            return False
        return True


class ApplicationInspector(Inspector):
    """Inspector for application lifecycle events."""

    __slots__ = ("_app",)

    def __init__(self, app: "Application") -> None:
        super().__init__(
            name="application_inspector",
            events={
                "application_started",
                "application_stopped",
                "request_processed",
                "error_occurred",
            },
        )
        self._app = app

    async def __call__(
        self,
        event: str,
        subject: t.Any,
        data: dict[str, t.Any],
    ) -> None:
        """React to application events."""
        logger = getattr(self._app, "_logger", None)
        if logger:
            logger.info(f"Application event: {event}", extra=data)

    @property
    def app(self) -> "Application":
        """Get application instance."""
        return self._app


class ApplicationSecurityPolicy(Policy):
    """Security policy for application operations."""

    def __init__(self, strict: bool = False) -> None:
        super().__init__(
            name="application_security",
            operations={"initialise", "process", "cleanup"},
            level="strict" if strict else "permissive",
        )

    async def __call__(
        self,
        subject: t.Any,
        operation: str | None = None,
        *,
        context: dict[str, t.Any] | None = None,
    ) -> bool:
        """Evaluate operation permission for application."""
        if not operation:
            return True
        if operation in self.operations:
            return True
        if self.level == "permissive":
            return True
        return False


class Application(Component):
    """Enterprise-grade application orchestrator with L.A.U.R.E.N architecture.

    This class provides the central coordination for all application operations,
    integrating configuration management, plugin orchestration, context management,
    and comprehensive observability through L.A.U.R.E.N's security-first approach.

    The application follows L.A.U.R.E.N principles:
    - Every operation is observable and auditable
    - Security policies are enforced at multiple layers
    - Configuration is immutable after initialisation
    - Component interactions are explicit and validated
    - Error handling provides detailed context for infoging
    """

    __slots__ = (
        "_name",
        "_config",
        "_logger",
        "_tracer",
        "_context_manager",
        "_plugins_manager",
    )

    def __init__(
        self,
        name: str | None = None,
        config: Config | None = None,
        **kwargs: t.Any,
    ) -> None:
        """Initialise L.A.U.R.E.N application with enterprise capabilities.

        :param config: Application configuration
        :param name: Human-readable application name
        :param kwargs: Additional component configuration
        """
        # Set application name first before calling super().__init__
        self._name = name or "Base Application"
        # Initialise component with L.A.U.R.E.N framework integration
        super().__init__(
            name=self._name,
            type_="application",
            config={"strict": True},
            validators=[ApplicationValidator()],
            inspectors=[ApplicationInspector(self)],
            policies=[ApplicationSecurityPolicy(strict=True)],
            **kwargs,
        )
        # Core application services
        self._config = config or Config()
        # Configure logging and observability
        configure(self._config.logging)
        self._logger = get_logger(__name__)
        self._logger.info(f"Initialising {self._name}")
        # Set up distributed tracing
        self._tracer = get_tracer(config=self._config)
        # Create application context
        context = ApplicationContext(
            name="app_main",
            type_="application",
            user_id=_user_id.get(None),
            session_id=_session_id.get(None),
            request=str(uuid4()),
            operation="initialisation",
            metadata=_metadata.get({}),
        )
        self._context_manager = ContextManager(context)
        # Set up plugin management
        self._plugins_manager = PluginManager(self)
        # Record application creation
        self._audit.record_event(
            "application_created",
            component=self.name,
            category=EventCategory.LIFECYCLE,
            severity=EventSeverity.INFO,
            application_type=self.type,
            config_sections=(
                list(self._config.__dict__.keys())
                if hasattr(self._config, "__dict__")
                else []
            ),
        )

    def __introspect__(self) -> t.Iterator[tuple[str, t.Any]]:
        """Show application-specific introspection data."""
        yield from super().__introspect__()
        yield "name", self._name
        yield "has_plugins", hasattr(self, "_plugins_manager")
        yield "id", self._context_manager.current.id

    async def __aenter__(self) -> "Application":
        """Async context manager entry."""
        await self.initialise()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.cleanup()

    async def initialise(self) -> None:
        """Initialise application with validation and setup."""
        # Validate configuration and components
        if not await self.validate():
            raise ValidationError("Application validation failed")
        # Check security policies
        if not await self.check_policies("initialise"):
            raise PermissionError(
                "Application initialisation denied by security policy"
            )
        # Initialise plugin system
        if hasattr(self, "_plugins_manager"):
            await self._plugins_manager.initialise()
        # Notify inspectors
        await self.notify_inspectors(
            "app_started",
            {
                "name": self._name,
                "plugin_count": getattr(
                    self._plugins_manager, "_plugin_count", 0
                ),
            },
        )
        self._logger.info(f"{self._name} initialisation complete")

    async def process(
        self,
        request: t.Any,
        context_updates: dict[str, t.Any] | None = None,
    ) -> t.Any:
        """Process a request through the application pipeline.

        :param request: Request data to process
        :param context_updates: Optional context updates for this request
        :return: Processed response
        """
        started_at = time.time()
        try:
            # Update context for this request
            if context_updates:
                await self._context_manager.update(**context_updates)
            # Check security policies
            if not await self.check_policies("process_request"):
                raise PermissionError(
                    "Request processing denied by security policy"
                )
            # Delegate to subclass implementation
            result = await self.handle(request)
            # Record successful operation
            duration = time.time() - started_at
            self._metrics.record_operation(duration, success=True)
            # Notify inspectors
            await self.notify_inspectors(
                "request_processed",
                {
                    "request_type": type(request).__name__,
                    "duration": duration,
                    "success": True,
                },
            )
            return result
        except Exception as error:
            # Record failed operation
            duration = time.time() - started_at
            self._metrics.record_operation(duration, success=False)
            # Log error with context
            self._logger.error(
                f"Request processing failed: {error}",
                extra={
                    "id": self.context.id,
                    "request_type": type(request).__name__,
                    "duration": duration,
                },
            )
            # Notify inspectors of error
            await self.notify_inspectors(
                "error_occurred",
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "id": self.context.id,
                },
            )
            raise

    async def handle(self, request: t.Any) -> t.Any:
        """Handle request processing (to be implemented by subclasses).

        :param request: Request data to process
        :return: Processed response
        """
        raise NotImplementedError("Subclasses must implement handle() method")

    async def cleanup(self) -> None:
        """Clean up application resources."""
        try:
            # Check security policies
            if not await self.check_policies("cleanup"):
                self._logger.warning(
                    "Cleanup denied by security policy, proceeding anyway"
                )
            # Clean up plugins
            if hasattr(self, "_plugins_manager"):
                await self._plugins_manager.cleanup()
            # Notify inspectors
            await self.notify_inspectors(
                "app_stopped",
                {
                    "app_name": self._name,
                    "total_operations": self._metrics.data.get(
                        "operations_count", 0
                    ),
                },
            )
            # Call parent cleanup
            await super().cleanup()
            self._logger.info(f"{self._name} cleanup complete")
        except Exception as error:
            self._logger.error(f"Cleanup error: {error}")
            raise

    @property
    def name(self) -> str:
        """Get application name."""
        return self._name

    @property
    def config(self) -> Config:
        """Get application configuration."""
        return self._config

    @property
    def logger(self) -> t.Any:
        """Get application logger."""
        return self._logger

    @property
    def tracer(self) -> t.Any:
        """Get application tracer."""
        return self._tracer

    @property
    def context_manager(self) -> ContextManager:
        """Get context manager."""
        return self._context_manager

    @property
    def plugin_manager(self) -> PluginManager:
        """Get plugin manager."""
        return self._plugins_manager

    @property
    def context(self) -> ApplicationContext:
        """Get current application context."""
        return self._context_manager.get()

    @property
    def plugins(self) -> PluginManager:
        """Get plugin manager."""
        return self._plugins_manager


# Legacy compatibility aliases
App = Application
AppContext = ApplicationContext
