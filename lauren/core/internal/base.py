"""\
Foundational objects
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, July 22 2025
Last updated on: Tuesday, July 29 2025

This module provides the foundational building blocks for creating
observable and context-aware components within the framework. These
components form the bedrock of the framework's observability, context
management, and state tracking capabilities.
"""

from __future__ import annotations

import enum
import keyword
import time
import typing as t
from abc import ABC
from abc import abstractmethod
from uuid import uuid4

if t.TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

__all__: tuple[str] = (
    "ComponentState",
    "ExecutionContext",
    "Observable",
    "_is_valid_name",
)


def _is_valid_name(name: str, element: str = "Component") -> None:
    """Check if the provided name is appropriate.

    This function checks if the provided name is a valid Python
    identifer. This name is used for naming the components, validators,
    policies, and other framework elements.

    :param name: The name to check.
    :param element: The type of object being named, defaults
        to `Component`. Available options are `Component`, `Validator`,
        `Policy`, and `Inspector`.
    :return: None if all checks pass.
    :raises AssertionError: If name is not a string or valid identifier.
    """
    if not isinstance(name, str):
        raise AssertionError("Name must be a string")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise AssertionError(f"Invalid name for {element}")
    return None


class ComponentState(enum.StrEnum):
    """Possible component states.

    This enumeration defines the lifecycle states that a component can
    be in during its operation. Components transition through these
    states during their lifecycle, from creation to destruction.

    States represent the operational status and can be used for
    monitoring, debugging, and ensuring proper component lifecycle
    management.
    """

    CREATED = "created"
    INITIALISING = "initialising"
    INITIALISED = "initialised"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    CLEANING_UP = "cleaning_up"
    DESTROYED = "destroyed"
    ERROR = "error"


class Observable(ABC):
    """Abstract base class for objects that can be introspected.

    This class provides a foundation for creating objects that can
    expose their internal state for debugging, monitoring, and
    introspection purposes. It establishes a consistent interface
    for examining object attributes and state information.

    .. important::

        Objects that inherit from this class must implement the
        `__inspect_attrs__` method to define what attributes should be
        visible during introspection. This enables consistent debugging
        and monitoring across all framework components.

    .. note::

        The introspection interface is designed to be safe and
        non-invasive, providing read-only access to object state
        without affecting the object's operation or exposing
        sensitive information inappropriately.

    .. code-block:: python

        class Controller(Observable):
            def __init__(self, value):
                self.value = value
                self.type = "Forward Controller"

            def __inspect_attrs__(self):
                yield "value", self.value
                yield "type", self.type

        controller = Controller(42)
        attrs = dict(controller.__inspect_attrs__())
        print(f"Object state: {attrs}")
    """

    __slots__: tuple[str] = ("__weakref__",)

    @abstractmethod
    def __inspect_attrs__(self) -> Iterator[tuple[str, t.Any]]:
        """Yield object attributes for introspection.

        This method should yield tuples of attribute name and value for
        all attributes that should be visible during introspection.
        Implementations should be careful to only expose appropriate
        information and avoid sensitive data.

        :yield: An iterator yielding tuples of attribute names and their
            corresponding values for inspection.

        .. note::

            This method should be implemented to provide a consistent
            interface for object introspection across all observable
            components in the framework.
        """
        raise NotImplementedError


class ExecutionContext:
    """Execution context for tracking operation state and metadata.

    This class provides a context object that tracks execution state,
    metadata, and hierarchical relationships between operations. It
    enables comprehensive tracing and monitoring of operations within
    the framework.

    Execution contexts can be nested to represent hierarchical
    operations and provide detailed tracing information for debugging
    and monitoring purposes. Each context maintains its own metadata
    and timing information.

    :param name: Name for the context.
    :param kind: Kind of identifier for the context.
    :param metadata: Additional context metadata, defaults to `None`.
    :param parent: Parent context for hierarchical tracking, defaults
        to `None`.

    .. note::

        Execution contexts are designed to be lightweight and efficient,
        suitable for use in high-frequency operations without
        significant performance impact.

    .. code-block:: python

        def validate_email_format(email):
            return re.match(r"[^@]+@[^@]+\\.[^@]+", email) is not None

        user_authentication = ExecutionContext(
            name="UserAuthentication",
            kind="security_operation",
            metadata={
                "request_id": "request-1234567890",
                "user_email": "xa@mes3.dev",
                "ip_address": "localhost",
                "user_agent": "AppleWebKit/537.36"
            }
        )
        with user_authentication:
            email_validation = user_authentication.extend(
                operation="EmailValidation",
                validation_rules=["format", "domain", "mx_record"]
            )
            with email_validation:
                if not validate_email_format(user_email):
                    raise ValidationError("Invalid email format")
    """

    __slots__: tuple[str, ...] = (
        "_id",
        "_name",
        "_kind",
        "_metadata",
        "_parent",
        "_started",
        "_expired",
        "_children",
    )

    def __init__(
        self,
        name: str,
        kind: str,
        *,
        metadata: dict[str, t.Any] | None = None,
        parent: "ExecutionContext" | None = None,
    ) -> None:
        """Initialise a new execution context."""
        self._id = str(uuid4())[:8]
        _is_valid_name(name, "Context")
        self._name = name
        self._kind = kind
        self._metadata = metadata or {}
        self._parent = parent
        self._started: float | None = None
        self._expired: float | None = None
        self._children: list["ExecutionContext"] = []
        if parent:
            parent._children.append(self)

    def __enter__(self) -> "ExecutionContext":
        """Enter the execution context."""
        self._started = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the execution context."""
        self._expired = time.time()

    def __repr__(self) -> str:
        """Return string representation of the context."""
        return f"ExecutionContext(id={self._id}, name={self._name!r})"

    def extend(self, **metadata: t.Any) -> "ExecutionContext":
        """Create a child context with additional metadata.

        This method creates a new execution context that inherits from
        the current context, adding or overriding metadata as specified.
        This is useful for creating sub-contexts that track specific
        operations within a larger execution scope.

        :return: New execution context with extended metadata.
        """
        return ExecutionContext(
            name=f"{self._name}_extended",
            kind=self._kind,
            metadata={**self._metadata, **metadata},
            parent=self,
        )

    @property
    def id(self) -> str:
        """Get the unique identifier for this context."""
        return self._id

    @property
    def name(self) -> str:
        """Get the human-readable name of this context."""
        return self._name

    @property
    def kind(self) -> str:
        """Get the kind or type of this context."""
        return self._kind

    @property
    def metadata(self) -> dict[str, t.Any]:
        """Get a copy of the context metadata."""
        return self._metadata.copy()

    @property
    def parent(self) -> "ExecutionContext" | None:
        """Get the parent context, if any."""
        return self._parent

    @property
    def children(self) -> list["ExecutionContext"]:
        """Get a list of child contexts."""
        return self._children.copy()

    @property
    def duration(self) -> float | None:
        """Get the duration of the context execution."""
        if self._started is None:
            return None
        return (self._expired or time.time()) - self._started

    @property
    def is_active(self) -> bool:
        """Check if the context is currently active."""
        return self._started is not None and self._expired is None
