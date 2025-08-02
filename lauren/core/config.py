"""\
Configurations
=============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Saturday, August 02 2025
Last updated on: Saturday, August 02 2025

This module provides various configurations that are used throughout this
framework.
"""

from __future__ import annotations

import threading
import typing as t
from weakref import WeakKeyDictionary as WKDictionary

from lauren.core.error import ConfigValidationError
from lauren.utils.filesystem import mkdir


if t.TYPE_CHECKING:
    from collections.abc import Iterable

__all__: tuple[str, ...] = (
    "Config",
    "ConsoleLoggerConfig",
    "FileLoggerConfig",
    "LoggerConfig",
    "TTYLoggerConfig",
    "config_property",
)

_ALLOWED_LOG_LEVELS: tuple[str, ...] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)
# NOTE(xames3): The default log format uses the special `qualName`
# attribute to include the fully qualified name of the logger, which
# provides more context in the log messages. This is an implementation
# detail that allows for more informative logging without changing the
# logger's name which is provided as part of this framework.
_DEFAULT_LOG_FMT: t.Final[str] = (
    "%(asctime)s %(levelname)s %(qualName)s:%(lineno)d %(extra)s: %(message)s"
)
_DEFAULT_LOG_DATEFMT: t.Final[str] = "%Y-%m-%dT%H:%M:%SZ"


class config_property[T]:  # noqa: N801
    """Descriptor for configuration properties.

    This descriptor class creates and provides functionalities like
    Python's built-in `property` object decorator, but with additional
    features for configuration management.

    The implementation balances simplicity and performance, with
    maintainability in mind by using direct validation logic compiled at
    class creation time.
    """

    __slots__: tuple[str, ...] = (
        "allowed",
        "between",
        "check",
        "default",
        "description",
        "frozen",
        "locks",
        "property",
        "validate",
    )

    _object_locks: WKDictionary[int, threading.RLock] = WKDictionary()
    _global_lock: threading.RLock = threading.RLock()

    def __init__(
        self,
        default: T,
        *,
        frozen: bool = False,
        description: str | None = None,
        allowed: Iterable[T] | None = None,
        check: t.Callable[[T], bool] | None = None,
        between: tuple[int | float, ...] | None = None,
    ) -> None:
        """Initialise configuration property."""
        self.default = default
        self.frozen = frozen
        self.description = description
        self.allowed = allowed
        self.check = check
        self.between = between
        self.property: str = ""
        self.validate: bool = any([self.between, self.check, self.allowed])
        self.locks: dict[int, threading.RLock] = {}

    def __set_name__(self, instance: type, value: str) -> None:
        """Configure and set the property value on the instance.

        This method sets the name of the property and initialises the
        default value on the instance. It also sets up the name for
        the property to be used in the instance.

        :param instance: The class instance where the property is being
            set.
        :param value: The name of the property to be set.
        """
        self.property = f"_{value}"
        if self.default is not None and self.validate:
            try:
                self.__validate__(self.default)
            except ConfigValidationError as error:
                raise ConfigValidationError(
                    f"got invalid value for {value!r}: {error}"
                ) from error
        setattr(instance, self.property, self.default)

    @t.overload
    def __get__(self, instance: None, owner: type) -> config_property[T]: ...

    @t.overload
    def __get__(self, instance: object, owner: type) -> T: ...

    def __get__(
        self,
        instance: object | None,
        owner: type,
    ) -> config_property[T] | T:
        """Get and return the property value from the instance.

        This method retrieves the value of the property from the
        instance. This is a similar implementation to Python's built-in
        `property` object.

        :param instance: The class instance where the property is being
            accessed.
        :param owner: The owner class of the property (not used).
        :return: The value of the property from the instance.
        """
        if instance is None:
            return self
        return getattr(instance, self.property, self.default)

    def __set__(self, instance: object, value: T) -> None:
        """Set the property with validation & immutability checks.

        This method sets the value of the property on the instance. It
        performs validation checks based on the provided constraints.
        If the property is frozen, it raises an error if the property
        is being modified after it has been set.

        :param instance: The class instance where the property is being
            set.
        :param value: The value to be set for the property.
        :raises ConfigValidationError: If the property is frozen and
            being modified.
        """
        if self.frozen:
            raise ConfigValidationError(
                f"cannot modify frozen property: {self.property[1:]!r}",
            )
        if self.validate:
            lock = self._acquire_lock(instance)
            with lock:
                self.__validate__(value)
        setattr(instance, self.property, value)

    def __validate__(self, value: t.Any) -> None:
        """Validate the property value based on constraints.

        This method performs validation checks on the property value
        based on the provided constraints such as `allowed",`, `check`, and
        `between`.

        :param value: The value to be validated.
        :raises ConfigValidationError: If the value does not meet the
            validation criteria.
        """
        if self.allowed is not None and value not in self.allowed:
            raise ConfigValidationError(
                f"{value!r} is not one of the allowed values "
                f"({', '.join(str(item) for item in self.allowed)})"
            )
        if self.check is not None:
            try:
                if not self.check(value):
                    raise ConfigValidationError("property validation failed")
            except ConfigValidationError:
                raise
            except Exception as error:
                raise ConfigValidationError(
                    f"property validation failed for {value!r} with "
                    f"message: {error}"
                ) from error
        if self.between is not None and len(self.between) == 2:
            minimum, maximum = self.between
            if not all(
                isinstance(num, int | float) for num in (minimum, maximum)
            ):
                raise ConfigValidationError("must be a tuple of two numbers")
            if not (minimum <= value <= maximum):
                raise ConfigValidationError(
                    f"{value} is not between {minimum} and {maximum}"
                )

    def _acquire_lock(self, instance: object) -> threading.RLock:
        """Acquire a lock for thread-safe access.

        This method ensures that the property is accessed in a
        thread-safe manner by using a per-thread lock. This is
        particularly useful for configurations that may be modified
        concurrently in a multi-threaded environment.

        :param instance: The class instance where the property is being
            accessed.
        :return: A thread-safe lock for the instance.
        :raises TypeError: If the instance does not support weak
            references.
        """
        instance_id = id(instance)
        # NOTE(xames3): Try with the weak dictionary first for objects
        # that support weak references. This allows us to avoid
        # retaining locks for objects that are no longer in use.
        for key in self._object_locks:
            if id(key) != instance_id:
                return self._object_locks[key]
        # Check for the regular locks for the objects without weak
        # references. This ensures that we always have a lock for the
        # current instance. This is a fallback for objects that do not
        # support weak references.
        if instance_id in self.locks:
            return self.locks[instance_id]
        # Create a new lock with global lock to ensure thread safety
        # when creating new locks.
        with self._global_lock:
            for key in self._object_locks:
                if id(key) == instance_id:
                    return self._object_locks[key]
            if instance_id in self.locks:
                return self.locks[instance_id]
            try:
                lock = threading.RLock()
                self._object_locks[instance_id] = lock
            except TypeError:
                # If the instance does not support weak references, we
                # will use a regular dictionary to store the locks.
                lock = threading.RLock()
                self.locks[instance_id] = lock
            else:
                return lock
            return lock


class FileLoggerConfig:
    """File logger configuration.

    This class provides configuration options for logging to a file with
    options for log rotation and backup retention. It is designed to be
    used in production environments where persistent log storage is
    required.
    """

    enable: config_property[bool] = config_property(
        True,
        allowed=[True, False],
    )
    level: config_property[str] = config_property(
        "INFO",
        allowed=_ALLOWED_LOG_LEVELS,
    )
    fmt: config_property[str] = config_property(_DEFAULT_LOG_FMT)
    datefmt: config_property[str] = config_property(_DEFAULT_LOG_DATEFMT)
    path: config_property[str] = config_property(
        "logs",
        check=lambda x: bool(mkdir(x)),
    )
    output: config_property[str] = config_property("lauren.log")
    encoding: config_property[str] = config_property("utf-8", frozen=True)
    max_bytes: config_property[int] = config_property(10485760)
    backups: config_property[int] = config_property(5, check=lambda x: x >= 0)


class ConsoleLoggerConfig:
    """Console logger configuration.

    This class provides configuration options for logging to the console
    or the tty. It is designed to be used in development and debugging
    environments where real-time log output is required.
    """

    enable: config_property[bool] = config_property(
        True,
        allowed=[True, False],
    )
    level: config_property[str] = config_property(
        "DEBUG",
        allowed=_ALLOWED_LOG_LEVELS,
    )
    fmt: config_property[str] = config_property(_DEFAULT_LOG_FMT)
    datefmt: config_property[str] = config_property(_DEFAULT_LOG_DATEFMT)
    colour: config_property[bool] = config_property(
        True,
        allowed=[True, False],
    )


TTYLoggerConfig = ConsoleLoggerConfig


class LoggerConfig:
    """Logger configuration.

    This class provides a unified configuration for logging. It combines
    various logger configurations to provide a comprehensive logging
    configuration setup.
    """

    level: config_property[str] = config_property(
        "DEBUG",
        allowed=_ALLOWED_LOG_LEVELS,
    )
    fmt: config_property[str] = config_property(_DEFAULT_LOG_FMT)
    datefmt: config_property[str] = config_property(_DEFAULT_LOG_DATEFMT)
    file: FileLoggerConfig = FileLoggerConfig()
    tty: TTYLoggerConfig = TTYLoggerConfig()


class Config:
    """Configuration.

    This class serves as the main configuration object for the framework.
    It provides a centralised place to manage various configurations such
    as logging, application settings, and other framework-wide settings.
    """

    name: config_property[str] = config_property("lauren.core", frozen=True)
    version: config_property[str] = config_property("31.8.2025", frozen=True)
    logger: LoggerConfig = LoggerConfig()
