"""\
Logging
=======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Sunday, July 20 2025

This module provides logging utilities and configuration helpers for the
L.A.U.R.E.N framework. The logging system aims to be production-ready,
performance-aware, and observable, integrating with various logging
backends and supporting structured output.

It includes custom formatters for coloured and JSON output, automatic
extra field handling, and request context management. The logging
system is designed to be flexible and extensible, allowing for easy
integration with different logging configurations and practices.

It provides a consistent logging interface across the application,
and supports features like log rotation, syslog integration, and
log level configuration. The logging utilities are designed to work
with the standard Python logging library, enhancing it with additional
features and best practices for modern applications.
"""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import re
import sys
import time
import typing as t
from pathlib import Path

if t.TYPE_CHECKING:
    from lauren.core.config import LoggingConfig

__all__: list[str] = [
    "ColouredFormatter",
    "LaurenFormatter",
    "JSONFormatter",
    "RequestContext",
    "RequestIdFilter",
    "configure",
    "get_logger",
    "perf_logger",
]


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    This formatter outputs log records in JSON format, which is useful
    for structured logging, usually in the production environments. It
    captures all relevant fields of the log record, including timestamp,
    log level, logger name, message, module, function, line number, and
    any exception information.

    It is primarily designed to be used in production systems where logs
    are collected and processed by log management systems or
    observability platforms. The JSON format allows for easy parsing
    and analysis of log data, making it suitable for structured logging
    practices.

    :param extras: Whether to include extra fields in output, defaults
        to `True`. If set to `False`, only the standard log fields will
        be included in the output.
    """

    def __init__(self, extras: bool = True):
        """Initialise the JSON formatter instance."""
        super().__init__()
        self.extras = extras

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        This method overrides the default format method to produce a
        JSON-formatted string from the log record. It captures all
        relevant fields of the log record and formats them into a JSON
        object.

        :param record: The log record to format.
        :return: JSON-formatted log message.
        """
        payload = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if self.extras:
            for key, value in record.__dict__.items():
                if key not in payload and not key.startswith("_"):
                    payload[key] = value
        return json.dumps(payload, default=str)


class LaurenFormatter(logging.Formatter):
    """Custom formatter that automatically includes extra fields.

    This formatter extends the standard logging formatter to
    automatically format and include extra fields in log messages. It
    provides flexible handling of extra fields, allowing them to be
    formatted as a structured string that can be inserted into the log
    format pattern.

    The formatter automatically detects extra fields (those not part of
    the standard `LogRecord` attributes) and formats them according to
    the specified pattern. This allows for consistent formatting of
    contextual information without requiring manual string construction
    in every log call.

    This formatter supports both plain text and coloured output, making
    it suitable for both console/tty and file logging scenarios.

    :param fmt: The format string for log messages. Can include
        `%(extra)s` placeholder for automatically formatted extra fields,
        defaults to `None`.
    :param datefmt: The format string for timestamps in log messages,
        defaults to `None`.
    :param extra_format: Format string for individual extra fields,
        defaults to `key=value` pairs separated by spaces.
    :param extra_separator: Separator between multiple extra fields,
        defaults to a single space.
    :param allow_empty_extra: Whether to allow the extra placeholder
        when no extra fields are present, defaults to `False`.
    :var LOG_RECORD_ATTRS: Set of standard `LogRecord` attributes.
    """

    LOG_RECORD_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "getMessage",
        "exc_info",
        "exc_text",
        "stack_info",
        "message",
        "asctime",
        "taskName",
        "qualName",
    }

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        extra_format: str = "{key}: {value}",
        extra_separator: str = " ",
        allow_empty_extra: bool = False,
    ) -> None:
        """Initialise the custom formatter."""
        super().__init__(fmt, datefmt)
        self.extra = extra_format
        self.extra_separator = extra_separator
        self.allow_empty_extra = allow_empty_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with automatic extra field handling.

        This method extends the standard format method to automatically
        detect and format extra fields from the log record. It creates
        a formatted string representation of all extra fields and makes
        it available as `%(extra)s` in the format string.

        :param record: The log record to format.
        :return: Formatted log message with extra fields.
        """
        clone = logging.makeLogRecord(record.__dict__)
        extras = {}
        for key, value in record.__dict__.items():
            if key not in self.LOG_RECORD_ATTRS and not key.startswith("_"):
                extras[key] = value
        if extras or self.allow_empty_extra:
            if extras:
                entries = []
                for key, value in sorted(extras.items()):
                    entries.append(self.extra.format(key=key, value=value))
                extra = self.extra_separator.join(entries) + " "
            else:
                extra = ""
            clone.extra = extra
        else:
            clone.extra = ""
        return super().format(clone)


class ColouredFormatter(LaurenFormatter):
    """Custom colored formatter with fully qualified function names.

    This formatter provides logging output with fully qualified
    class/function names, consistent spacing, and professional colour
    coding. In a way, it mimics the logging style used in Spring Boot
    applications, providing a clean and structured output that is easy
    to read and understand.

    It supports both coloured and plain text output, making it suitable
    for both development and production environments. The formatter
    automatically generates fully qualified names for functions and
    methods, including the module path and class name if applicable.

    :param fmt: The format string for log messages. Can include
        `%(extra)s` placeholder for automatically formatted extra fields,
        defaults to `None`.
    :param datefmt: The format string for timestamps in log messages,
        defaults to `None`.
    :param extra_format: Format string for individual extra fields,
        defaults to `key=value` pairs separated by spaces.
    :param extra_separator: Separator between multiple extra fields,
        defaults to a single space.
    :param allow_empty_extra: Whether to include the extra placeholder
        when no extra fields are present, defaults to `False`.
    :var COLORS: Dictionary mapping log levels to ANSI colour codes.
    """

    COLORS = {
        "DEBUG": "\x1b[38;5;14m",
        "INFO": "\x1b[38;5;41m",
        "WARNING": "\x1b[38;5;215m",
        "ERROR": "\x1b[38;5;204m",
        "CRITICAL": "\x1b[38;5;197m",
        "QUALNAME": "\x1b[38;5;140m",
        "RESET": "\x1b[0m",
    }

    def make_qualname(self, record: logging.LogRecord) -> str:
        """Generate fully qualified function/method/class name.

        This method constructs a complete path to the function or method
        that generated the log entry, including the module path and
        class name if applicable.

        :param record: The log record containing function information.
        :return: Fully qualified name string.
        """
        parts = []
        if hasattr(record, "pathname") and record.pathname:
            namespace = self.find_module(record.pathname)
            if namespace:
                parts.append(namespace)
        elif hasattr(record, "module") and record.module:
            parts.append(record.module)
        if hasattr(record, "funcName") and record.funcName:
            func = record.funcName
            if hasattr(record, "pathname") and record.pathname:
                try:
                    import ast
                    from pathlib import Path

                    path = Path(record.pathname)
                    if path.exists():
                        try:
                            with open(path, encoding="utf-8") as f:
                                code = f.read()
                            tree = ast.parse(code)
                            line = record.lineno
                            klass = self.class_at_line(tree, line, func)
                            if klass:
                                if func == "__init__":
                                    parts.append(klass)
                                else:
                                    parts.append(f"{klass}.{func}")
                            else:
                                parts.append(func)
                        except (SyntaxError, UnicodeDecodeError, OSError):
                            parts.append(func)
                    else:
                        parts.append(func)
                except Exception:
                    parts.append(func)
            else:
                parts.append(func)
                parts.append(func)
        qualname = ".".join(parts) if parts else "unknown"
        return qualname

    def find_module(self, pathname: str) -> str | None:
        """Find the proper module path from a file pathname.

        This method converts a file path into a proper module path by
        detecting the package root dynamically. It works for any
        package, whether it's part of the local project, installed in a
        virtual environment, or in system site-packages.

        :param pathname: The full file path from the log record.
        :return: The extracted module path or None if extraction fails.
        """
        try:
            import sys
            from pathlib import Path

            source = Path(pathname).resolve()
            parts = source.parts
            for path in [Path(p).resolve() for p in sys.path if p]:
                try:
                    relparts = source.relative_to(path).parts
                    for index, part in enumerate(relparts[:-1]):
                        candidate = path / Path(*relparts[: index + 1])
                        if (candidate / "__init__.py").exists():
                            segments = relparts[index:]
                            if segments[-1].endswith(".py"):
                                segments = segments[:-1] + (segments[-1][:-3],)
                            return ".".join(segments)
                except ValueError:
                    continue
            components = []
            parent = source.parent
            while parent != parent.parent:
                if (parent / "__init__.py").exists():
                    components.insert(0, parent.name)
                    parent = parent.parent
                else:
                    break
            if components:
                components.append(source.stem)
                return ".".join(components)
            for index, part in enumerate(parts):
                if part in ("site-packages", "dist-packages"):
                    if index + 1 < len(parts):
                        rest = parts[index + 1 :]
                        for idx, _ in enumerate(rest[:-1]):
                            candidate = Path(*parts[: index + 1]) / Path(
                                *rest[: idx + 1]
                            )
                            if (candidate / "__init__.py").exists():
                                segments = rest[idx:]
                                if segments[-1].endswith(".py"):
                                    segments = segments[:-1] + (
                                        segments[-1][:-3],
                                    )
                                return ".".join(segments)
                        break
            for index in range(len(parts) - 1, 0, -1):
                if any(
                    (Path(*parts[:index]) / marker).exists()
                    for marker in [
                        "setup.py",
                        "pyproject.toml",
                        "setup.cfg",
                        "requirements.txt",
                    ]
                ):
                    for idx in range(index, len(parts)):
                        candidate = Path(*parts[: idx + 1])
                        if (candidate / "__init__.py").exists():
                            segments = parts[idx:]
                            if segments[-1].endswith(".py"):
                                segments = segments[:-1] + (segments[-1][:-3],)
                            return ".".join(segments)
                    break
            for index in range(len(parts) - 1, -1, -1):
                if (Path(*parts[:index]) / "__init__.py").exists():
                    if index < len(parts) - 1:
                        segments = parts[index:]
                        if segments[-1].endswith(".py"):
                            segments = segments[:-1] + (segments[-1][:-3],)
                        return ".".join(segments)
            return source.stem
        except Exception:
            return None

    def class_at_line(self, tree: t.Any, line: int, func: str) -> str | None:
        """Find the class containing a function at the given line.

        :param tree: The AST tree of the source file.
        :param line: The line number where the function is called.
        :param func: The function name to look for.
        :return: Class name if found, `None` otherwise.
        """
        import ast

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if (
                        isinstance(method, ast.FunctionDef)
                        and method.name == func
                    ):
                        if (
                            hasattr(node, "lineno")
                            and hasattr(method, "lineno")
                            and node.lineno <= line <= method.end_lineno
                        ):
                            return node.name
        return None

    def format(self, record: logging.LogRecord) -> str:
        """Format log record.

        This method creates a log message with fully qualified names,
        consistent spacing, and professional colours. Colours are only
        applied when outputting to a TTY (terminal), ensuring that log
        files remain clean and free of ANSI escape sequences.

        :param record: The log record to format.
        :return: Formatted log message, with colours only for TTY
            output.
        """
        clone = logging.makeLogRecord(record.__dict__)
        qualname = self.make_qualname(record)
        clone.qualName = qualname
        if hasattr(self, "is_tty") and self.is_tty:
            colour = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            clone.levelname = (
                f"{colour}{record.levelname:>8s}{self.COLORS['RESET']}"
            )
            clone.qualName = (
                f"{self.COLORS['QUALNAME']}{qualname}{self.COLORS['RESET']}"
            )
        else:
            clone.levelname = f"{record.levelname:>8s}"
        return super().format(clone)


def configure(config: LoggingConfig) -> None:
    """Configure logging based on provided configuration settings.

    This function sets up the logging configuration for the application,
    including console and file handlers, and log rotation. It supports
    both structured JSON logging for production environments and colored
    output for development.

    It also allows for per-component log level configuration and
    integrates with syslog if enabled. It follows best practices for
    logging in production systems, including log rotation and retention
    policies.

    :param config: Logging configuration settings.
    """
    handlers: list[logging.Handler] = []
    levels: list[str] = []
    logger = logging.getLogger()
    logger.handlers.clear()
    if config.tty.enabled:
        levels.append(getattr(logging, config.tty.level.upper()))
    if config.file.enabled:
        levels.append(getattr(logging, config.file.level.upper()))
    if config.syslog.enabled:
        levels.append(getattr(logging, config.level.upper()))
    logger.setLevel(
        min(levels) if levels else getattr(logging, config.level.upper())
    )
    if config.tty.enabled:
        tty = logging.StreamHandler(sys.stdout)
        tty.setLevel(getattr(logging, config.tty.level.upper()))
        if config.as_json:
            formatter = JSONFormatter()
        else:
            formatter = ColouredFormatter(
                fmt=config.tty.fmt,
                datefmt=config.datefmt,
                extra_format="[{key}: {value}]",
                extra_separator=" ",
            )
            formatter.is_tty = True
        tty.setFormatter(formatter)
        handlers.append(tty)
    if config.file.enabled:
        file = Path(config.file.path)
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            filename=config.file.path,
            maxBytes=dehumanise(config.file.max_size),
            backupCount=config.file.backups,
            encoding=config.file.encoding,
        )
        handler.setLevel(getattr(logging, config.file.level.upper()))
        if config.as_json:
            formatter = JSONFormatter()
        else:
            formatter = ColouredFormatter(
                fmt=config.file.fmt,
                datefmt=config.datefmt,
                extra_format="[{key}: {value}]",
                extra_separator=" ",
            )
            formatter.is_tty = False
        handler.setFormatter(formatter)
        handlers.append(handler)
    if config.syslog.enabled:
        try:
            handler = logging.handlers.SysLogHandler(
                address=config.syslog.address,
                facility=getattr(
                    logging.handlers.SysLogHandler,
                    f"LOG_{config.syslog.facility.upper()}",
                ),
            )
            handler.setLevel(getattr(logging, config.level.upper()))
            formatter = logging.Formatter("lauren: %(levelname)s %(message)s")
            handler.setFormatter(formatter)
            handlers.append(handler)
        except Exception as error:
            print(
                f"Warning: Could not configure syslog handler: {error}",
                file=sys.stderr,
            )
    for handler in handlers:
        logger.addHandler(handler)
    for component, level in config.component_levels.items():
        logging.getLogger(component).setLevel(getattr(logging, level.upper()))
    if not config.development.debug_requests:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    if not config.development.debug_sql:
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def dehumanise(size: str) -> int:
    """Parse size string to bytes.

    This function converts a human-readable size string
    (like `10MB`, `1GB`, etc.) into an integer representing the size in
    bytes. The function handles both uppercase and lowercase unit
    specifications and allows for optional whitespace between the number
    and the unit.

    :param size: Size string like `10MB`, `1GB`, etc.
    :return: Size in bytes.
    """
    size = size.upper().strip()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    matched = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$", size)
    if not matched:
        raise ValueError(f"Invalid size format: {size}")
    value, unit = matched.groups()
    return int(float(value) * multipliers.get(unit or "B", 1))


def get_logger(logger_name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    This function retrieves a logger instance with the given name. If a
    logger with that name does not exist, it will create one with the
    default logging configuration.

    :param logger_name: Logger name.
    :return: Logger instance.
    """
    return logging.getLogger(logger_name)


def perf_logger(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """Decorator to log function execution time.

    This decorator wraps a function and logs its execution time. It
    captures the start time before the function call and calculates the
    elapsed time after the function completes. If an exception occurs,
    it logs the error along with the execution time.

    It is useful for performance monitoring and debugging, allowing
    developers to track how long specific functions take to execute and
    identify potential bottlenecks in the code.

    It also integrates with the application's logging system, ensuring
    that performance metrics are captured in a structured and consistent
    manner.

    :param func: Function to wrap.
    :return: Wrapped function with performance logging.
    """

    @functools.wraps(func)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        """Wrapper function to log execution time."""
        logger = get_logger(func.__module__)
        started = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - started
            logger.debug(
                f"Function: {func.__qualname__!r} completed in "
                f"{elapsed:.4f}s",
                extra={
                    "function": func.__qualname__,
                    "func_module": func.__module__,
                    "elapsed": elapsed,
                },
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error(
                f"Function {func.__name__!r} failed after "
                f"{elapsed:.4f}s: {exc}",
                extra={
                    "function": func.__name__,
                    "func_module": func.__module__,
                    "error": str(exc),
                    "elapsed": elapsed,
                },
                exc_info=True,
            )
            raise

    return wrapper


class RequestContext:
    """Context manager for tracking request IDs across log messages.

    This class provides a context manager that allows you to set a
    unique request ID for the duration of a request. It is useful for
    tracking requests across different parts of the application,
    especially in distributed systems or microservices architectures.

    It allows you to set a request ID that can be accessed globally
    within the application. This is particularly useful for correlating
    logs from different services or components that handle the same
    request.

    The request ID can be set at the start of a request and will be
    automatically cleared when the request is completed. This ensures
    that logs generated during the request are associated with the
    correct request ID, making it easier to trace the flow of a request
    through the system.

    Example::

        .. code-block:: python

            with RequestContext("unique-request-id") as ctx:
                # Perform operations within the request context
                logger.info("This log will include the request ID")

    :param id: Unique request identifier.
    :var current: Class variable to store the current request ID.
        This is used to track the request ID across different parts of
        the application and is set when entering the context manager.
    """

    current: str | None = None

    def __init__(self, id: str):
        """Initialise a request context instance with an ID."""
        self.id = id
        self.previous = None

    def __enter__(self) -> RequestContext:
        """Enter request context.

        This method sets the current request ID to the one provided
        when the context manager is entered. It also stores the previous
        request ID so that it can be restored when exiting the context.

        This allows for nested request contexts, where the outer context
        can be restored after the inner context is exited.

        :return: The current request context instance.
        """
        self.previous = RequestContext.current
        RequestContext.current = self.id
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: t.Any
    ) -> None:
        """Exit request context.

        This method restores the previous request ID when exiting the
        context manager. It ensures that the request ID is cleared after
        the request is completed, preventing any accidental leakage of
        request IDs across different requests.
        """
        RequestContext.current = self.previous

    @classmethod
    def get_current_id(cls) -> str | None:
        """Get current request ID."""
        return cls.current


class RequestIdFilter(logging.Filter):
    """Filter to add request ID to log records.

    This filter modifies log records to include the current request ID
    from the `RequestContext`. It is useful for ensuring that all log
    messages generated during a request are tagged with the same request
    ID, making it easier to trace logs related to a specific request.

    The filter checks the current request ID from the `RequestContext`
    and adds it to the log record as an `id` attribute. If no request ID
    is set, it defaults to "unknown". This allows logs to be easily
    correlated with specific requests, especially in distributed systems
    or microservices architectures.

    Example::

        .. code-block:: python

            logger = logging.getLogger("my_logger")
            logger.addFilter(RequestIdFilter())
            logger.info("This log will include the request ID")
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request ID to log record.

        This method adds the current request ID to the log record. If no
        request ID is set, it defaults to "unknown".

        :param record: Log record to modify.
        :return: `True` to keep the record.
        """
        record.id = RequestContext.getcurrent() or "unknown"
        return True
