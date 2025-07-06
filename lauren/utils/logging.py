"""\
Logging
=======

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Sunday, July 06 2025

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
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        if self.extras:
            for key, value in record.__dict__.items():
                if key not in log_data and not key.startswith("_"):
                    log_data[key] = value
        return json.dumps(log_data, default=str)


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
        record_copy = logging.makeLogRecord(record.__dict__)
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in self.LOG_RECORD_ATTRS and not key.startswith("_"):
                extra_fields[key] = value
        if extra_fields or self.allow_empty_extra:
            if extra_fields:
                formatted_extra_fields = []
                for key, value in sorted(extra_fields.items()):
                    formatted_field = self.extra.format(key=key, value=value)
                    formatted_extra_fields.append(formatted_field)
                extra = self.extra_separator.join(formatted_extra_fields) + " "
            else:
                extra = ""
            record_copy.extra = extra
        else:
            record_copy.extra = ""
        return super().format(record_copy)


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

    def generate_qualname(self, record: logging.LogRecord) -> str:
        """Generate fully qualified function/method/class name.

        This method constructs a complete path to the function or method
        that generated the log entry, including the module path and
        class name if applicable.

        :param record: The log record containing function information.
        :return: Fully qualified name string.
        """
        parts = []
        if hasattr(record, "pathname") and record.pathname:
            module_path = self.isolate_module(record.pathname)
            if module_path:
                parts.append(module_path)
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
                            with open(path, encoding="utf-8") as code_file:
                                source_code = code_file.read()
                            tree = ast.parse(source_code)
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
        qualified_name = ".".join(parts) if parts else "unknown"
        return qualified_name

    def isolate_module(self, pathname: str) -> str | None:
        """Extract the proper module path from a file pathname.

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

            source_path = Path(pathname).resolve()
            path_parts = source_path.parts
            python_search_paths = [Path(p).resolve() for p in sys.path if p]
            for python_search_path in python_search_paths:
                try:
                    relative_path = source_path.relative_to(python_search_path)
                    relative_parts = relative_path.parts
                    for index, part in enumerate(relative_parts[:-1]):
                        potential_package = python_search_path / Path(
                            *relative_parts[: index + 1]
                        )
                        if (potential_package / "__init__.py").exists():
                            module_parts = relative_parts[index:]
                            if module_parts[-1].endswith(".py"):
                                module_parts = module_parts[:-1] + (
                                    module_parts[-1][:-3],
                                )
                            return ".".join(module_parts)
                except ValueError:
                    continue
            package_parts = []
            current_path = source_path.parent
            while current_path != current_path.parent:
                if (current_path / "__init__.py").exists():
                    package_parts.insert(0, current_path.name)
                    current_path = current_path.parent
                else:
                    break
            if package_parts:
                module_name = source_path.stem
                package_parts.append(module_name)
                return ".".join(package_parts)
            for index, part in enumerate(path_parts):
                if part in ("site-packages", "dist-packages"):
                    if index + 1 < len(path_parts):
                        remaining_parts = path_parts[index + 1 :]
                        for idx, _ in enumerate(remaining_parts[:-1]):
                            potential_package = Path(
                                *path_parts[: index + 1]
                            ) / Path(*remaining_parts[: idx + 1])
                            if (potential_package / "__init__.py").exists():
                                module_parts = remaining_parts[idx:]
                                if module_parts[-1].endswith(".py"):
                                    module_parts = module_parts[:-1] + (
                                        module_parts[-1][:-3],
                                    )
                                return ".".join(module_parts)
                        break
            for index in range(len(path_parts) - 1, 0, -1):
                potential_root = Path(*path_parts[:index])
                if any(
                    (potential_root / marker).exists()
                    for marker in [
                        "setup.py",
                        "pyproject.toml",
                        "setup.cfg",
                        "requirements.txt",
                    ]
                ):
                    for idx in range(index, len(path_parts)):
                        potential_package = Path(*path_parts[: idx + 1])
                        if (potential_package / "__init__.py").exists():
                            module_parts = path_parts[idx:]
                            if module_parts[-1].endswith(".py"):
                                module_parts = module_parts[:-1] + (
                                    module_parts[-1][:-3],
                                )
                            return ".".join(module_parts)
                    break
            for index in range(len(path_parts) - 1, -1, -1):
                potential_path = Path(*path_parts[:index])
                if (potential_path / "__init__.py").exists():
                    if index < len(path_parts) - 1:
                        module_parts = path_parts[index:]
                        if module_parts[-1].endswith(".py"):
                            module_parts = module_parts[:-1] + (
                                module_parts[-1][:-3],
                            )
                        return ".".join(module_parts)
            return source_path.stem
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
        record_copy = logging.makeLogRecord(record.__dict__)
        qualified_name = self.generate_qualname(record)
        record_copy.qualName = qualified_name
        if hasattr(self, "is_tty") and self.is_tty:
            colour = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            record_copy.levelname = (
                f"{colour}{record.levelname:>8s}{self.COLORS['RESET']}"
            )
            record_copy.qualName = f"{self.COLORS['QUALNAME']}{qualified_name}{self.COLORS['RESET']}"
        else:
            record_copy.levelname = f"{record.levelname:>8s}"
        return super().format(record_copy)


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
    log_handlers: list[logging.Handler] = []
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, config.level.upper()))
    if config.tty.enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.tty.level.upper()))
        if config.as_json:
            console_formatter = JSONFormatter()
        else:
            console_formatter = ColouredFormatter(
                fmt=config.tty.fmt,
                datefmt=config.datefmt,
                extra_format="[{key}: {value}]",
                extra_separator=" ",
            )
            console_formatter.is_tty = True
        console_handler.setFormatter(console_formatter)
        log_handlers.append(console_handler)
    if config.file.enabled:
        file_path = Path(config.file.path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=config.file.path,
            maxBytes=dehumanise(config.file.max_size),
            backupCount=config.file.backups,
            encoding=config.file.encoding,
        )
        file_handler.setLevel(getattr(logging, config.file.level.upper()))
        if config.as_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = ColouredFormatter(
                fmt=config.file.fmt,
                datefmt=config.datefmt,
                extra_format="[{key}: {value}]",
                extra_separator=" ",
            )
            file_formatter.is_tty = False
        file_handler.setFormatter(file_formatter)
        log_handlers.append(file_handler)
    if config.syslog.enabled:
        try:
            syslog_handler = logging.handlers.SysLogHandler(
                address=config.syslog.address,
                facility=getattr(
                    logging.handlers.SysLogHandler,
                    f"LOG_{config.syslog.facility.upper()}",
                ),
            )
            syslog_handler.setLevel(getattr(logging, config.level.upper()))
            syslog_formatter = logging.Formatter(
                "lauren: %(levelname)s %(message)s"
            )
            syslog_handler.setFormatter(syslog_formatter)
            log_handlers.append(syslog_handler)
        except Exception as error:
            print(
                f"Warning: Could not configure syslog handler: {error}",
                file=sys.stderr,
            )
    for log_handler in log_handlers:
        root_logger.addHandler(log_handler)
    for component_name, log_level in config.component_levels.items():
        logging.getLogger(component_name).setLevel(
            getattr(logging, log_level.upper())
        )
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
    size_multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    size_match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$", size)
    if not size_match:
        raise ValueError(f"Invalid size format: {size}")
    numeric_value, size_unit = size_match.groups()
    return int(
        float(numeric_value) * size_multipliers.get(size_unit or "B", 1)
    )


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
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.debug(
                f"Function: {func.__qualname__} completed in "
                f"{execution_time:.4f}s",
                extra={
                    "function": func.__qualname__,
                    "func_module": func.__module__,
                    "execution_time": execution_time,
                },
            )
            return result
        except Exception as exc:
            execution_time = time.perf_counter() - start_time
            logger.error(
                f"Function {func.__name__} failed after "
                f"{execution_time:.4f}s: {exc}",
                extra={
                    "function": func.__name__,
                    "func_module": func.__module__,
                    "error": str(exc),
                    "execution_time": execution_time,
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

    :param request_id: Unique request identifier.
    :var _current_id: Class variable to store the current request ID.
        This is used to track the request ID across different parts of
        the application and is set when entering the context manager.
    """

    _current_id: str | None = None

    def __init__(self, request_id: str):
        """Initialise a request context instance with an ID."""
        self.request_id = request_id
        self.previous_request_id = None

    def __enter__(self) -> RequestContext:
        """Enter request context.

        This method sets the current request ID to the one provided
        when the context manager is entered. It also stores the previous
        request ID so that it can be restored when exiting the context.

        This allows for nested request contexts, where the outer context
        can be restored after the inner context is exited.

        :return: The current request context instance.
        """
        self.previous_request_id = RequestContext._current_id
        RequestContext._current_id = self.request_id
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
        RequestContext._current_id = self.previous_request_id

    @classmethod
    def get_current_id(cls) -> str | None:
        """Get current request ID."""
        return cls._current_id


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
        record.request_id = RequestContext.get_current_id() or "unknown"
        return True
