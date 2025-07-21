"""\
Configurations
=============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Sunday, July 20 2025

This module defines the configuration settings for the L.A.U.R.E.N
framework. These settings are designed to be extensible and can be
loaded from environment variables, allowing for flexible configuration
management.

It provides comprehensive configuration options for logging, storage,
telemetry, plugins, and other components of the framework. Each
configuration class is designed to be modular and can be customised
independently, making it easy to adapt the framework to different
deployment scenarios.

It includes settings for logging levels, formats, file paths, and
handlers, as well as storage directories, telemetry options, and plugin
discovery paths. The configuration is designed to be production-ready,
with support for structured logging, log rotation, and retention
policies. It also supports development features like SQL query logging
and HTTP request/response logging for debugging purposes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

__all__: list[str] = [
    "AsyncLoggingConfig",
    "Config",
    "DevelopmentConfig",
    "FileLoggingConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "PluginConfig",
    "RemoteLoggingConfig",
    "StorageConfig",
    "SyslogConfig",
    "TTYLoggingConfig",
    "TelemetryConfig",
]

log_format: str = (
    "%(asctime)s %(levelname)s %(qualName)s:%(lineno)d "
    "%(extra)s: %(message)s"
)


class FileLoggingConfig(BaseSettings):
    """File logging configurations.

    This class provides configuration settings for logging to a file with
    options for log rotation, backup retention, and encoding. It is
    designed to be used in production environments where persistent log
    storage is required.

    It supports features like log rotation based on file size, backup
    retention policies, and customisable log formats. This allows for
    efficient log management and ensures that logs are stored in a
    structured and accessible manner.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_FILE_`.
    """

    enabled: bool = Field(
        default=True,
        description="Enable logging to a file.",
    )
    fmt: str = Field(
        default=log_format,
        description="Output log message format.",
    )
    level: str = Field(
        default="DEBUG",
        description="Logging level for the log file.",
    )
    path: str = Field(
        default="logs/lauren.log",
        description="Path to the log file.",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding.",
    )
    max_size: str = Field(
        default="10MB",
        description="Maximum size of the log file before rotation.",
    )
    backups: int = Field(
        default=5,
        description="Number of backup log files to retain.",
        ge=0,
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_FILE_"
        case_sensitive = False


class TTYLoggingConfig(BaseSettings):
    """Console or TTY logging configuration.

    This class provides configuration settings for logging to the console
    or standard output (`stdout`). It includes options for
    enabling/disabling console logging, setting the logging level, and
    customising the log message format. It is designed to be used in
    development and debugging environments where immediate feedback is
    required.

    It supports features like coloured output for better readability and
    customisable log formats. This allows developers to quickly identify
    log messages and their severity levels, enhancing the debugging
    experience.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_TTY_`.
    """

    enabled: bool = Field(
        default=True,
        description="Enable logging to console/stdout.",
    )
    level: str = Field(
        default="INFO",
        description="Logging level for the console output.",
    )
    fmt: str = Field(
        default=log_format,
        description="Output log message format.",
    )
    colours: bool = Field(
        default=True,
        description="Enable coloured console output (if supported).",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_TTY_"
        case_sensitive = False


class SyslogConfig(BaseSettings):
    """Syslog configuration.

    This class provides configuration for logging to syslog, allowing
    integration with system logging services. It includes options for
    enabling/disabling syslog logging, setting the syslog socket
    address, and specifying the syslog facility.

    It is designed to be used in production environments where logs need
    to be sent to a central logging service or system log. This allows
    for better log management and monitoring, as syslog can aggregate
    logs from multiple sources and provide a unified view of system
    events.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_SYSLOG_`.
    """

    enabled: bool = Field(
        default=False,
        description="Enable logging to syslog.",
    )
    address: str = Field(
        default="/dev/log",
        description="Syslog socket address.",
    )
    facility: str = Field(
        default="user",
        description="Syslog facility to use.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_SYSLOG_"
        case_sensitive = False


class RemoteLoggingConfig(BaseSettings):
    """Remote log aggregation configuration.

    This class provides configuration for sending logs to a remote
    log aggregation system such as `ELK`, `Loki`, or similar services.
    It includes options for enabling/disabling remote logging,
    specifying the remote URL, and providing an API key for
    authentication.

    It is designed to be used in production environments where logs
    need to be aggregated and stored in a central location for analysis
    and monitoring. This allows for better log management, search
    capabilities, and integration with observability platforms.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_REMOTE_`.
    """

    enabled: bool = Field(
        default=False,
        description="Enable remote log handler.",
    )
    url: str = Field(
        default="",
        description="URL for remote log aggregation system.",
    )
    api_key: str = Field(
        default="",
        description="API key for remote log system authentication.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_REMOTE_"
        case_sensitive = False


class PerformanceConfig(BaseSettings):
    """Performance and observability configuration.

    This class provides configuration for performance logging, including
    timing logs for operations, request ID tracking, and `OpenTelemetry`
    trace correlation. It is designed to enhance observability and
    performance monitoring in production environments.

    It includes options for enabling performance timing logs, specifying
    the request ID header for tracking, and enabling metrics collection.
    It also supports health check endpoint logging to monitor the
    application's health status.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_PERFORMANCE_`.
    """

    enabled: bool = Field(
        default=False,
        description="Enable performance timing logs for operations.",
    )
    request_id_header: str = Field(
        default="X-Request-ID",
        description="Header name for request ID tracking.",
    )
    include_trace_correlation: bool = Field(
        default=True,
        description="Include OpenTelemetry trace correlation in logs.",
    )
    metrics_enabled: bool = Field(
        default=False,
        description="Enable logging metrics collection.",
    )
    health_check_logging: bool = Field(
        default=False,
        description="Enable health check endpoint logging.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_PERFORMANCE_"
        case_sensitive = False


class AsyncLoggingConfig(BaseSettings):
    """Asynchronous logging configuration for high-performance
    applications.

    This class provides configuration for asynchronous logging, which is
    useful in high-performance applications where log writing should not
    block the main application thread. It includes options for enabling
    asynchronous logging, setting the buffer size, flush interval, and
    sampling rate. It is designed to improve performance in high-volume
    logging scenarios by allowing logs to be processed in the background
    without blocking the main application flow.

    It supports features like log buffering, periodic flushing of logs,
    and log sampling to reduce the volume of logs while still capturing
    important events. This allows for efficient log management and
    ensures that logs are written without impacting application
    performance.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_ASYNC_`.
    """

    enabled: bool = Field(
        default=False,
        description="Enable asynchronous logging.",
    )
    buffer_size: int = Field(
        default=1000,
        description="Buffer size for async logging (number of records).",
        gt=0,
    )
    flush_interval: float = Field(
        default=5.0,
        description="Flush interval for async logging (seconds).",
        gt=0.0,
    )
    sampling_rate: float = Field(
        default=1.0,
        description="Log sampling rate (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_ASYNC_"
        case_sensitive = False


class DevelopmentConfig(BaseSettings):
    """Development and debugging configuration.

    This class provides configuration for development and debugging
    environments. It includes options for enabling SQL query logging,
    HTTP request/response logging, and other debugging features. It is
    designed to assist developers in debugging and testing the
    application during development, providing detailed logs for SQL
    queries and HTTP interactions.

    It supports features like SQL query logging to track database
    interactions, HTTP request/response logging to monitor API calls,
    and other debugging options to enhance the development experience.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_DEV_`.
    """

    debug_sql: bool = Field(
        default=False,
        description="Enable SQL query logging (development only).",
    )
    debug_requests: bool = Field(
        default=False,
        description="Enable HTTP request/response logging.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_DEV_"
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """Logging configuration.

    This class provides extensive logging configuration options suitable
    for both development and production environments. It includes
    settings for file logging, console output, syslog integration, remote
    log aggregation, performance logging, and asynchronous logging. Each
    component can be configured independently, allowing for a flexible and
    powerful logging system.

    The logging system is designed to be production-ready, with support
    for structured logging, log rotation, and retention policies. It
    also supports development features like SQL query logging and HTTP
    request/response logging for debugging purposes. The configuration is
    designed to be extensible and can be loaded from environment
    variables, making it easy to adapt to different deployment
    scenarios.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_LOGGING_`.
    """

    level: str = Field(
        default="INFO",
        description="Root logging level for the application.",
    )
    fmt: str = Field(
        default=log_format,
        description="Output log message format.",
    )
    datefmt: str = Field(
        default="%Y-%m-%dT%H:%M:%SZ",
        description="Timestamp format for log messages.",
    )
    log_retention_days: int = Field(
        default=30,
        description="Log retention period in days (for compliance).",
        gt=0,
    )
    as_json: bool = Field(
        default=False,
        description="Enable structured JSON logging.",
    )
    component_levels: dict[str, str] = Field(
        default_factory=dict,
        description="Per-component logging levels.",
    )
    file: FileLoggingConfig = Field(
        default_factory=FileLoggingConfig,
        description="File logging configuration.",
    )
    tty: TTYLoggingConfig = Field(
        default_factory=TTYLoggingConfig,
        description="Console/TTY logging configuration.",
    )
    syslog: SyslogConfig = Field(
        default_factory=SyslogConfig,
        description="Syslog configuration.",
    )
    remote: RemoteLoggingConfig = Field(
        default_factory=RemoteLoggingConfig,
        description="Remote log aggregation configuration.",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance and observability configuration.",
    )
    async_: AsyncLoggingConfig = Field(
        default_factory=AsyncLoggingConfig,
        description="Asynchronous logging configuration.",
        alias="async",
    )
    development: DevelopmentConfig = Field(
        default_factory=DevelopmentConfig,
        description="Development and debugging configuration.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_LOGGING_"
        env_nested_delimiter = "__"
        case_sensitive = False
        populate_by_name = True
        extra = "ignore"


class StorageConfig(BaseSettings):
    """Storage configuration for application data.

    This class provides configuration settings for the storage system. It
    includes options for the data directory where application data is
    stored. This is designed to be flexible and can be adapted to
    different storage backends or directories as needed.

    The storage configuration is intended to be used for persisting
    application data, such as user sessions, cached data, or other
    application-specific information. It allows for easy management of
    application data storage paths, making it simple to change the
    storage location without modifying the application code.
    """

    directory: Path = Field(
        default="data",
        description="The directory for storing application data.",
    )


class TelemetryConfig(BaseSettings):
    """Telemetry and observability configuration.

    This class provides configuration settings for telemetry and
    observability features. It includes options for enabling
    `OpenTelemetry` integration, setting the service name for telemetry
    reporting, and configuring other telemetry-related features. This is
    designed to enhance observability and performance monitoring in
    production environments.

    The telemetry configuration allows for integration with
    `OpenTelemetry` to collect and report metrics, traces, and other
    telemetry data. It supports features like service name configuration
    for trace reporting, enabling or disabling telemetry features, and
    other observability options. This allows for better monitoring of
    the application's performance and health, providing insights into
    system behavior and user interactions.
    """

    enabled: bool = Field(
        default=False, description="Enable OpenTelemetry integration."
    )
    name: str = Field(
        default="lauren",
        description="The service name for telemetry reporting.",
    )


class PluginConfig(BaseSettings):
    """Plugin discovery and management configuration.

    This class provides configuration settings for discovering and
    managing plugins. It includes options for specifying the directory
    where plugins are located, enabling or disabling plugin discovery,
    and other plugin-related features. This is designed to facilitate
    the integration of plugins into the framework, allowing for
    extensibility and customisation of the application.

    The plugin configuration allows for automatic discovery of plugins
    within a specified directory. It supports features like enabling or
    disabling plugin discovery, specifying the directory to scan for
    plugins, and other plugin management options. This allows developers
    to easily add, remove, or update plugins without modifying the core
    application code, enhancing the modularity and flexibility of the
    framework.
    """

    directory: Path = Field(
        default="lauren/plugins",
        description="The directory to scan for plugins.",
    )


class Config(BaseSettings):
    """Primary configuration object.

    This class serves as the main configuration object. It aggregates
    various configuration components, including logging, storage,
    telemetry, plugins, and external service configurations. It is
    designed to be extensible and can be loaded from environment
    variables, allowing for flexible configuration management.

    It provides a central point for configuring the application, making
    it easy to manage settings across different components. The
    configuration is designed to be comprehensive, covering all aspects
    of the application, from logging and storage to external service
    integrations.

    .. note::

        The configuration setting can be customised via environment
        variables with the prefix `LAUREN_`.
    """

    debug: bool = Field(
        default=False,
        description="Enable debug mode.",
    )
    name: str = Field(
        default="lauren",
        description="The name of the application.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configurations.",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage configurations.",
    )
    telemetry: TelemetryConfig = Field(
        default_factory=TelemetryConfig,
        description="Telemetry configurations.",
    )
    plugins: PluginConfig = Field(
        default_factory=PluginConfig,
        description="Plugin configurations.",
    )

    class Config:
        """Pydantic configuration class."""

        env_prefix = "LAUREN_"
        env_nested_delimiter = "__"
        case_sensitive = False
        extra = "ignore"
