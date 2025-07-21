"""\
OpenTelemetry
=============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Friday, July 04 2025
Last updated on: Monday, July 21 2025

This module provides `OpenTelemetry` integration for the L.A.U.R.E.N
framework, offering a standardised way to handle tracing and metrics.
This allows developers to monitor and trace the execution of their
applications, providing insights into performance and behaviour during
runtime.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from lauren.core.config import Config

__all__: list[str] = ["get_tracer"]


def get_tracer(
    config: Config | None = None,
    name: str | None = None,
) -> trace.Tracer:
    """Configure and return a tracer with proper integration.

    This function sets up `OpenTelemetry TracerProvider` with
    configuration based on the framework's configurations. It
    automatically selects appropriate processors and exporters based on
    the environment and telemetry configuration.

    :param config: An optional configuration object to initialise the
        tracer. If not provided, a default `Config` instance is created.
    :param name: Override for the service name, defaults to `None`. If
        not provided, uses the name from the configuration.
    :return: A configured `OpenTelemetry Tracer` instance.
    """
    if config is None:
        config = Config()
    service = name or config.telemetry.name or config.name
    resource = Resource.create(
        {
            "service.name": service,
            "service.version": getattr(config, "version", "unknown"),
            "deployment.environment": (
                "development" if config.debug else "production"
            ),
            "telemetry.sdk.name": "lauren",
        }
    )
    if not config.telemetry.enabled:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        return trace.get_tracer(service)
    if config.debug:
        processor = SimpleSpanProcessor(ConsoleSpanExporter())
    else:
        try:
            exporter = OTLPSpanExporter()
            processor = BatchSpanProcessor(exporter)
        except Exception:
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service)
