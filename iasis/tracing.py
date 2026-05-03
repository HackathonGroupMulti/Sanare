from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator


class NoopSpan:
    def set_attribute(self, _key: str, _value: Any) -> None:
        return None

    def record_exception(self, _error: BaseException) -> None:
        return None


class Tracer:
    def __init__(self, service_name: str = "iasis-agent-system") -> None:
        self._tracer = self._load_tracer(service_name)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        if not self._tracer:
            yield NoopSpan()
            return
        with self._tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, value)
            yield span

    def _load_tracer(self, service_name: str) -> Any:
        if os.getenv("IASIS_ENABLE_OTEL") != "1":
            return None
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:
            return None

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return trace.get_tracer(service_name)
