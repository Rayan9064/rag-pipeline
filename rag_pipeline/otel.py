from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader

def setup_otel():
    # Tracing
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)
    span_processor = BatchSpanProcessor(ConsoleSpanExporter())
    tracer_provider.add_span_processor(span_processor)

    # Metrics
    meter_provider = MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())]
    )
    metrics.set_meter_provider(meter_provider)