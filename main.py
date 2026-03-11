from __future__ import annotations

import argparse

from rag_pipeline import RAGPipeline
from rag_pipeline.otel import setup_otel
from opentelemetry import trace, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="First RAG pipeline (HR policy assistant)")
    parser.add_argument("--docs", default="data", help="Path to folder containing .txt knowledge docs")
    parser.add_argument("--question", default="", help="Question to ask (leave blank for interactive mode)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face embedding model name",
    )
    parser.add_argument(
        "--llm-model",
        default="google/flan-t5-base",
        help="Hugging Face open-source LLM model name",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Initialize OpenTelemetry
    setup_otel()
    tracer = trace.get_tracer("rag-pipeline")
    meter = metrics.get_meter("rag-pipeline")
    answer_time_hist = meter.create_histogram(
        name="answer_time_ms",
        unit="ms",
        description="Time to generate answer (ms)"
    )
    accuracy_counter = meter.create_counter(
        name="answer_accuracy",
        unit="1",
        description="Count of correct answers (manual marking)"
    )

    pipeline = RAGPipeline(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
    )
    num_chunks = pipeline.ingest_documents(args.docs)
    print(f"Indexed chunks: {num_chunks}")

    import time
    while True:
        try:
            question = args.question
            if not question:
                question = input("\nEnter your question (or 'exit' to quit): ").strip()
            else:
                print(f"\nQuestion: {question}")
            if question.lower() in ("exit", "quit"): break

            with tracer.start_as_current_span("rag_query") as span:
                t0 = time.perf_counter()
                answer, retrieved = pipeline.answer(question, top_k=args.top_k)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000
                answer_time_hist.record(elapsed_ms)
                span.set_attribute("answer.time_ms", elapsed_ms)

                print("\nRetrieved context:")
                for rank, item in enumerate(retrieved, start=1):
                    print(f"{rank}. source={item.chunk.source} score={item.score:.4f}")

                print("\nAnswer:")
                print(answer)

                user_acc = input("Was the answer correct? (y/n): ").strip().lower()
                if user_acc == "y":
                    accuracy_counter.add(1)
                    span.set_attribute("answer.accuracy", 1)
                else:
                    span.set_attribute("answer.accuracy", 0)

            # After first question, clear args.question so next is interactive
            args.question = ""
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            break


if __name__ == "__main__":
    main()
