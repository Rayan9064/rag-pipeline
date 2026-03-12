from __future__ import annotations

from pathlib import Path

from .chunking import TextChunk, load_documents, chunk_documents
from .generator import Generator
from .retriever import RetrievedChunk, Retriever


class RAGPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.retriever = Retriever(embedding_model_name=embedding_model_name)
        self.generator = Generator(model_name=llm_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[TextChunk] = []

    def ingest_documents(self, docs_dir: str | Path) -> int:
        docs_path = Path(docs_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            raise ValueError(f"Invalid docs directory: {docs_path}")

        docs = load_documents(str(docs_path))
        all_chunks = chunk_documents(docs, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if not all_chunks:
            raise ValueError("No documents with content were found to ingest.")
        self.retriever.build_index(all_chunks)
        self._chunks = all_chunks
        return len(all_chunks)

    def answer(self, question: str, top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            [f"[{i+1}] {item.chunk.text}" for i, item in enumerate(retrieved)]
        )

        prompt = (
            "You are an HR assistant for internal policy Q&A. "
            "Answer only using the provided context. "
            "If the answer is not in context, say you don't know.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        response = self.generator.generate(prompt)
        return response, retrieved
