from __future__ import annotations

from dataclasses import dataclass


import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .chunking import TextChunk
from dataclasses import dataclass

@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float

class Retriever:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks: list[TextChunk] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None

    def build_index(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to build the index")
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embeddings.astype(np.float32)
        dim = self._embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(self._embeddings)
        self._faiss_index = index

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self._faiss_index is None or self._embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
        query_vec = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = query_vec.astype(np.float32)
        k = min(top_k, len(self.chunks))
        scores, indices = self._faiss_index.search(query_vec, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score)))
        return results
