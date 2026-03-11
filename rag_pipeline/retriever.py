from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .chunking import TextChunk


@dataclass
class RetrievedChunk:
    chunk: TextChunk
    score: float


class Retriever:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks: list[TextChunk] = []
        self._embeddings: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def build_index(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks supplied to build the index")

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self._embeddings = embeddings.astype(np.float32)
        self._nn = NearestNeighbors(metric="cosine")
        self._nn.fit(self._embeddings)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self._nn is None or self._embeddings is None:
            raise ValueError("Index not built. Call build_index first.")

        query_vec = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = query_vec.astype(np.float32)

        k = min(top_k, len(self.chunks))
        distances, indices = self._nn.kneighbors(query_vec, n_neighbors=k)

        results: list[RetrievedChunk] = []
        for distance, idx in zip(distances[0], indices[0]):
            similarity = 1.0 - float(distance)
            results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=similarity))
        return results
