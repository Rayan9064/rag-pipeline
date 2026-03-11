from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_id: str
    source: str
    text: str


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[TextChunk] = []

    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        snippet = normalized[start:end].strip()
        if snippet:
            chunks.append(
                TextChunk(
                    chunk_id=f"{source}::chunk_{chunk_index}",
                    source=source,
                    text=snippet,
                )
            )
            chunk_index += 1

        if end == len(normalized):
            break
        start += step

    return chunks
