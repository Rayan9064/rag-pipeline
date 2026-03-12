from __future__ import annotations


from dataclasses import dataclass
from typing import List
import os

@dataclass
class TextChunk:
    chunk_id: str
    source: str
    text: str

def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(file_path: str) -> str:
    import pdfplumber
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def load_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def load_documents(folder: str) -> List[dict]:
    docs = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if fname.lower().endswith(".txt"):
            text = load_txt(fpath)
        elif fname.lower().endswith(".pdf"):
            text = load_pdf(fpath)
        elif fname.lower().endswith(".docx"):
            text = load_docx(fpath)
        else:
            continue
        docs.append({"source": fname, "text": text})
    return docs

def chunk_documents(docs: List[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for doc in docs:
        text = " ".join(doc["text"].split())
        source = doc["source"]
        if not text:
            continue
        step = chunk_size - chunk_overlap
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            snippet = text[start:end].strip()
            if snippet:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{source}::chunk_{chunk_index}",
                        source=source,
                        text=snippet,
                    )
                )
                chunk_index += 1
            if end == len(text):
                break
            start += step
    return chunks
