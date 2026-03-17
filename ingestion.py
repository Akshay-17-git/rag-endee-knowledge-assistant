"""
ingestion.py
Handles document loading, chunking, embedding, and indexing into Endee.
"""

import io
import re
from typing import List

import endee
from sentence_transformers import SentenceTransformer

from src.config import ENDEE_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL

# Lazy-loaded model (cached after first call)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# ── Text extraction ──────────────────────────────────────────────────────────

def _extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    except ImportError:
        # Fallback: PyPDF2
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )


def _extract_text(file) -> str:
    file_bytes = file.read()
    name = file.name.lower()
    if name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    return _extract_text_from_txt(file_bytes)


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


# ── Endee indexing ───────────────────────────────────────────────────────────

def ingest_documents(uploaded_files) -> int:
    """
    Ingest uploaded Streamlit files into the Endee vector database.
    Returns the total number of chunks indexed.
    """
    model = _get_model()

    # Ensure collection exists
    db = endee.Client()
    if ENDEE_COLLECTION not in db.list_collections():
        db.create_collection(ENDEE_COLLECTION, dimension=model.get_sentence_embedding_dimension())

    collection = db.get_collection(ENDEE_COLLECTION)

    all_chunks: List[str] = []
    chunk_ids: List[str] = []
    metadatas: List[dict] = []

    for file in uploaded_files:
        text = _extract_text(file)
        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{file.name}__chunk_{idx}"
            all_chunks.append(chunk)
            chunk_ids.append(chunk_id)
            metadatas.append({"source": file.name, "chunk_index": idx})

    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded files.")

    # Batch embed
    embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()

    # Upsert into Endee
    collection.upsert(
        ids=chunk_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=metadatas,
    )

    return len(all_chunks)
