"""
retriever.py
Encodes the user query and retrieves top-k similar chunks from Endee.
"""

from typing import List

import endee
from sentence_transformers import SentenceTransformer

from src.config import ENDEE_COLLECTION, EMBED_MODEL

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def query_knowledge_base(query: str, top_k: int = 3) -> List[str]:
    """
    Embed the query and retrieve the top-k most relevant document chunks
    from the Endee vector database.

    Args:
        query:  Natural language question from the user.
        top_k:  Number of chunks to retrieve.

    Returns:
        List of document strings (chunks) ordered by relevance.
    """
    model = _get_model()
    query_embedding = model.encode(query).tolist()

    db = endee.Client()
    collection = db.get_collection(ENDEE_COLLECTION)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents"],
    )

    # results["documents"] is a list-of-lists (one list per query)
    documents: List[str] = results["documents"][0] if results["documents"] else []
    return documents
