"""
utils/embeddings.py
Generates vector embeddings using SentenceTransformers.
"""

from sentence_transformers import SentenceTransformer

# Lazy-loaded model (cached after first call)
_model = None
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embedding(text: str) -> list:
    """
    Generate embedding vector for the given text using SentenceTransformers.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats (384-dimensional embedding for all-MiniLM-L6-v2)
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    
    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def get_embeddings_batch(texts: list) -> list:
    """
    Generate embedding vectors for multiple texts in batch.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors (each a list of floats)
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    
    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """
    Get the dimension of the embedding vectors.
    
    Returns:
        Integer dimension of the embedding (384 for all-MiniLM-L6-v2)
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.get_sentence_embedding_dimension()
