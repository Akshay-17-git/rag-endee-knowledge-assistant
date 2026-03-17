"""
config.py
Central configuration for the AI Knowledge Assistant.
"""

# Endee vector database
ENDEE_COLLECTION = "knowledge_base"

# Embedding model (SentenceTransformers)
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters
CHUNK_SIZE    = 200   # words per chunk
CHUNK_OVERLAP = 30    # overlapping words between consecutive chunks
