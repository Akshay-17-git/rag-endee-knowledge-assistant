"""
utils/endee_client.py
Endee vector database client for storing and searching embeddings.
Provides an in-memory implementation with cosine similarity search.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class EndeeClient:
    """
    In-memory vector database client that mimics the Endee SDK interface.
    Provides high-performance similarity search with cosine distance.
    """
    
    def __init__(self, collection_name: str = "default"):
        """
        Initialize the EndeeClient.
        
        Args:
            collection_name: Name of the collection (for future multi-collection support)
        """
        self.collection_name = collection_name
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def insert(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """
        Insert a single vector with metadata into the collection.
        
        Args:
            id: Unique identifier for the vector
            vector: Embedding vector (list of floats)
            metadata: Associated metadata dictionary
        """
        self.vectors[id] = np.array(vector, dtype=np.float32)
        self.metadata[id] = metadata
    
    def insert_batch(self, records: List[Dict[str, Any]]) -> None:
        """
        Insert multiple vectors in batch for efficiency.
        
        Args:
            records: List of dictionaries with 'id', 'vector', and 'metadata' keys
        """
        for record in records:
            self.insert(record['id'], record['vector'], record['metadata'])
    
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for the top-k most similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of result dictionaries with 'id', 'score', and 'metadata'
        """
        if not self.vectors:
            return []
        
        query = np.array(query_vector, dtype=np.float32)
        
        # Calculate cosine similarity for all vectors
        results = []
        for id, vector in self.vectors.items():
            # Cosine similarity = dot product / (norm(a) * norm(b))
            norm_query = np.linalg.norm(query)
            norm_vector = np.linalg.norm(vector)
            
            if norm_query == 0 or norm_vector == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query, vector) / (norm_query * norm_vector)
            
            results.append({
                "id": id,
                "score": float(similarity),
                "metadata": self.metadata[id]
            })
        
        # Sort by similarity score (descending) and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def delete(self, id: str) -> bool:
        """
        Remove a specific vector from the collection.
        
        Args:
            id: Unique identifier of the vector to delete
            
        Returns:
            True if deleted, False if not found
        """
        if id in self.vectors:
            del self.vectors[id]
            del self.metadata[id]
            return True
        return False
    
    def clear(self) -> None:
        """
        Clear all vectors and metadata from the collection.
        """
        self.vectors.clear()
        self.metadata.clear()
    
    def info(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection info
        """
        return {
            "collection_name": self.collection_name,
            "num_vectors": len(self.vectors),
            "dimension": len(next(iter(self.vectors.values()), [])) if self.vectors else 0
        }
    
    def get(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific vector and its metadata.
        
        Args:
            id: Unique identifier of the vector
            
        Returns:
            Dictionary with 'id', 'vector', and 'metadata', or None if not found
        """
        if id in self.vectors:
            return {
                "id": id,
                "vector": self.vectors[id].tolist(),
                "metadata": self.metadata[id]
            }
        return None
    
    def __len__(self) -> int:
        """Return the number of vectors in the collection."""
        return len(self.vectors)
