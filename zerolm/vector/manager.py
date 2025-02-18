"""Vector management and similarity search implementation for ZeroLM."""

import numpy as np
from typing import List, Tuple

class VectorManager:
    """Optimized vector storage with sparse representation"""
    def __init__(self, vector_dim: int = 100, use_sparse: bool = False):
        self.vector_dim = vector_dim
        self.use_sparse = use_sparse
        self.vectors = {}
        self.responses = {}
        
    def vectorize(self, tokens: List[str]) -> np.ndarray:
        """Convert tokens to vector representation"""
        if not tokens:
            return np.zeros(self.vector_dim)
            
        # Simple averaging of word vectors
        vector = np.zeros(self.vector_dim)
        for token in tokens:
            # Generate a random but consistent vector for each token
            token_hash = hash(token) % self.vector_dim
            token_vector = np.zeros(self.vector_dim)
            token_vector[token_hash] = 1.0
            vector += token_vector
            
        return vector / max(len(tokens), 1)
        
    def add_vector(self, vector: np.ndarray, response: str) -> None:
        """Store vector and response"""
        vector_key = tuple(vector)  # Convert to hashable type
        self.vectors[vector_key] = vector
        self.responses[vector_key] = response
        
    def find_similar(
        self,
        query_vector: np.ndarray,
        threshold: float = 0.5,
        max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar vectors with proper parameters"""
        results = []
        for vec_key, response in self.responses.items():
            similarity = np.dot(query_vector, np.array(vec_key))
            if similarity >= threshold:
                results.append((response, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)[:max_results]