from typing import List
import numpy as np
from collections import defaultdict, deque

class HierarchicalContext:
    """Multi-level context tracking"""
    def __init__(self):
        self.immediate = deque(maxlen=5)
        self.conversation = deque(maxlen=20)
        self.topics = defaultdict(float)
        
    def update(self, tokens: List[str]):
        # Update context queues
        self.immediate.append(tokens)
        self.conversation.append(tokens)
        
        # Update topic weights with decay
        for token in tokens:
            self.topics[token] = 0.95 * self.topics[token] + 0.05
            
    def get_context_vector(self, vector_mgr) -> np.ndarray:
        """Create weighted context vector"""
        all_tokens = [t for ctx in self.conversation for t in ctx]
        weights = [self.topics[t] for t in all_tokens]
        vectors = [vector_mgr.get_vector(t) for t in all_tokens]
        return np.average(vectors, axis=0, weights=weights)
