from typing import List
import numpy as np
import time
import math

from zerolm.types import ResponseType
from zerolm.core import Response
from zerolm.memory import VectorManager

class KnowledgeDistiller:
    """Knowledge distiller integrated with ZeroLM components"""
    
    def __init__(self, vector_mgr: VectorManager, memory):
        self.vector_mgr = vector_mgr
        self.memory = memory
        self.source_models = {
            'primary': {
                'name': 'vector_similarity',
                'handler': self._handle_vector_model,
                'weight': 0.6
            },
            'secondary': {
                'name': 'pattern_matching',
                'handler': self._handle_pattern_model,
                'weight': 0.4
            }
        }

    def distill_response(self, query: str, context: dict) -> Response:
        """Main distillation flow integrated with the model"""
        vector_resp = self.source_models['primary']['handler'](query)
        pattern_resp = self.source_models['secondary']['handler'](query)
        
        if self._responses_aligned(vector_resp, pattern_resp):
            return self._merge_responses(vector_resp, pattern_resp, context)
        return self._select_response(vector_resp, pattern_resp)

    def _handle_vector_model(self, query: str) -> Response:
        """Vector processing using existing infrastructure"""
        tokens = self._tokenize(query)
        query_vec = self.vector_mgr.get_query_vector(tokens)
        matches = self.vector_mgr.find_similar(query_vec)
        
        if matches:
            best_match = matches[0]
            return Response(
                text=best_match['response'],
                type=ResponseType.VECTOR_BASED,
                confidence=best_match['confidence'],
                metadata={
                    'timestamp': time.time(),
                    'method': 'vector_similarity',
                    'matched_vector': best_match['vector']
                }
            )
        return Response(
            text="No relevant vector match found.",
            type=ResponseType.LEARNING,
            confidence=0.0
        )

    def _handle_pattern_model(self, query: str) -> Response:
        """Pattern matching using existing memory"""
        tokens = self._tokenize(query)
        expanded_tokens = self._expand_tokens(tokens)
        
        best_match = None
        max_confidence = 0.0
        
        for pattern in self._get_candidate_patterns(expanded_tokens):
            match_score = self._calculate_pattern_score(pattern, expanded_tokens)
            if match_score > max_confidence:
                responses = self.memory.get(pattern, [])
                if responses:
                    best_response = max(responses, key=lambda x: x[1])
                    max_confidence = match_score
                    best_match = (best_response[0], best_response[1])
        
        if best_match and max_confidence >= self.memory.min_confidence:
            return Response(
                text=best_match[0],
                type=ResponseType.PATTERN_MATCH,
                confidence=max_confidence,
                metadata={
                    'timestamp': time.time(),
                    'method': 'pattern_matching',
                    'matched_pattern': list(pattern)
                }
            )
        return Response(
            text="Pattern not recognized.",
            type=ResponseType.LEARNING,
            confidence=0.0
        )

    def _responses_aligned(self, resp1: Response, resp2: Response) -> bool:
        """Checks alignment considering context and semantics"""
        return (
            abs(resp1.confidence - resp2.confidence) < 0.2 and
            self._semantic_similarity(resp1.text, resp2.text) > 0.7
        )

    def _merge_responses(self, resp1: Response, resp2: Response, context: dict) -> Response:
        """Combines responses using contextual weights"""
        time_diff = abs(resp1.metadata['timestamp'] - resp2.metadata['timestamp'])
        recency_weight = math.exp(-0.1 * (time_diff / 3600))  # Decay per hour
        
        weights = {
            'vector': self.source_models['primary']['weight'] * recency_weight,
            'pattern': self.source_models['secondary']['weight'] * (1 - recency_weight)
        }
        
        total_weight = weights['vector'] + weights['pattern']
        normalized_weights = {
            'vector': weights['vector'] / total_weight,
            'pattern': weights['pattern'] / total_weight
        }
        
        merged_text = f"{resp1.text} ({resp2.text})" if recency_weight > 0.7 else resp1.text
        merged_confidence = (resp1.confidence * normalized_weights['vector'] +
                            resp2.confidence * normalized_weights['pattern'])
        
        return Response(
            text=merged_text,
            type=resp1.type if resp1.confidence >= resp2.confidence else resp2.type,
            confidence=merged_confidence,
            metadata={
                'components': [resp1.metadata, resp2.metadata],
                'merge_weights': normalized_weights
            }
        )

    def _select_response(self, resp1: Response, resp2: Response) -> Response:
        """Selection based on confidence and temporal context"""
        time_diff = resp2.metadata['timestamp'] - resp1.metadata['timestamp']
        recency_bonus = 0.1 * (1 - math.exp(-time_diff/3600))  # 10% bonus for more recent responses
        
        adjusted_confidence = {
            'vector': resp1.confidence + (recency_bonus if time_diff < 0 else 0),
            'pattern': resp2.confidence + (recency_bonus if time_diff > 0 else 0)
        }
        
        if adjusted_confidence['vector'] >= adjusted_confidence['pattern']:
            return resp1
        return resp2

    # Helper methods integrated with the rest of the system
    def _tokenize(self, text: str) -> List[str]:
        return [word.lower() for word in text.split() if len(word) > 2]

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        vec1 = self.vector_mgr.get_query_vector(self._tokenize(text1))
        vec2 = self.vector_mgr.get_query_vector(self._tokenize(text2))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_candidate_patterns(self, tokens: List[str]) -> List[frozenset]:
        """Gets candidate patterns from all memory layers"""
        candidates = []
        for storage in [self.memory.hot_storage,
                        self.memory.warm_storage,
                        self.memory.cold_storage]:
            candidates.extend(
                p for p in storage.keys()
                if any(t in p for t in tokens)
            )
        return sorted(candidates, key=lambda x: len(x), reverse=True)[:100]
