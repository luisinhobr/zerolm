from typing import List, Optional, Dict, Set, Tuple, Any
import numpy as np
import time
import logging
from collections import defaultdict, deque
import os
import json
import pickle

from zerolm.types import ResponseType
from zerolm.core import Response
from zerolm.memory import AdaptiveMemoryManager, VectorManager, ConcurrentLRUCache
from zerolm.matching import PatternMatcher
from zerolm.context import HierarchicalContext, ContextWeighter
from zerolm.learning import LearningValidator
from zerolm.validation import TemplateValidator, TemplateEnforcer
from zerolm.utils import PerformanceMetrics, AutoCorrector, logger

class ZeroShotLM:
    def __init__(
        self,
        vector_dim: int = 100,
        use_sparse_vectors: bool = False,
        min_confidence: float = 0.3,
        temporal_weight: float = 0.7,
        context_window: int = 5,
        language: str = "en",
        max_patterns: int = 10000,
        learning_rate: float = 0.1,
        use_vectors: bool = True
    ):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Store configuration parameters
        self.vector_dim = vector_dim
        self.use_vectors = use_vectors  # Set this before bootstrapping
        self.use_sparse_vectors = use_sparse_vectors
        self.min_confidence = min_confidence
        self.temporal_weight = temporal_weight
        self.language = language.lower()
        self.learning_rate = learning_rate
        
        # Initialize components
        self.memory = AdaptiveMemoryManager(max_patterns=max_patterns)
        self.vector_mgr = VectorManager(
            vector_dim=vector_dim,
            use_sparse=use_sparse_vectors
        )
        self.pattern_matcher = PatternMatcher(min_confidence=min_confidence)
        
        # Initialize context window
        self.context_window = deque(maxlen=context_window)
        
        # Initialize synonym dictionary
        self.synonyms = defaultdict(set)
        
        # Bootstrap core patterns
        self._bootstrap_core_patterns()
        
        # Initialize learning responses
        self.learning_responses = self._initialize_learning_responses()
        
        # Vector-based components
        self.vector_index = defaultdict(set)  # {token: set(frozenset patterns)}
        
        # Performance optimizations
        self.pattern_index = defaultdict(set)  # token -> set(frozenset patterns)
        self.recent_patterns = deque(maxlen=max_patterns)
        self.vector_cache = ConcurrentLRUCache(maxsize=1000)
        
        # Learning controls
        self.vector_decay = 0.99
        self.batch_buffer = []
        self.batch_size = 100
        
        # Memory management
        self.cleanup_interval = 3600
        self.last_cleanup = time.time()
        self.pattern_usage = defaultdict(int)
        self.token_usage = defaultdict(int)
        
        # Enhanced components
        self.context = HierarchicalContext()
        self.learning_validator = LearningValidator()
        self.template_validator = TemplateValidator()
        self.workers = 4
        self.metrics = PerformanceMetrics()

        self.weighter = ContextWeighter()
        self.corrector = AutoCorrector()
        self.enforcer = TemplateEnforcer()

        # Add config dictionary
        self.config = {
            'min_confidence': min_confidence,
            'temporal_weight': temporal_weight,
            'learning_rate': learning_rate,
            'vector_dim': vector_dim,
            'use_vectors': use_vectors,
            'use_sparse_vectors': use_sparse_vectors,
            'language': language,
            'max_patterns': max_patterns
        }

    def _initialize_learning_responses(self) -> Dict[str, List[str]]:
        """Initialize responses for learning interactions"""
        if self.language == "pt":
            return {
                "unknown": [
                    "Não sei como responder isso ainda. Pode me ensinar?",
                    "Ainda não aprendi sobre isso. Pode me explicar?",
                    "Não conheço essa! Como você responderia?",
                ],
                "learning_success": [
                    "Obrigado! Aprendi algo novo!",
                    "Entendi! Vou lembrar disso!",
                    "Legal! Agora sei como responder isso!",
                ],
                "learning_error": [
                    "Desculpe, tive um problema ao aprender isso.",
                    "Ops! Algo deu errado ao tentar aprender.",
                    "Não consegui aprender isso corretamente.",
                ],
                "low_confidence": [
                    "Não tenho certeza, mas acho que seria assim...",
                    "Talvez seja assim, mas não estou muito confiante.",
                    "Posso tentar responder, mas não tenho certeza.",
                ]
            }
        else:  # Default to English
            return {
                "unknown": [
                    "I don't know how to answer this yet. Can you teach me?",
                    "I haven't learned about this. Could you explain?",
                    "Interesting! How would you answer this?",
                ],
                "learning_success": [
                    "Thank you! I learned something new!",
                    "Got it! I'll remember that!",
                    "Great! Now I know how to respond to this!",
                ],
                "learning_error": [
                    "Sorry, I had trouble learning that.",
                    "Oops! Something went wrong while learning.",
                    "I couldn't learn that properly.",
                ],
                "low_confidence": [
                    "I'm not sure, but I think it might be...",
                    "Maybe this, but I'm not very confident.",
                    "I can try to answer, but I'm uncertain.",
                ]
            }

    def get_context(self) -> List[str]:
        """Retrieves contextual tokens from the context window"""
        return [token for tokens, _ in self.context_window for token in tokens]

    def process_query(self, query: str) -> Response:
        """Process input query and generate response"""
        try:
            tokens = self.tokenize(query)
            if not tokens:
                return Response("Please provide a valid input.", 0.0, ResponseType.ERROR)
                
            # Update context window
            self.context_window.append((tokens, time.time()))
            
            # Try to generate response
            result = self._generate_response(tokens)
            if not result:
                return Response(
                    "I don't know how to answer that yet. Can you teach me?",
                    0.0,
                    ResponseType.UNKNOWN
                )
                
            response_text, confidence = result
            return Response(response_text, confidence, ResponseType.NORMAL)
            
        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            return Response(
                "I encountered an error processing your message.",
                0.0,
                ResponseType.ERROR
            )

    def _generate_response(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Generate response for input tokens"""
        try:
            # Convert tokens to set for pattern matching
            token_set = frozenset(tokens)
            
            # Check patterns in memory
            best_match = None
            best_confidence = 0.0
            
            for pattern, data in self.memory.hot_storage.items():
                # Calculate pattern match confidence
                confidence = self._calculate_match_confidence(pattern, token_set)
                
                # Update best match if confidence is higher and meets minimum threshold
                if confidence > best_confidence and confidence >= self.min_confidence:
                    best_match = data
                    best_confidence = confidence
            
            if best_match:
                return best_match['response'], best_confidence
                
            # Try semantic matching if no exact match and vectors are enabled
            if self.use_vectors:
                vector_matches = self._semantic_fallback(tokens)
                if vector_matches:
                    return vector_matches
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return None

    def learn(self, query: str, response: str) -> bool:
        """Learn new pattern-response pair"""
        try:
            tokens = self.tokenize(query)
            pattern = frozenset(tokens)
            
            # Calculate learning confidence
            confidence = self._calculate_learning_confidence(query, response)
            
            # Add pattern to memory with confidence
            success = self.memory.add_pattern(pattern, response, confidence)
            
            # Update vectors if enabled
            if success and self.use_vectors:
                vector = self.vector_mgr.vectorize(tokens)
                self.vector_mgr.add_vector(vector, response)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Learning failed: {str(e)}")
            return False

    def _update_usage_stats(self, pattern: frozenset, tokens: List[str]):
        """Track pattern and token usage"""
        self.pattern_usage[pattern] += 1
        for token in tokens:
            self.token_usage[token] += 1

    def _update_vectors(self, tokens: List[str], response: str) -> None:
        """Update vector representations for tokens"""
        if not self.use_vectors or not tokens:
            return
            
        try:
            vector = self.vector_mgr.vectorize(tokens)
            self.vector_mgr.add_vector(vector, response)
        except Exception as e:
            self.logger.error(f"Vector update failed: {str(e)}")

    def _store_pattern(self, pattern: frozenset, response: str):
        """Store pattern with timestamp"""
        current_time = time.time()
        self.memory.add_pattern(pattern, response, current_time)
        self.recent_patterns.append(pattern)
        self._update_indexes(pattern, list(pattern))

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text"""
        if not text:
            return []
            
        # Basic tokenization
        tokens = text.lower().split()
        return [token.strip() for token in tokens if token.strip()]

    def _update_context(self, tokens: List[str]):
        """Update conversation context window"""
        current_time = time.time()
        self.context_window.append((tokens, current_time))
        
        # Maintain window size and remove old context
        window_start = current_time - (3600 * 24)  # 24 hour window
        self.context_window = [
            (tokens, timestamp) 
            for tokens, timestamp in self.context_window[-self.config["context_window"]:]
            if timestamp >= window_start
        ]

    def _calculate_match_confidence(self, pattern: frozenset, tokens: frozenset) -> float:
        """Calculate match confidence between pattern and input tokens"""
        try:
            if len(pattern) == 0 or len(tokens) == 0:
                return 0.0
                
            # Calculate intersection and union
            intersection = len(pattern.intersection(tokens))
            union = len(pattern.union(tokens))
            
            if union == 0:
                return 0.0
                
            # Basic Jaccard similarity
            confidence = intersection / union
            
            # Apply temporal weighting if context window is not empty
            if self.context_window:
                time_factor = 1.0
                current_time = time.time()
                for past_tokens, timestamp in self.context_window:
                    if pattern.intersection(past_tokens):
                        time_delta = current_time - timestamp
                        time_factor *= (1.0 - (time_delta * self.temporal_weight))
                confidence *= time_factor
                
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating match confidence: {str(e)}")
            return 0.0

    def _analyze_patterns(self, tokens: List[str], response: str):
        """Analyze patterns to improve understanding"""
        # Extract content words (non-function words)
        content_words = set(tokens) - self._get_function_words()
        
        # Update synonym relationships
        for word1 in content_words:
            for word2 in content_words:
                if word1 != word2:
                    self.synonyms[word1].add(word2)

    def _get_function_words(self) -> Set[str]:
        """Get function words for the configured language"""
        if self.config["language"] == "pt-BR":
            return {
                'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
                'de', 'em', 'por', 'para', 'com', 'sem', 'sob',
                'que', 'e', 'mas', 'se', 'como', 'quando'
            }
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'with', 'by', 'from', 'up', 'about', 'like',
            'through', 'after', 'over', 'between', 'out', 'against',
            'during', 'without', 'before', 'under', 'around', 'among'
        }

    def _get_current_context(self) -> List[str]:
        """Get current context window tokens"""
        return [t for tokens, _ in self.context_window for t in tokens]

    def _create_metadata(self, tokens: List[str], confidence: float) -> Dict[str, Any]:
        """Create response metadata"""
        return {
            "token_count": len(tokens),
            "context_size": len(self.context_window),
            "confidence": confidence,
            "timestamp": time.time()
        }

    def _select_best_match(self, matches: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Select best matching response based on recency"""
        return max(matches, key=lambda x: x[1])

    def save(self, file_path: str, export_format: str = 'binary') -> None:
        """Enhanced save with format options and validation"""
        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError(f"Invalid path: {file_path}")
        
        if export_format == 'binary':
            self._save_binary(file_path)
        elif export_format == 'json':
            self._save_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {export_format}")

    def _save_binary(self, file_path: str):
        state = {
            'memory': dict(self.memory),
            'synonyms': dict(self.synonyms),
            'config': self.config,
            'vectors': dict(self.vector_mgr.vectors) if self.use_vectors else None,
            'pattern_index': dict(self.pattern_index)
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_json(self, file_path: str):
        export_data = {
            'patterns': [
                {
                    'tokens': list(pattern),
                    'responses': [
                        {'text': r, 'timestamp': ts}
                        for r, ts in responses
                    ]
                }
                for pattern, responses in self.memory.items()
            ],
            'synonyms': {
                k: list(v) for k, v in self.synonyms.items()
            },
            'config': self.config
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False)

    def _vectorize(self, tokens: List[str]) -> np.ndarray:
        """Create sentence vector using token embeddings"""
        if not self.use_vectors or not tokens:
            return np.zeros(self.vector_dim)
            
        return np.mean([self.vector_mgr.get_vector(token) for token in tokens], axis=0)

    def _find_vector_matches(self, query_vector: np.ndarray) -> List[Tuple[str, float]]:
        """Efficient vector similarity search"""
        candidates = set()
        for token in query_vector.nonzero()[0]:  # Assuming sparse representation
            candidates.update(self.vector_index.get(token, []))
        
        return sorted(
            ((resp, self._cosine_similarity(query_vector, vec), ts)
             for pattern in candidates
             for resp, ts in self.memory.hot_storage.get(pattern, [])
             if (vec := self._pattern_vector(pattern)) is not None),
            key=lambda x: (-x[1], -x[2])
        )[:100]  # Return top 100 matches

    def _pattern_vector(self, pattern: frozenset) -> np.ndarray:
        """Get cached pattern vector"""
        if pattern not in self.vector_cache.cache:
            self.vector_cache.put(pattern, self._vectorize(list(pattern)))
        return self.vector_cache.cache[pattern]

    def _validate_vector(self, vector: np.ndarray):
        """Vector validation"""
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"Vector dimension mismatch: {len(vector)} != {self.vector_dim}"
            )

    def set_learning_parameters(
        self,
        learning_rate: Optional[float] = None,
        vector_decay: Optional[float] = None,
        max_patterns: Optional[int] = None
    ):
        """Parameter validation"""
        if learning_rate is not None:
            if not 0 < learning_rate <= 1:
                raise ValueError("Learning rate must be between 0 and 1")
            self.learning_rate = learning_rate
            
        if vector_decay is not None:
            if not 0.9 <= vector_decay < 1:
                raise ValueError("Vector decay must be â‰¥0.9 and <1")
            self.vector_decay = vector_decay
            
        if max_patterns is not None:
            if max_patterns <= 0:
                raise ValueError("max_patterns must be greater than 0")
            self.recent_patterns = deque(list(self.recent_patterns)[:max_patterns], maxlen=max_patterns)

    def export_patterns(self, min_confidence: float = 0.5) -> Dict:
        """Export learned patterns meeting confidence threshold"""
        return {
            'patterns': [
                {
                    'tokens': list(pattern),
                    'responses': [
                        {'text': r, 'last_used': ts}
                        for r, ts in responses
                        if self._calculate_confidence(pattern, r, ts) >= min_confidence
                    ]
                }
                for pattern, responses in self.memory.items()
            ],
            'synonyms': dict(self.synonyms)
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def get_memory_stats(self) -> str:
        """Get formatted memory statistics"""
        stats = self.memory.get_stats()
    
        return f"""
        Memory Statistics:
        - Total Patterns: {stats['total_patterns']}
        - Storage Distribution:
          • Hot: {stats['storage_distribution']['hot']}
          • Warm: {stats['storage_distribution']['warm']}
          • Cold: {stats['storage_distribution']['cold']}
        - Confidence Distribution:
          ≥0.9: {stats['confidence_distribution']['high']}
          0.7-0.9: {stats['confidence_distribution']['medium']}
          <0.7: {stats['confidence_distribution']['low']}
        - Recent Patterns:
          {self._format_recent_patterns()}
    """
    
    def _format_recent_patterns(self) -> str:
        """Format recent patterns for display"""
        recent = sorted(
            self.memory.hot_storage.items(),
            key=lambda x: x[1]['timestamp'],
            reverse=True
        )[:5]
    
        if not recent:
         return "No patterns learned yet"
        
        return "\n          ".join(
            f"• {' '.join(pattern)} → {data['response'][:30]}..."
            for pattern, data in recent
            )

    def optimize_memory(self, target_size_mb: float):
        """Memory optimization implementation"""
        current_stats = self.get_memory_stats()
        
        if current_stats.vector_memory_mb <= target_size_mb:
            logger.info(f"Memory already at {current_stats.vector_memory_mb:.2f}MB (target: {target_size_mb}MB)")
            return

        # Calculate how much we need to prune
        excess_mb = current_stats.vector_memory_mb - target_size_mb
        logger.warning(f"Memory optimization triggered. Current: {current_stats.vector_memory_mb:.2f}MB, Target: {target_size_mb}MB, Need to free: {excess_mb:.2f}MB")

        # Phase 1: Vector pruning
        bytes_per_vector = self.vector_dim * 4  # 4 bytes per float32
        vectors_to_remove = int((excess_mb * 1024 * 1024) / bytes_per_vector)
        
        # Remove least used tokens first
        sorted_tokens = sorted(self.token_usage.items(), key=lambda x: x[1])
        removed_vectors = 0
        for token, _ in sorted_tokens[:vectors_to_remove]:
            if token in self.vector_mgr.vectors:
                del self.vector_mgr.vectors[token]
                removed_vectors += 1
            # Cleanup related indexes
            if token in self.vector_index:
                del self.vector_index[token]
            if token in self.synonyms:
                del self.synonyms[token]

        # Phase 2: Pattern pruning if still over limit
        new_stats = self.get_memory_stats()
        if new_stats.vector_memory_mb > target_size_mb:
            # Remove oldest patterns first
            all_patterns = [
                (pattern, max(ts for _, ts in responses))
                for pattern, responses in self.memory.items()
            ]
            sorted_patterns = sorted(all_patterns, key=lambda x: x[1])
            
            # Calculate how many patterns to remove
            excess_after_phase1 = new_stats.vector_memory_mb - target_size_mb
            patterns_to_remove = int((excess_after_phase1 * 1024 * 1024) / bytes_per_vector)
            
            for pattern, _ in sorted_patterns[:patterns_to_remove]:
                del self.memory[pattern]
                # Cleanup pattern indexes
                for token in pattern:
                    self.pattern_index[token].discard(pattern)

        # Final cleanup
        self.prune_memory()
        final_stats = self.get_memory_stats()
        logger.info(f"Memory optimization complete. Final size: {final_stats.vector_memory_mb:.2f}MB. "
                   f"Removed: {removed_vectors} vectors, {patterns_to_remove if 'patterns_to_remove' in locals() else 0} patterns")

    def _generate_candidates(self, tokens: List[str]) -> List[List[str]]:
        """Gera variações candidatas para correspondência aproximada"""
        candidates = []
        
        # Gera combinações removendo 1 token por vez
        if len(tokens) > 1:
            for i in range(len(tokens)):
                candidate = tokens[:i] + tokens[i+1:]
                candidates.append(candidate)
                
        # Adiciona versão lematizada
        lemmatized = [self.lemmatize(token) for token in tokens]
        if lemmatized != tokens:
            candidates.append(lemmatized)
            
        return candidates

    def lemmatize(self, token: str) -> str:
        """Lematização básica (implementação de exemplo)"""
        lemmas = {
            'correndo': 'correr',
            'gatos': 'gato',
            'bonitas': 'bonito'
        }
        return lemmas.get(token, token)

    def _semantic_fallback(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Estratégia de fallback semântico usando vetores"""
        try:
            query_vector = self.vector_mgr.vectorize(tokens)
            similar = self.vector_mgr.find_similar(query_vector)
            
            if similar:
                best_match = max(similar, key=lambda x: x[1])
                return best_match[0], best_match[1] * 0.8
            return None
        except Exception as e:
            self.logger.error(f"Semantic fallback error: {str(e)}")
            return None

    def _calculate_learning_confidence(self, query: str, response: str) -> float:
        """Calculate confidence for learning new patterns"""
        if not query or not response:
            return 0.0
            
        # Basic confidence calculation
        query_length = len(self.tokenize(query))
        response_length = len(self.tokenize(response))
        
        if query_length < 2 or response_length < 2:
            return 0.0
            
        # Higher confidence for balanced query/response lengths
        length_ratio = min(query_length, response_length) / max(query_length, response_length)
        
        # Base confidence
        confidence = 0.5 + (length_ratio * 0.5)
        
        return min(confidence, 1.0)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text"""
        if not text:
            return []
            
        # Basic tokenization
        tokens = text.lower().split()
        return [token.strip() for token in tokens if token.strip()]

    def _normalize_score(self, score: float, max_score: float) -> float:
        """Safely normalize score"""
        if max_score <= 0:
            return 0.0
        return min(score / max_score, 1.0)

    def _calculate_match_confidence(self, pattern_length: int, matched_tokens: int) -> float:
        """Safely calculate match confidence"""
        if pattern_length <= 0:
            return 0.0
        return min(matched_tokens / pattern_length, 1.0)

    def _bootstrap_core_patterns(self) -> None:
        """Initialize core patterns and responses based on language"""
        try:
            # Define basic patterns by language
            patterns = {
                "en": {
                    "hello": {
                        "synonyms": {"hi", "hey", "howdy", "greetings"},
                        "response": "Hello! How can I help you today?"
                    },
                    "goodbye": {
                        "synonyms": {"bye", "cya", "farewell", "see you"},
                        "response": "Goodbye! Have a great day!"
                    },
                    "thanks": {
                        "synonyms": {"thank you", "thx", "appreciate it"},
                        "response": "You're welcome!"
                    }
                },
                "pt": {
                    "oi": {
                        "synonyms": {"olá", "ola", "eae", "e ai"},
                        "response": "Olá! Como posso ajudar?"
                    },
                    "tchau": {
                        "synonyms": {"adeus", "até logo", "ate logo", "falou"},
                        "response": "Tchau! Tenha um ótimo dia!"
                    },
                    "obrigado": {
                        "synonyms": {"obrigada", "valeu", "agradeço"},
                        "response": "De nada!"
                    }
                }
            }
            
            # Get patterns for the current language
            lang_patterns = patterns.get(self.language, patterns["en"])
            
            # Add patterns and their synonyms
            for word, data in lang_patterns.items():
                # Add synonyms
                self.synonyms[word].update(data["synonyms"])
                for synonym in data["synonyms"]:
                    self.synonyms[synonym].add(word)
                    self.synonyms[synonym].update(data["synonyms"] - {synonym})
                
                # Learn the pattern with high confidence
                pattern = frozenset({word})
                self.memory.add_pattern(
                    pattern=pattern,
                    response=data["response"],
                    confidence=0.9
                )
                
                # Add vector representation if enabled
                if self.use_vectors:
                    vector = self.vector_mgr.vectorize([word])
                    self.vector_mgr.add_vector(vector, data["response"])
            
            self.logger.info(f"Core patterns bootstrapped for language: {self.language}")
            
        except Exception as e:
            self.logger.error(f"Error bootstrapping core patterns: {str(e)}")
            # Continue initialization even if bootstrapping fails
            pass





