import re
import time
import random
import logging
import pickle
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Set, Tuple, Any, Generator
import os
import json
from heapq import nlargest
from collections import deque
from threading import Lock
import jellyfish
import math
from collections import Counter
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LEARNING = "learning"
    ERROR = "error"
    UNKNOWN = "unknown"
    NORMAL = "normal"

@dataclass
class Response:
    text: str
    confidence: float
    type: ResponseType
    metadata: Optional[Dict[str, Any]] = None
    context: Optional[List[str]] = None
    
    def explain(self) -> str:
        """Generate human-readable explanation of the response"""
        explanation = [f"Response: {self.text}"]
        explanation.append(f"Type: {self.type.value}")
        explanation.append(f"Confidence: {self.confidence:.2f}")
        
        if self.vector_contribution and self.token_contribution:
            explanation.append(
                f"Match source: {self.vector_contribution:.1%} vector, "
                f"{self.token_contribution:.1%} token"
            )
        
        return "\n".join(explanation)

class ValidationResult:
    def __init__(self, passed: bool, reason: Optional[str] = None):
        self.passed = passed
        self.reason = reason

@dataclass
class MemoryStats:
    """Statistics about model memory usage"""
    pattern_count: int
    token_count: int
    vector_memory_mb: float
    confidence_histogram: Dict[str, int]
    recent_usage: List[Tuple[str, float]]

class ConcurrentLRUCache:
    """Thread-safe LRU cache implementation"""
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.maxsize = maxsize
        self.lock = Lock()
        self.access_times = defaultdict(float)
        
    def get(self, key: Any) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
            
    def put(self, key: Any, value: Any) -> None:
        with self.lock:
            if len(self.cache) >= self.maxsize:
                lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_key]
                del self.access_times[lru_key]
            self.cache[key] = value
            self.access_times[key] = time.time()

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

class AdaptiveMemoryManager:
    def __init__(
        self,
        max_patterns: int = 10000,
        hot_ratio: float = 0.1,
        warm_ratio: float = 0.3
    ):
        self.max_patterns = max_patterns
        self.hot_capacity = int(max_patterns * hot_ratio)
        self.warm_capacity = int(max_patterns * warm_ratio)
        self.cold_capacity = max_patterns - (self.hot_capacity + self.warm_capacity)
        
        self.hot_storage = {}
        self.warm_storage = {}
        self.cold_storage = {}

    def __iter__(self):
        """Make memory manager iterable"""
        for storage in [self.hot_storage, self.warm_storage, self.cold_storage]:
            yield from storage.items()

    def __len__(self) -> int:
        """Total patterns across all storage tiers"""
        return (
            len(self.hot_storage) + 
            len(self.warm_storage) + 
            len(self.cold_storage)
        )

    def items(self):
        """Combine items from all storage tiers"""
        return list(self.hot_storage.items()) + \
               list(self.warm_storage.items()) + \
               list(self.cold_storage.items())

    def add_pattern(self, pattern: frozenset, response: str, confidence: float) -> bool:
        """Add pattern to appropriate storage tier based on confidence"""
        try:
            entry = {
                'response': response,
                'confidence': confidence,
                'timestamp': time.time(),
                'access_count': 0
            }
            
            # Determine storage tier based on confidence
            if confidence >= 0.9:
                if len(self.hot_storage) >= self.hot_capacity:
                    # Move oldest hot pattern to warm
                    oldest = min(self.hot_storage.items(), key=lambda x: x[1]['timestamp'])
                    self.warm_storage[oldest[0]] = oldest[1]
                    del self.hot_storage[oldest[0]]
                self.hot_storage[pattern] = entry
            elif confidence >= 0.7:
                if len(self.warm_storage) >= self.warm_capacity:
                    # Move oldest warm pattern to cold
                    oldest = min(self.warm_storage.items(), key=lambda x: x[1]['timestamp'])
                    self.cold_storage[oldest[0]] = oldest[1]
                    del self.warm_storage[oldest[0]]
                self.warm_storage[pattern] = entry
            else:
                if len(self.cold_storage) >= self.cold_capacity:
                    # Remove oldest cold pattern
                    oldest = min(self.cold_storage.items(), key=lambda x: x[1]['timestamp'])
                    del self.cold_storage[oldest[0]]
                self.cold_storage[pattern] = entry
                
            return True
            
        except Exception as e:
            logging.error(f"Error adding pattern: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            # Calculate total patterns
            total_patterns = len(self.hot_storage) + len(self.warm_storage) + len(self.cold_storage)
            
            # Calculate storage distribution
            storage_dist = {
                "hot": len(self.hot_storage),
                "warm": len(self.warm_storage),
                "cold": len(self.cold_storage)
            }
            
            # Calculate confidence distribution
            confidence_dist = {
                "high": 0,   # >= 0.9
                "medium": 0, # 0.7-0.9
                "low": 0,    # 0.5-0.7
                "very_low": 0 # < 0.5
            }
            
            # Count patterns by confidence
            for storage in [self.hot_storage, self.warm_storage, self.cold_storage]:
                for pattern_data in storage.values():
                    conf = pattern_data.get('confidence', 0.0)
                    if conf >= 0.9:
                        confidence_dist["high"] += 1
                    elif conf >= 0.7:
                        confidence_dist["medium"] += 1
                    elif conf >= 0.5:
                        confidence_dist["low"] += 1
                    else:
                        confidence_dist["very_low"] += 1
            
            # Get recent patterns (from hot storage)
            recent_patterns = sorted(
                [
                    (pattern, data.get('timestamp', 0), data.get('confidence', 0))
                    for pattern, data in self.hot_storage.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Get top 10 most recent
            
            return {
                "total_patterns": total_patterns,
                "storage_distribution": storage_dist,
                "confidence_distribution": confidence_dist,
                "recent_patterns": [
                    {
                        "pattern": " ".join(pattern),
                        "confidence": conf,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
                    }
                    for pattern, ts, conf in recent_patterns
                ]
            }
            
        except Exception as e:
            logging.error(f"Error getting memory stats: {str(e)}")
            return {
                "total_patterns": 0,
                "storage_distribution": {"hot": 0, "warm": 0, "cold": 0},
                "confidence_distribution": {
                    "high": 0, "medium": 0, "low": 0, "very_low": 0
                },
                "recent_patterns": []
            }

class PatternMatcher:
    """Advanced pattern matching with multiple similarity metrics"""
    def __init__(
        self,
        min_confidence: float = 0.3,
        use_fuzzy: bool = True
    ):
        self.min_confidence = min_confidence
        self.use_fuzzy = use_fuzzy

    @staticmethod
    def enhanced_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        euclidean = 1 / (1 + np.linalg.norm(vec1 - vec2))
        manhattan = 1 / (1 + np.sum(np.abs(vec1 - vec2)))
        return 0.5*cosine_sim + 0.3*euclidean + 0.2*manhattan

    def ngram_match(self, tokens: List[str], n: int = 3) -> List[str]:
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def hybrid_match(self, query_tokens: List[str], pattern: frozenset) -> float:
        token_overlap = len(set(query_tokens) & pattern) / len(pattern)
        query_ngrams = set(self.ngram_match(query_tokens))
        pattern_ngrams = set(self.ngram_match(list(pattern)))
        ngram_overlap = len(query_ngrams & pattern_ngrams) / len(pattern_ngrams)
        return 0.7*token_overlap + 0.3*ngram_overlap

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

class LearningValidator:
    """Validador especializado para operações de aprendizagem"""
    def validate(self, query: str, response: str, memory) -> bool:
        """Valida se o novo aprendizado é válido"""
        # Verifica requisitos mínimos
        if len(query) < 3 or len(response) < 3:
            return False
            
        # Verifica similaridade com padrões existentes
        existing_patterns = memory.items()
        similarity_scores = [
            self._calculate_similarity(query, pattern)
            for pattern, _ in existing_patterns
        ]
        
        # Rejeita se for muito similar a padrão existente
        if max(similarity_scores) > 0.85:
            return False
            
        return True

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade usando Jaro-Winkler"""
        return jellyfish.jaro_winkler_similarity(text1.lower(), text2.lower())

class TemporalWeightingSystem:
    """Universal temporal relevance integration"""
    def __init__(self):
        self.base_weights = {
            'domain': 0.5,  # Reduced base weight to accommodate temporal factor
            'temporal': 0.3,
            'user': 0.2,
            'recency': 0.4  # New temporal dimension
        }
        self.decay_rate = 0.01  # Daily knowledge decay

    def calculate_weighted_score(self, query: str, user_data: Dict, 
                               context: Dict) -> float:
        """Integrated temporal weighting calculation"""
        time_factor = self._calculate_time_factor(context.get('timestamp'))
        recency_score = self._calculate_recency(context.get('last_accessed'))
        
        return (
            self._domain_score(query) * self._apply_temporal(self.base_weights['domain'], time_factor) +
            self._temporal_score() * self.base_weights['temporal'] +
            self._user_score(user_data) * self._apply_temporal(self.base_weights['user'], time_factor) +
            recency_score * self.base_weights['recency']
        )

    def _apply_temporal(self, base_weight: float, time_factor: float) -> float:
        """Apply temporal modulation to any weight"""
        return base_weight * (0.8 + 0.2 * time_factor)

    def _calculate_time_factor(self, timestamp: float) -> float:
        """Time-sensitive importance decay"""
        hours_old = (time.time() - timestamp) / 3600
        return max(0, 1 - self.decay_rate * hours_old)

    def _calculate_recency(self, last_accessed: float) -> float:
        """Recency-based scoring with exponential decay"""
        days_since = (time.time() - last_accessed) / 86400
        return math.exp(-0.5 * days_since)

class ContextWeighter:
    """Enhanced with temporal relevance at multiple levels"""
    def __init__(self):
        self.temporal_system = TemporalWeightingSystem()
        self.seasonal_factors = self._load_seasonal_patterns()

    def calculate_context_score(self, query: str, user_data: Dict, 
                              context: Dict) -> float:
        # Get base temporal score
        temporal_score = self.temporal_system.calculate_weighted_score(
            query, user_data, context
        )
        
        # Apply seasonal adjustment
        return temporal_score * self._seasonal_adjustment_factor()

    def _seasonal_adjustment_factor(self) -> float:
        """Time-of-year adjustment based on historical patterns"""
        now = datetime.now()
        return self.seasonal_factors.get((now.month, now.weekday()), 1.0)

    def _load_seasonal_patterns(self) -> Dict[Tuple[int, int], float]:
        # Example seasonal adjustment pattern
        return {
            (12, 0): 1.2,  # December weekdays
            (7, 4): 0.8,    # July weekends
            # ... other seasonal patterns
        }

class AutoCorrector:
    """Implementação completa do sistema de autocorreção"""
    def __init__(self):
        self.correction_layers = [
            {
                'priority': 1,
                'type': 'structural',
                'actions': [
                    self.correct_namespace,
                    self.enforce_component_order,
                    self.validate_numeric_constraints,
                    self.verify_required_elements
                ]
            },
            {
                'priority': 2,
                'type': 'semantic',
                'thresholds': {
                    'value_correction': 0.8,
                    'context_adjustment': 0.7
                }
            }
        ]
        
        # Definir a ordem correta dos componentes
        self.component_order = [
            'context_analysis',
            'validation_hierarchy',
            'attribute_constraints',
            'processing_pipeline'
        ]

    def enforce_component_order(self, xml_tree) -> Dict:
        """Garante a ordem correta dos componentes conforme o template"""
        errors = {}
        current_order = [elem.tag for elem in xml_tree]
        
        for i, expected_tag in enumerate(self.component_order):
            if expected_tag in current_order:
                position = current_order.index(expected_tag)
                if position < i:
                    errors[expected_tag] = {
                        'action': 'move',
                        'from': position,
                        'to': i
                    }
        
        if errors:
            return {
                'name': 'component_order',
                'modified': True,
                'details': errors
            }
        return {
            'name': 'component_order',
            'modified': False
        }

    def validate_numeric_constraints(self, xml_tree) -> Dict:
        """Valida restrições numéricas do template"""
        constraints = {
            './/param[@name]': {
                'min': 0,
                'max': 1,
                'decimal_places': 2
            }
        }
        
        errors = []
        for xpath, rules in constraints.items():
            for element in xml_tree.findall(xpath):
                try:
                    value = float(element.get('value', 0))
                    if not (rules['min'] <= value <= rules['max']):
                        errors.append(f"Valor {value} fora do intervalo em {element.tag}")
                    if len(str(value).split('.')[-1]) > rules['decimal_places']:
                        errors.append(f"Precisão excessiva em {element.tag}")
                except ValueError:
                    errors.append(f"Valor não numérico em {element.tag}")
        
        return {
            'name': 'numeric_constraints',
            'modified': bool(errors),
            'details': errors
        }

    def verify_required_elements(self, xml_tree) -> Dict:
        """Verifica elementos obrigatórios do template"""
        required_elements = [
            'context_analysis/weighting_system',
            'validation_hierarchy/layer',
            'attribute_constraints/param'
        ]
        
        missing = []
        for xpath in required_elements:
            if not xml_tree.find(xpath):
                missing.append(xpath)
        
        return {
            'name': 'required_elements',
            'modified': bool(missing),
            'details': missing
        }

    # Mantém a implementação existente dos outros métodos
    def correct_namespace(self, xml_tree) -> Dict:
        """Corrige namespace conforme template"""
        if 'xmlns' not in xml_tree.attrib:
            xml_tree.set('xmlns', 'template_v3')
            return {
                'name': 'namespace',
                'modified': True
            }
        return {
            'name': 'namespace',
            'modified': False
        }

class TemplateEnforcer:
    """Enforces template compliance rules"""
    def __init__(self):
        self.allowed_tags = {
            'agent_definition', 'processing_pipeline', 'compliance_system',
            'context_analysis', 'validation_hierarchy', 'attribute_constraints'
        }
        
    def validate_compliance(self, xml_tree) -> List[str]:
        """Enforce template's structural compliance rules"""
        errors = []
        
        # Check for prohibited content
        for element in xml_tree.iter():
            if element.tag not in self.allowed_tags:
                errors.append(f"Prohibited element: {element.tag}")
                
            if element.text and element.text.strip():
                if not any(child.tag in self.allowed_tags for child in element):
                    errors.append(f"Free text block in {element.tag}")
        
        # Validate attribute formats
        for param in xml_tree.iterfind('.//param'):
            if not re.match(r'^0\.\d{1,2}$', param.get('value', '')):
                errors.append(f"Invalid value format: {param.get('value')}")
                
        return errors

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
        """Recupera tokens contextuais da janela de contexto"""
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
            
            # Add pattern to memory
            success = self.memory.add_pattern(pattern, {
                'response': response,
                'confidence': confidence,
                'timestamp': time.time(),
                'access_count': 0
            })
            
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
            if not pattern or not tokens:
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

    def prune_memory(self, max_items: Optional[int] = None, min_confidence: float = 0.2):
        """Clean up memory with default values from config"""
        max_items = max_items or self.config.get("max_patterns", 10000)
        # Score patterns by recency and usage
        pattern_scores = {
            pattern: self._calculate_pattern_score(pattern)
            for pattern in self.memory.items()
        }
        
        # Keep top patterns and those above confidence threshold
        keep_patterns = set(nlargest(max_items, pattern_scores, key=pattern_scores.get))
        keep_patterns.update(
            p for p, score in pattern_scores.items() 
            if score >= min_confidence
        )
        
        # Prune memory and indexes
        self.memory = {
            p: v for p, v in self.memory.items()
            if p in keep_patterns
        }
        self._rebuild_indexes()

    def _calculate_pattern_score(self, pattern: frozenset) -> float:
        """Calculate pattern retention score"""
        recency = max(ts for _, ts in self.memory[pattern]) / time.time()
        frequency = len(self.memory[pattern])
        return (0.7 * recency) + (0.3 * frequency)

    def _rebuild_indexes(self):
        """Reconstruct pattern indexes after pruning"""
        self.pattern_index.clear()
        for pattern in self.memory.keys():
            for token in pattern:
                self.pattern_index[token].add(pattern)

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
                raise ValueError("Vector decay must be ≥0.9 and <1")
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

class PerformanceMetrics:
    """Comprehensive performance tracking"""
    def __init__(self):
        self.response_times = []
        self.error_log = []
        self.cache_stats = {'hits': 0, 'misses': 0}
        
    def log_time(self, duration: float):
        self.response_times.append(duration)
        
    def log_error(self, error: str):
        self.error_log.append(error)
        
    def get_stats(self) -> Dict:
        return {
            'avg_response_time': np.mean(self.response_times[-100:]),
            'error_rate': len(self.error_log),
            'cache_hit_rate': self.cache_stats['hits'] / 
                             (self.cache_stats['hits'] + self.cache_stats['misses'])
        }

class TemplateValidator:
    """Implements the structural validation from the template"""
    def __init__(self):
        self.required_nodes = [
            ('context_analysis/weighting_system', 0.7),
            ('validation_hierarchy/layer[@type]', 0.9),
            ('attribute_constraints/param[@name]', 0.8)
        ]
        
    def validate_structure(self, xml_tree) -> dict:
        """Validação robusta com tratamento de erros"""
        try:
            errors = []
            for xpath, _ in self.required_nodes:
                if not xml_tree.find(xpath):
                    errors.append(f"Elemento obrigatório ausente: {xpath}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'confidence': 1.0 - (len(errors) * 0.1)
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Erro de validação: {str(e)}"],
                'confidence': 0.0
            }

    def enforce_attribute_rules(self, xml_tree) -> List[str]:
        """Enforces attribute format rules from the template"""
        errors = []
        for param in xml_tree.iterfind('.//param'):
            if param.get('name') and not re.match(r'^param_\d+$', param.get('name')):
                errors.append(f"Invalid parameter name: {param.get('name')}")
            if param.get('value') and not re.match(r'^0\.\d{1,2}$', param.get('value')):
                errors.append(f"Invalid value format: {param.get('value')}")
        return errors

class KnowledgeDistiller:
    """Distilador de conhecimento integrado com os componentes do ZeroShotLM"""
    
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
        """Fluxo principal de destilação integrado com o modelo"""
        vector_resp = self.source_models['primary']['handler'](query)
        pattern_resp = self.source_models['secondary']['handler'](query)
        
        if self._responses_aligned(vector_resp, pattern_resp):
            return self._merge_responses(vector_resp, pattern_resp, context)
        return self._select_response(vector_resp, pattern_resp)

    def _handle_vector_model(self, query: str) -> Response:
        """Processamento vetorial usando a infraestrutura existente"""
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
            text="Não encontrei correspondência vetorial relevante.",
            type=ResponseType.LEARNING,
            confidence=0.0
        )

    def _handle_pattern_model(self, query: str) -> Response:
        """Correspondência de padrões usando a memória existente"""
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
            text="Padrão não reconhecido.",
            type=ResponseType.LEARNING,
            confidence=0.0
        )

    def _responses_aligned(self, resp1: Response, resp2: Response) -> bool:
        """Verifica alinhamento considerando contexto e semântica"""
        return (
            abs(resp1.confidence - resp2.confidence) < 0.2 and
            self._semantic_similarity(resp1.text, resp2.text) > 0.7
        )

    def _merge_responses(self, resp1: Response, resp2: Response, context: dict) -> Response:
        """Combina respostas usando pesos contextuais"""
        time_diff = abs(resp1.metadata['timestamp'] - resp2.metadata['timestamp'])
        recency_weight = math.exp(-0.1 * (time_diff / 3600))  # Decaimento por hora
        
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
        """Seleção baseada em confiança e contexto temporal"""
        time_diff = resp2.metadata['timestamp'] - resp1.metadata['timestamp']
        recency_bonus = 0.1 * (1 - math.exp(-time_diff/3600))  # Bônus de 10% para respostas mais recentes
        
        adjusted_confidence = {
            'vector': resp1.confidence + (recency_bonus if time_diff < 0 else 0),
            'pattern': resp2.confidence + (recency_bonus if time_diff > 0 else 0)
        }
        
        if adjusted_confidence['vector'] >= adjusted_confidence['pattern']:
            return resp1
        return resp2

    # Métodos auxiliares integrados com o restante do sistema
    def _tokenize(self, text: str) -> List[str]:
        return [word.lower() for word in text.split() if len(word) > 2]

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        vec1 = self.vector_mgr.get_query_vector(self._tokenize(text1))
        vec2 = self.vector_mgr.get_query_vector(self._tokenize(text2))
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _get_candidate_patterns(self, tokens: List[str]) -> List[frozenset]:
        """Obtém padrões candidatos de todas as camadas de memória"""
        candidates = []
        for storage in [self.memory.hot_storage,
                        self.memory.warm_storage,
                        self.memory.cold_storage]:
            candidates.extend(
                p for p in storage.keys()
                if any(t in p for t in tokens)
            )
        return sorted(candidates, key=lambda x: len(x), reverse=True)[:100]
