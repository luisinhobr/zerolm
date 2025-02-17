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

@dataclass
class Response:
    text: str
    type: ResponseType
    confidence: float
    context: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    vector_contribution: Optional[float] = None
    token_contribution: Optional[float] = None
    
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
    def __init__(self, vector_dim: int, use_sparse: bool = True):
        self.vector_dim = vector_dim
        self.use_sparse = use_sparse
        self.vectors = {}
        
    def add_vector(self, key: str, vector: np.ndarray) -> None:
        if self.use_sparse:
            nonzero_idx = vector.nonzero()[0]
            nonzero_values = vector[nonzero_idx].astype(np.float32)
            self.vectors[key] = (nonzero_idx, nonzero_values)
        else:
            self.vectors[key] = vector.astype(np.float32)
            
    def get_vector(self, key: str) -> np.ndarray:
        if key not in self.vectors:
            return np.zeros(self.vector_dim)
            
        if self.use_sparse:
            vector = np.zeros(self.vector_dim, dtype=np.float32)
            idx, values = self.vectors[key]
            vector[idx] = values
            return vector
        return self.vectors[key]

class AdaptiveMemoryManager:
    """Memory management with adaptive thresholds"""
    def __init__(self):
        self.hot_storage = defaultdict(list)
        self.warm_storage = defaultdict(list)
        self.cold_storage = defaultdict(list)
        self.access_counts = defaultdict(int)
        self.usage_history = deque(maxlen=1000)
        
    def _calculate_thresholds(self) -> Tuple[float, float]:
        """Dynamic threshold calculation using z-scores"""
        if len(self.usage_history) < 100:
            return 10, 100  # Default thresholds
        
        counts = np.array(self.usage_history)
        mean = np.mean(counts)
        std = np.std(counts)
        
        warm_thresh = mean + std
        hot_thresh = mean + 2*std
        return warm_thresh, hot_thresh

    def add_pattern(self, pattern: frozenset, response: str, ts: float):
        count = self.access_counts[pattern]
        self.usage_history.append(count)
        
        warm_thresh, hot_thresh = self._calculate_thresholds()
        
        if count > hot_thresh:
            self.hot_storage[pattern].append((response, ts))
        elif count > warm_thresh:
            self.warm_storage[pattern].append((response, ts))
        else:
            self.cold_storage[pattern].append((response, ts))

class PatternMatcher:
    """Advanced pattern matching with multiple similarity metrics"""
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
    """Ensures consistency in learned patterns"""
    def validate(self, query: str, response: str, memory) -> bool:
        existing = self._find_related_patterns(query, memory)
        if not existing:
            return True
            
        similarity = max(self._semantic_similarity(response, r) 
                        for r in existing)
        return similarity >= 0.3

    def _semantic_similarity(self, a: str, b: str) -> float:
        # Implement basic semantic similarity measure
        a_words = set(a.split())
        b_words = set(b.split())
        return len(a_words & b_words) / len(a_words | b_words)

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
    """Implements the template's auto-correction rules"""
    def __init__(self):
        self.correction_layers = [
            {
                'priority': 1,
                'type': 'structural',
                'actions': [
                    self.correct_namespace,
                    self.enforce_component_order,
                    self.validate_numeric_constraints
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
        
    def apply_corrections(self, xml_tree, confidence: float) -> Dict:
        """Apply corrections based on template rules"""
        corrections = {}
        for layer in self.correction_layers:
            if layer['type'] == 'structural':
                for action in layer['actions']:
                    result = action(xml_tree)
                    if result['modified']:
                        corrections[result['name']] = result
            elif layer['type'] == 'semantic' and confidence < 0.7:
                corrections.update(self._apply_semantic_corrections(xml_tree))
        return corrections

    def correct_namespace(self, xml_tree) -> Dict:
        """Namespace correction per template rules"""
        # Implementation matching template's example
        if 'xmlns' not in xml_tree.attrib:
            xml_tree.set('xmlns', 'template_v3')
            return {'name': 'namespace', 'modified': True}
        return {'name': 'namespace', 'modified': False}

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
        use_vectors: bool = False,
        vector_dim: int = 100,
        min_confidence: float = 0.3,
        temporal_weight: float = 0.7,
        context_window: int = 5,
        language: str = "pt-BR",
        max_patterns: int = 10_000,
        learning_rate: float = 0.1,
        vector_decay: float = 0.99,
        batch_size: int = 100,
        cleanup_interval: int = 3600,
        max_memory_mb: float = 500,
        use_sparse_vectors: bool = True,
        workers: int = 4
    ):
        # Core memory structures
        self.memory = AdaptiveMemoryManager()
        self.synonyms = defaultdict(set)
        self.context_window: List[Tuple[List[str], float]] = []
        
        # Configuration
        self.config = {
            "min_confidence": min_confidence,
            "temporal_weight": temporal_weight,
            "context_window": context_window,
            "language": language
        }
        
        # Initial learning responses
        self.learning_responses = self._initialize_learning_responses()
        
        # Initialize basic patterns
        self._bootstrap_core_patterns()
        
        # Vector-based components
        self.use_vectors = use_vectors
        self.vector_dim = vector_dim
        self.vector_mgr = VectorManager(vector_dim, use_sparse_vectors)
        self.vector_index = defaultdict(set)  # {token: set(frozenset patterns)}
        
        # Performance optimizations
        self.pattern_index = defaultdict(set)  # token -> set(frozenset patterns)
        self.recent_patterns = deque(maxlen=max_patterns)
        self.vector_cache = ConcurrentLRUCache(maxsize=1000)
        
        # Learning controls
        self.learning_rate = learning_rate
        self.vector_decay = vector_decay
        self.batch_buffer = []
        self.batch_size = batch_size
        
        # Memory management
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.pattern_usage = defaultdict(int)
        self.token_usage = defaultdict(int)
        
        # Enhanced components
        self.matcher = PatternMatcher()
        self.context = HierarchicalContext()
        self.validator = LearningValidator()
        self.workers = workers
        self.metrics = PerformanceMetrics()

        self.validator = TemplateValidator()
        self.distiller = KnowledgeDistiller()
        self.weighter = ContextWeighter()
        self.corrector = AutoCorrector()
        self.enforcer = TemplateEnforcer()

    def process_query(self, query: str, xml_template: str) -> Response:
        # Validate template structure
        validation_results = self.validator.validate_structure(xml_template)
        
        # Apply auto-corrections
        corrections = self.corrector.apply_corrections(xml_template)
        
        # Calculate context weights
        context_score = self.weighter.calculate_context_score(query)
        
        # Distill knowledge from multiple sources
        response = self.distiller.distill_response(query, context_score)
        
        # Enforce compliance
        compliance_errors = self.enforcer.validate_compliance(xml_template)
        
        return self._format_response(response, compliance_errors)

    def process_query(self, query: str) -> Response:
        """Process input query and generate response"""
        try:
            if time.time() - self.last_cleanup > self.cleanup_interval:
                self.prune_memory()
                self.last_cleanup = time.time()
                
            return self._process_query_core(query)
        except Exception as e:
            logger.error(f"Query processing error: {str(e)}")
            return self._create_learning_response("Processing error")

    def _process_query_core(self, query: str) -> Response:
        """Core query processing logic"""
        tokens = self._tokenize(query)
        if not tokens:
            return self._create_learning_response("Empty query")
            
        self.context.update(tokens)
        response_data = self._generate_response(tokens)
        
        if response_data:
            text, confidence = response_data
            return Response(
                text=text,
                type=self._determine_response_type(confidence),
                confidence=confidence,
                context=self._get_current_context(),
                metadata=self._create_metadata(tokens, confidence)
            )
            
        return self._create_learning_response()

    def learn(self, query: str, response: str) -> bool:
        """Enhanced learning with validation"""
        if not self.validator.validate(query, response, self.memory):
            return False
            
        tokens = self._tokenize(query)
        pattern = frozenset(tokens)
        
        # Update vectors in batch
        vecs = [self.vector_mgr.get_vector(t) for t in tokens]
        query_vec = np.mean(vecs, axis=0)
        for t, v in zip(tokens, vecs):
            new_vec = 0.9*v + 0.1*query_vec
            self.vector_mgr.add_vector(t, new_vec)
            
        self.memory.add_pattern(pattern, response, time.time())
        return True

    def _update_usage_stats(self, pattern: frozenset, tokens: List[str]):
        """Track pattern and token usage"""
        self.pattern_usage[pattern] += 1
        for token in tokens:
            self.token_usage[token] += 1

    def _update_vectors(self, tokens: List[str]):
        """Update vector representations"""
        query_vector = self._vectorize(tokens)
        for token in tokens:
            self.vector_mgr.add_vector(token, (
                self.vector_decay * self.vector_mgr.get_vector(token) + 
                (1 - self.vector_decay) * query_vector
            ))
            self._validate_vector(self.vector_mgr.get_vector(token))

    def _store_pattern(self, pattern: frozenset, response: str):
        """Store pattern with timestamp"""
        current_time = time.time()
        self.memory.add_pattern(pattern, response, current_time)
        self.recent_patterns.append(pattern)
        self._update_indexes(pattern, list(pattern))

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\sáéíóúâêîôûãõç]', '', text)
        return [word for word in text.split() if word]

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

    def _generate_response(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Enhanced response generation pipeline"""
        # Try exact match first
        if response := self._exact_match(tokens):
            return response
            
        # Vector-based match
        if response := self._vector_match(tokens):
            return response
            
        # Contextual partial match
        if response := self._context_match(tokens):
            return response
            
        # Semantic fallback
        return self._semantic_fallback(tokens)

    def _exact_match(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Try to generate response with confidence score"""
        # Check exact matches
        if matches := self.memory.hot_storage.get(frozenset(tokens)):
            response, timestamp = self._select_best_match(matches)
            confidence = self._calculate_confidence(tokens, response, timestamp)
            return response, confidence
        
        # Try expanded token matching
        expanded_tokens = self._expand_tokens(tokens)
        for candidate_tokens in self._generate_candidates(expanded_tokens):
            if matches := self.memory.hot_storage.get(frozenset(candidate_tokens)):
                response, timestamp = self._select_best_match(matches)
                confidence = self._calculate_confidence(candidate_tokens, response, timestamp) * 0.8
                if confidence >= self.config["min_confidence"]:
                    return response, confidence
        
        return None

    def _vector_match(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Optimized vector matching with caching"""
        cache_key = frozenset(tokens)
        if cached := self.vector_cache.get(cache_key):
            return cached
            
        query_vector = self._vectorize(tokens)
        matches = self._find_vector_matches(query_vector)
        
        if matches:
            best_match = matches[0]
            self.vector_cache.put(cache_key, best_match)
            return best_match
            
        return None

    def _context_match(self, tokens: List[str]) -> Optional[Tuple[str, float]]:
        """Context-aware partial matching"""
        context_tokens = self._get_context()
        token_scores = self._calculate_token_importance(tokens, context_tokens)
        
        candidates = []
        for pattern in self._get_candidate_patterns(tokens):
            score = sum(token_scores.get(t, 0) for t in pattern)
            if score > 0.5:
                responses = self.memory.hot_storage.get(pattern)
                if responses:
                    best_response = max(responses, key=lambda x: x[1])
                    confidence = score * self._temporal_decay(best_response[1])
                    candidates.append((best_response[0], confidence))
        
        return max(candidates, key=lambda x: x[1]) if candidates else None

    def _calculate_token_importance(self, tokens: List[str], context: List[str]) -> Dict[str, float]:
        """TF-IDF inspired scoring"""
        scores = {}
        freq = Counter(context)
        total_patterns = sum(len(s) for s in [self.memory.hot_storage, 
                                             self.memory.warm_storage,
                                             self.memory.cold_storage])
        
        for token in tokens:
            pattern_count = len(self.pattern_index.get(token, []))
            idf = math.log(total_patterns / (pattern_count + 1)) if total_patterns > 0 else 0
            context_boost = 1 + (freq[token] * 0.1)
            scores[token] = idf * context_boost
            
        max_score = max(scores.values(), default=1)
        return {t: s/max_score for t, s in scores.items()}

    def _expand_tokens(self, tokens: List[str]) -> List[str]:
        """Expand tokens with synonyms and context"""
        expanded = set(tokens)
        
        # Add synonyms
        for token in tokens:
            expanded.update(self.synonyms[token])
        
        # Add relevant context tokens
        context_tokens = [t for tokens, _ in self.context_window for t in tokens]
        expanded.update(self._filter_relevant_tokens(context_tokens, tokens))
        
        return list(expanded)

    def _filter_relevant_tokens(self, context_tokens: List[str], query_tokens: List[str]) -> List[str]:
        """Filter context tokens based on relevance to query"""
        return [
            token for token in context_tokens
            if any(self._token_similarity(token, qt) > 0.7 for qt in query_tokens)
        ]

    def _token_similarity(self, token1: str, token2: str) -> float:
        """Enhanced similarity with multiple strategies"""
        if token1 == token2:
            return 1.0
            
        # Check synonyms
        if token1 in self.synonyms.get(token2, set()):
            return 0.9
            
        # Levenshtein similarity
        lev_dist = jellyfish.levenshtein_distance(token1, token2)
        max_len = max(len(token1), len(token2))
        if max_len > 0:
            lev_sim = 1 - (lev_dist / max_len)
            if lev_sim > 0.8:
                return lev_sim * 0.8
                
        # Phonetic similarity
        if jellyfish.metaphone(token1) == jellyfish.metaphone(token2):
            return 0.7
            
        return 0.0

    def _calculate_confidence(self, tokens: List[str], response: str, timestamp: float) -> float:
        """Calculate confidence score for a response"""
        # Base confidence from token match
        confidence = 1.0
        
        # Apply temporal decay
        time_diff = time.time() - timestamp
        temporal_factor = max(0.5, 1.0 - (time_diff / (3600 * 24 * 7)))  # 7 day decay
        confidence *= temporal_factor
        
        # Consider context overlap
        context_overlap = len(set(tokens) & set(t for ctx, _ in self.context_window for t in ctx))
        context_factor = 1.0 + (context_overlap * 0.1)
        confidence *= context_factor
        
        return min(1.0, confidence)

    def _determine_response_type(self, confidence: float) -> ResponseType:
        """Determine response type based on confidence"""
        if confidence >= 0.8:
            return ResponseType.EXACT
        elif confidence >= self.config["min_confidence"]:
            return ResponseType.APPROXIMATE
        return ResponseType.LEARNING

    def _create_learning_response(self, reason: str = None) -> Response:
        """Create response for learning mode"""
        text = random.choice(self.learning_responses)
        if reason:
            text = f"{text} ({reason})"
            
        return Response(
            text=text,
            type=ResponseType.LEARNING,
            confidence=0.0,
            context=self._get_current_context(),
            metadata={"reason": reason} if reason else None
        )

    def _initialize_learning_responses(self) -> List[str]:
        """Initialize learning mode responses"""
        if self.config["language"] == "pt-BR":
            return [
                "Não sei responder isso ainda. Pode me ensinar?",
                "Interessante! Como você responderia isso?",
                "Ainda estou aprendendo sobre isso. Qual seria uma boa resposta?",
                "Não tenho certeza. Poderia me explicar?",
                "Gostaria de aprender como responder isso corretamente."
            ]
        return [
            "I don't know how to answer that yet. Can you teach me?",
            "Interesting! How would you answer this?",
            "I'm still learning about this. What would be a good response?",
            "I'm not sure. Could you explain?",
            "I'd like to learn how to answer this correctly."
        ]

    def _bootstrap_core_patterns(self):
        """Initialize basic language patterns"""
        if self.config["language"] == "pt-BR":
            patterns = {
                "saudacao": {"olá", "oi", "bom dia", "boa tarde", "boa noite"},
                "despedida": {"tchau", "adeus", "até logo", "até mais"},
                "concordancia": {"sim", "claro", "certamente", "exato"},
                "discordancia": {"não", "nunca", "jamais", "incorreto"}
            }
        else:
            patterns = {
                "greeting": {"hello", "hi", "good morning", "good afternoon", "good evening"},
                "farewell": {"goodbye", "bye", "see you", "farewell"},
                "agreement": {"yes", "sure", "certainly", "exactly"},
                "disagreement": {"no", "never", "incorrect", "wrong"}
            }
            
        for pattern_type, words in patterns.items():
            for word in words:
                self.synonyms[word].update(words - {word})

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

    def prune_memory(self, max_items: int = None, min_confidence: float = 0.2):
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
            self.recent_patterns = deque(self.recent_patterns, maxlen=max_patterns)

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

    def get_memory_stats(self) -> MemoryStats:
        """Detailed memory statistics"""
        # Calculate memory usage components
        vector_memory = len(self.vector_mgr.vectors) * self.vector_dim * 4  # 4 bytes per float32
        pattern_count = len(self.memory)
        
        # Calculate token statistics
        all_tokens = set()
        for pattern in self.memory:
            all_tokens.update(pattern)
        token_count = len(all_tokens)
        
        # Build confidence histogram
        conf_bins = defaultdict(int)
        for responses in self.memory.values():
            for response, _ in responses:
                if response.confidence >= 0.9:
                    conf_bins["0.9-1.0"] += 1
                elif response.confidence >= 0.7:
                    conf_bins["0.7-0.89"] += 1
                elif response.confidence >= 0.5:
                    conf_bins["0.5-0.69"] += 1
                else:
                    conf_bins["<0.5"] += 1
                    
        # Get recent usage patterns (last 24 hours)
        recent_usage = []
        cutoff = time.time() - 86400
        for pattern, responses in self.memory.items():
            latest_ts = max(ts for _, ts in responses)
            if latest_ts > cutoff:
                recent_usage.append((" ".join(pattern), latest_ts))
        
        return MemoryStats(
            pattern_count=pattern_count,
            token_count=token_count,
            vector_memory_mb=vector_memory / 1e6,
            confidence_histogram=dict(conf_bins),
            recent_usage=sorted(recent_usage, key=lambda x: -x[1])[:5]  # Top 5 recent
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
        
    def validate_structure(self, xml_tree) -> Dict:
        """Implements the structural validation rules"""
        results = {}
        for xpath, min_confidence in self.required_nodes:
            node = xml_tree.find(xpath)
            exists = node is not None
            results[xpath] = {
                'exists': exists,
                'confidence': min_confidence if exists else 0.0,
                'error': None if exists else f"Missing required node: {xpath}"
            }
        return results

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
    """Temporal-aware knowledge distillation"""
    def __init__(self):
        self.source_models = {
            'primary': {
                'name': 'vector_similarity',
                'capability_rating': 0.85,
                'handler': self._handle_vector_model
            },
            'secondary': {
                'name': 'pattern_matching', 
                'optimization_profile': 'efficiency',
                'handler': self._handle_pattern_model
            }
        }
        
    def distill_response(self, query: str, context: Dict) -> Response:
        """Main distillation workflow"""
        vector_resp = self.source_models['primary']['handler'](query)
        pattern_resp = self.source_models['secondary']['handler'](query)
        
        if self._responses_aligned(vector_resp, pattern_resp):
            return self._merge_responses(vector_resp, pattern_resp)
        return self._select_response(vector_resp, pattern_resp)

    def _responses_aligned(self, resp1: Response, resp2: Response) -> bool:
        """Check response alignment per template rules"""
        return abs(resp1.confidence - resp2.confidence) < 0.2 and \
               resp1.type == resp2.type

    def _merge_responses(self, resp1: Response, resp2: Response) -> Response:
        """Time-weighted response merging"""
        time_diff = abs(resp1.metadata['timestamp'] - resp2.metadata['timestamp'])
        recency_weight = math.exp(-0.1 * time_diff/3600)  # Hourly decay
        
        if recency_weight > 0.7:
            merged_text = f"{resp1.text} ({resp2.text})"
        else:
            merged_text = resp1.text if resp1.confidence > resp2.confidence else resp2.text
            
        merged_conf = (resp1.confidence * recency_weight + 
                      resp2.confidence * (1 - recency_weight))
        
        return Response(merged_text, resp1.type, merged_conf)

class MemoryManager:
    """Temporal memory management with decay"""
    def _apply_temporal_decay(self):
        """Decay less recently used patterns"""
        current_time = time.time()
        for pattern in list(self.memory.keys()):
            last_used = self.memory[pattern]['last_accessed']
            days_since = (current_time - last_used) / 86400
            
            # Apply exponential decay to confidence
            self.memory[pattern]['confidence'] *= math.exp(-0.1 * days_since)
            
            # Remove decayed patterns
            if self.memory[pattern]['confidence'] < 0.1:
                del self.memory[pattern]

class LearningValidator:
    """Temporal validation of learned patterns"""
    def validate_learning(self, query: str, response: str) -> bool:
        # Check temporal consistency with recent patterns
        recent_patterns = self._get_recent_patterns(hours=24)
        if not self._check_temporal_consistency(response, recent_patterns):
            return False
        return True

    def _check_temporal_consistency(self, response: str, 
                                   recent_patterns: List) -> bool:
        # Check if response aligns with recent knowledge
        return any(
            self._semantic_similarity(response, r) > 0.7
            for r in recent_patterns
        )