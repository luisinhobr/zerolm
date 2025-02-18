"""Module containing PatternMatcher."""

from collections import Counter
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from heapq import nlargest
from scipy import stats
from threading import Lock
from typing import List, Optional, Dict, Set, Tuple, Any, Generator
import jellyfish
import json
import logging
import math
import numpy as np
import os
import pickle
import random
import re
import time

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
