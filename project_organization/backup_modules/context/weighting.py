"""Module containing TemporalWeightingSystem, ContextWeighter."""

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
