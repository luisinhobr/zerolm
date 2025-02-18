"""Module containing ResponseType, Response, ValidationResult, MemoryStats."""

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

class ResponseType(Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LEARNING = "learning"
    ERROR = "error"
    UNKNOWN = "unknown"
    NORMAL = "normal"

@dataclass

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
