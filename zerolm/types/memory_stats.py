from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class MemoryStats:
    """Statistics about model memory usage"""
    pattern_count: int
    token_count: int
    vector_memory_mb: float
    confidence_histogram: Dict[str, int]
    recent_usage: List[Tuple[str, float]]
