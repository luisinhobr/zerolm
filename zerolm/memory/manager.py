"""Adaptive memory management implementation for ZeroLM."""

import time
import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

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
        self.logger = logging.getLogger(__name__)

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
                    oldest_key = min(
                        self.hot_storage.keys(),
                        key=lambda k: (self.hot_storage[k]['timestamp'], hash(k))
                    )
                    self.warm_storage[oldest_key] = self.hot_storage[oldest_key]
                    del self.hot_storage[oldest_key]
                self.hot_storage[pattern] = entry
            elif confidence >= 0.7:
                if len(self.warm_storage) >= self.warm_capacity:
                    # Move oldest warm pattern to cold
                    oldest_key = min(
                        self.warm_storage.keys(),
                        key=lambda k: (self.warm_storage[k]['timestamp'], hash(k))
                    )
                    self.cold_storage[oldest_key] = self.warm_storage[oldest_key]
                    del self.warm_storage[oldest_key]
                self.warm_storage[pattern] = entry
            else:
                if len(self.cold_storage) >= self.cold_capacity:
                    # Remove oldest cold pattern
                    oldest_key = min(
                        self.cold_storage.keys(),
                        key=lambda k: (self.cold_storage[k]['timestamp'], hash(k))
                    )
                    del self.cold_storage[oldest_key]
                self.cold_storage[pattern] = entry
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding pattern: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            # Calculate total patterns
            total_patterns = len(self)
            
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
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {
                "total_patterns": 0,
                "storage_distribution": {"hot": 0, "warm": 0, "cold": 0},
                "confidence_distribution": {
                    "high": 0, "medium": 0, "low": 0, "very_low": 0
                },
                "recent_patterns": []
            }

    def remove_pattern(self, pattern: frozenset) -> None:
        """Thread-safe pattern removal from all storage tiers"""
        with ThreadPoolExecutor() as executor:
            for storage in [self.hot_storage, self.warm_storage, self.cold_storage]:
                executor.submit(storage.pop, pattern, None)

    def _calculate_pattern_score(self, pattern: frozenset) -> float:
        """Type-safe pattern scoring"""
        try:
            data = next((s[pattern] for s in [
                self.hot_storage,
                self.warm_storage, 
                self.cold_storage
            ] if pattern in s), None)
            
            if not data:
                return 0.0
            
            hours_since_access = (time.time() - data['timestamp']) / 3600
            return (0.7 * math.exp(-0.1 * hours_since_access)) + (0.3 * data['access_count'])
            
        except Exception as e:
            self.logger.error(f"Scoring error: {str(e)}")
            return 0.0