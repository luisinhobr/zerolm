from typing import Dict
import numpy as np

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
