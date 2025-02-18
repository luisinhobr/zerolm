from typing import Optional, Any
import time
from threading import Lock
from collections import defaultdict

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
