"""Module containing ConcurrentLRUCache."""

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
