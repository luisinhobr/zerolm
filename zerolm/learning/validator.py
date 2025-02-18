"""Module containing LearningValidator."""

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
