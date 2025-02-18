from typing import Dict, Tuple
from datetime import datetime
from zerolm.context import TemporalWeightingSystem

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
