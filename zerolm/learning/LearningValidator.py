import jellyfish

class LearningValidator:
    """Specialized validator for learning operations"""
    def validate(self, query: str, response: str, memory) -> bool:
        """Validates if the new learning is valid"""
        # Check minimum requirements
        if len(query) < 3 or len(response) < 3:
            return False
            
        # Check similarity with existing patterns
        existing_patterns = memory.items()
        similarity_scores = [
            self._calculate_similarity(query, pattern)
            for pattern, _ in existing_patterns
        ]
        
        # Reject if too similar to existing pattern
        if max(similarity_scores) > 0.85:
            return False
            
        return True

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using Jaro-Winkler"""
        return jellyfish.jaro_winkler_similarity(text1.lower(), text2.lower())
