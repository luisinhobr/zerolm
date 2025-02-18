from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from zerolm.types import ResponseType

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