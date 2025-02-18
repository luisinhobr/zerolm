from enum import Enum

class ResponseType(Enum):
    EXACT = "exact"
    APPROXIMATE = "approximate"
    LEARNING = "learning"
    ERROR = "error"
    UNKNOWN = "unknown"
    NORMAL = "normal"