from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from .types import DuplicateStatus, DetectionTier


class Question(BaseModel):
    id: str
    text: str
    tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata tags (e.g. topic, grade, bloom_level)"
    )
    normalized_text: Optional[str] = None
    hash_key: Optional[str] = None
    embedding: Optional[List[float]] = None

    class Config:
        arbitrary_types_allowed = True


class DetectionResult(BaseModel):
    candidate_id: Optional[str] = None
    candidate_text: str
    status: DuplicateStatus
    tier: DetectionTier
    matched_question_id: Optional[str] = None
    matched_question_text: Optional[str] = None
    similarity_score: float = 0.0
    explanation: str = ""
