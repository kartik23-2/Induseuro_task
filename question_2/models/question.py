import uuid
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict, model_validator

try:
    from .schema_types import DuplicateStatus, DetectionTier
    from ..utils.normalizer import TextNormalizer
except (ImportError, ValueError):
    # Allow direct execution: python question_2/models/question.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from question_2.models.schema_types import DuplicateStatus, DetectionTier
    from question_2.utils.normalizer import TextNormalizer


class Question(BaseModel):
    """
    Representation of an assessment question with content and arbitrary metadata tags.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for the question"
    )
    text: str = Field(..., description="The original raw text of the question")
    tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata tags (e.g. topic, grade, bloom_level)"
    )
    normalized_text: Optional[str] = Field(
        default=None,
        description="Canonicalized lowercase text with punctuation stripped for exact matching"
    )
    hash_key: Optional[str] = Field(
        default=None,
        description="MD5 hash of normalized text for O(1) exact duplicate lookup"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Dense vector embedding of the question text"
    )

    @model_validator(mode="after")
    def populate_normalized_and_hash(self) -> "Question":
        """
        Automatically populate normalized_text and hash_key if text is provided
        and they have not been manually set.
        """
        if self.text:
            if not self.normalized_text:
                self.normalized_text = TextNormalizer.normalize(self.text)
            if not self.hash_key and self.normalized_text:
                self.hash_key = TextNormalizer.compute_hash(self.normalized_text)
        return self


class DetectionResult(BaseModel):
    """
    Result of evaluating a candidate question against existing question corpus.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate_id: Optional[str] = None
    candidate_text: str
    status: DuplicateStatus
    tier: DetectionTier
    matched_question_id: Optional[str] = None
    matched_question_text: Optional[str] = None
    similarity_score: float = 0.0
    explanation: str = ""

    @property
    def is_duplicate(self) -> bool:
        """Returns True if the candidate was flagged as exact or semantic duplicate."""
        return self.status in (DuplicateStatus.EXACT_DUPLICATE, DuplicateStatus.SEMANTIC_DUPLICATE)

    @property
    def requires_review(self) -> bool:
        """Returns True if candidate requires human or secondary review."""
        return self.status == DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED


if __name__ == "__main__":
    q = Question(text="What is reflection of light?")
    print(f"Sample Question Created: ID={q.id}, Normalized='{q.normalized_text}'")
