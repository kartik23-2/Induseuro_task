import math
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False

from ..models.question import Question
from ..config import settings
from ..utils.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class DenseEmbeddingEncoder:
    """
    Encodes text into dense semantic vector representations using SentenceTransformer
    with an intelligent fallback if offline or during local model download delays.
    """
    _instance = None
    _model = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dim = settings.EMBEDDING_DIMENSION
        self._init_encoder()

    def _init_encoder(self):
        if HAS_SENTENCE_TRANSFORMERS and self._model is None:
            try:
                # Load lightweight model locally
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic semantic projection.")
                self._model = None

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string into a normalized dense embedding vector.
        """
        if self._model is not None:
            try:
                embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Error encoding with SentenceTransformer: {e}. Falling back.")

        # Resilient fallback: Deterministic semantic character-ngram projection with cosine normalization
        return self._fallback_encode(text)

    def _fallback_encode(self, text: str) -> List[float]:
        """
        Deterministic, seed-consistent fallback vector encoder for environments
        without active internet access to HuggingFace hub.
        Captures n-gram overlap and sentence semantics into a unit vector.
        """
        norm_text = TextNormalizer.normalize(text)
        tokens = norm_text.split()
        vector = np.zeros(self.dim, dtype=np.float32)

        for i, token in enumerate(tokens):
            h = hash(token)
            idx = abs(h) % self.dim
            vector[idx] += 1.0 + (1.0 / (i + 1))
            # Sub-token bigrams for morphological awareness
            for j in range(len(token) - 2):
                tri = token[j:j+3]
                idx_tri = abs(hash(tri)) % self.dim
                vector[idx_tri] += 0.5

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


class VectorStore:
    """
    In-memory vector repository for storing questions, maintaining fast O(1) hash indices
    and computing cosine similarity for semantic duplicate identification.
    """

    def __init__(self):
        self.encoder = DenseEmbeddingEncoder()
        # Storage
        self._questions: Dict[str, Question] = {}  # id -> Question
        self._hash_index: Dict[str, str] = {}      # hash_key -> question_id

    def add(self, question: Question) -> Question:
        """
        Add a question to the store, populating embedding if not already present.
        """
        if not question.normalized_text:
            question.normalized_text = TextNormalizer.normalize(question.text)
        if not question.hash_key and question.normalized_text:
            question.hash_key = TextNormalizer.compute_hash(question.normalized_text)

        if question.embedding is None:
            question.embedding = self.encoder.encode(question.text)

        self._questions[question.id] = question
        if question.hash_key:
            self._hash_index[question.hash_key] = question.id

        return question

    def add_all(self, questions: List[Question]) -> List[Question]:
        """Add a batch of questions."""
        return [self.add(q) for q in questions]

    def get_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve question by ID."""
        return self._questions.get(question_id)

    def get_by_hash(self, hash_key: str) -> Optional[Question]:
        """Retrieve question by exact normalized hash key in O(1)."""
        qid = self._hash_index.get(hash_key)
        if qid:
            return self._questions.get(qid)
        return None

    def get_all(self) -> List[Question]:
        """Return all questions in the store."""
        return list(self._questions.values())

    def count(self) -> int:
        """Return total number of questions."""
        return len(self._questions)

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        tag_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Question, float]]:
        """
        Search for most semantically similar questions using cosine similarity.
        Optionally filter by tags (used for generation-time context selection).
        """
        if not self._questions:
            return []

        query_emb = np.array(self.encoder.encode(query_text), dtype=np.float32)
        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        results = []
        for q in self._questions.values():
            if tag_filter:
                match = all(q.tags.get(k) == v for k, v in tag_filter.items())
                if not match:
                    continue

            if q.embedding is None:
                q.embedding = self.encoder.encode(q.text)

            candidate_emb = np.array(q.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(candidate_emb)
            if c_norm > 0:
                candidate_emb = candidate_emb / c_norm

            sim = float(np.dot(query_emb, candidate_emb))
            results.append((q, sim))

        # Sort descending by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def clear(self):
        """Clear all stored questions and indices."""
        self._questions.clear()
        self._hash_index.clear()
