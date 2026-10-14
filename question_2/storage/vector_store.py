import os
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
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
    with class-level model caching, deterministic seeding, batch encoding,
    and a seed-stable deterministic fallback encoder.
    """
    _cached_model: Optional[Any] = None
    _cached_model_name: Optional[str] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dim = settings.EMBEDDING_DIMENSION
        self._init_encoder()

    def _init_encoder(self):
        # Set deterministic seeds for reproducibility
        np.random.seed(42)
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                import torch
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)
            except Exception:
                pass

        if HAS_SENTENCE_TRANSFORMERS:
            if DenseEmbeddingEncoder._cached_model is None or DenseEmbeddingEncoder._cached_model_name != self.model_name:
                try:
                    logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
                    model = SentenceTransformer(self.model_name)
                    if hasattr(model, "eval"):
                        model.eval()
                    DenseEmbeddingEncoder._cached_model = model
                    DenseEmbeddingEncoder._cached_model_name = self.model_name
                    logger.info(f"Successfully loaded and cached SentenceTransformer: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic fallback.")
                    DenseEmbeddingEncoder._cached_model = None

    @property
    def model(self):
        return DenseEmbeddingEncoder._cached_model

    def encode(self, text: str) -> List[float]:
        """
        Encode a single text string into a normalized dense embedding vector.
        """
        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    embedding = self.model.encode(
                        text,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                return embedding.astype(np.float32).tolist()
            except Exception as e:
                logger.warning(f"Error encoding with SentenceTransformer: {e}. Falling back.")

        return self._fallback_encode(text)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts in a single high-throughput vectorized batch.
        """
        if not texts:
            return []

        if self.model is not None:
            try:
                import torch
                with torch.no_grad():
                    embeddings = self.model.encode(
                        texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=32,
                        show_progress_bar=False
                    )
                return [emb.astype(np.float32).tolist() for emb in embeddings]
            except Exception as e:
                logger.warning(f"Error in batch encode with SentenceTransformer: {e}. Falling back.")

        return [self._fallback_encode(t) for t in texts]

    def _fallback_encode(self, text: str) -> List[float]:
        """
        Deterministic, seed-independent fallback vector encoder using SHA-256 tokens
        and character trigrams. Guaranteed identical outputs across processes, runs, and machines.
        """
        norm_text = TextNormalizer.normalize(text)
        tokens = norm_text.split()
        vector = np.zeros(self.dim, dtype=np.float32)

        for i, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], byteorder="big") % self.dim
            vector[idx] += 1.0 + (1.0 / (i + 1))

            # Sub-token character trigrams for morphological awareness
            for j in range(len(token) - 2):
                tri = token[j:j+3]
                tri_digest = hashlib.sha256(tri.encode("utf-8")).digest()
                idx_tri = int.from_bytes(tri_digest[:4], byteorder="big") % self.dim
                vector[idx_tri] += 0.5

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


class VectorStore:
    """
    High-performance vector store powered by native FAISS (IndexFlatIP)
    with exact cosine similarity inner-product search, O(1) canonical
    hash lookup, inverted token indexing for sub-millisecond lexical filtering,
    and seamless tag-filtered nearest-neighbor retrieval.
    """

    def __init__(self, encoder: Optional[DenseEmbeddingEncoder] = None):
        self.encoder = encoder or DenseEmbeddingEncoder()
        self.dim = self.encoder.dim

        # In-memory primary storage: question_id -> Question
        self._questions: Dict[str, Question] = {}
        # O(1) canonical hash index: hash_key -> question_id
        self._hash_index: Dict[str, str] = {}
        # Inverted token index for fast Tier 2 lexical overlap candidate pre-filtering
        self._token_index: Dict[str, Set[str]] = {}

        # FAISS index mapping
        self._faiss_id_to_qid: List[str] = []
        self._init_faiss_index()

    def _init_faiss_index(self):
        """
        Initializes the FAISS index with inner-product distance metric.
        For L2-normalized vectors: <u, v> == cos(theta).
        """
        if HAS_FAISS:
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            self._faiss_id_to_qid = []
        else:
            self._faiss_index = None
            self._faiss_id_to_qid = []
            logger.warning("FAISS is not installed. Falling back to NumPy vector operations.")

    def add(self, question: Question) -> Question:
        """
        Add a question to the store, compute embedding if absent, and index into FAISS.
        """
        if not question.normalized_text:
            question.normalized_text = TextNormalizer.normalize(question.text)
        if not question.hash_key and question.normalized_text:
            question.hash_key = TextNormalizer.compute_hash(question.normalized_text)

        if question.embedding is None:
            question.embedding = self.encoder.encode(question.text)

        # Update primary dictionary
        self._questions[question.id] = question

        # Update canonical hash index
        if question.hash_key:
            self._hash_index[question.hash_key] = question.id

        # Update inverted token index
        tokens = set(question.normalized_text.split())
        for token in tokens:
            if token not in self._token_index:
                self._token_index[token] = set()
            self._token_index[token].add(question.id)

        # Index into FAISS
        self._add_to_faiss(question)

        return question

    def add_all(self, questions: List[Question]) -> List[Question]:
        """
        Batch add questions, utilizing high-throughput batch vector encoding
        and vectorized batch insertion into FAISS.
        """
        if not questions:
            return []

        unencoded_indices = []
        unencoded_texts = []

        for idx, q in enumerate(questions):
            if not q.normalized_text:
                q.normalized_text = TextNormalizer.normalize(q.text)
            if not q.hash_key and q.normalized_text:
                q.hash_key = TextNormalizer.compute_hash(q.normalized_text)

            if q.embedding is None:
                unencoded_indices.append(idx)
                unencoded_texts.append(q.text)

        # Batch encode missing embeddings
        if unencoded_texts:
            encoded_batch = self.encoder.encode_batch(unencoded_texts)
            for orig_idx, emb in zip(unencoded_indices, encoded_batch):
                questions[orig_idx].embedding = emb

        # Batch insert into FAISS
        vectors = []
        for q in questions:
            self._questions[q.id] = q
            if q.hash_key:
                self._hash_index[q.hash_key] = q.id

            tokens = set(q.normalized_text.split())
            for token in tokens:
                if token not in self._token_index:
                    self._token_index[token] = set()
                self._token_index[token].add(q.id)

            vectors.append(q.embedding)
            self._faiss_id_to_qid.append(q.id)

        if HAS_FAISS and self._faiss_index is not None and vectors:
            vec_arr = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vec_arr)
            self._faiss_index.add(vec_arr)

        return questions

    def _add_to_faiss(self, question: Question):
        """Insert a single vector into the FAISS index."""
        if not HAS_FAISS or self._faiss_index is None:
            return

        vec = np.array([question.embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self._faiss_index.add(vec)
        self._faiss_id_to_qid.append(question.id)

    def get_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve question by ID in O(1)."""
        return self._questions.get(question_id)

    def get_by_hash(self, hash_key: str) -> Optional[Question]:
        """Retrieve question by exact normalized hash key in O(1)."""
        qid = self._hash_index.get(hash_key)
        if qid:
            return self._questions.get(qid)
        return None

    def get_lexical_candidates(self, query_text: str, max_candidates: int = 50) -> List[Question]:
        """
        Uses inverted token index to retrieve questions sharing at least one token with query,
        avoiding full O(N) corpus scans during Tier 2 lexical evaluation.
        """
        norm_query = TextNormalizer.normalize(query_text)
        query_tokens = norm_query.split()
        if not query_tokens:
            return []

        matched_qids: Set[str] = set()
        for token in query_tokens:
            qids = self._token_index.get(token)
            if qids:
                matched_qids.update(qids)
                if len(matched_qids) >= max_candidates:
                    break

        return [self._questions[qid] for qid in matched_qids if qid in self._questions]

    def get_all(self) -> List[Question]:
        """Return all questions in the store."""
        return list(self._questions.values())

    def count(self) -> int:
        """Return total number of questions."""
        return len(self._questions)

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        tag_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Question, float]]:
        """
        Search for most semantically similar questions using a pre-computed vector
        via FAISS inner-product (cosine) search. Supports tag filtering and fallback.
        """
        if not self._questions:
            return []

        query_emb = np.array([query_vector], dtype=np.float32)

        if HAS_FAISS and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            faiss.normalize_L2(query_emb)

            # If tag filtering, search a wider window to ensure we collect top_k matches after filtering
            k_search = self.count() if tag_filter else min(top_k * 2, self.count())
            k_search = max(1, min(k_search, self._faiss_index.ntotal))

            distances, indices = self._faiss_index.search(query_emb, k_search)

            results: List[Tuple[Question, float]] = []
            seen_qids: Set[str] = set()

            for score, int_id in zip(distances[0], indices[0]):
                if int_id < 0 or int_id >= len(self._faiss_id_to_qid):
                    continue
                qid = self._faiss_id_to_qid[int(int_id)]
                if qid in seen_qids:
                    continue
                seen_qids.add(qid)

                q = self._questions.get(qid)
                if not q:
                    continue

                if tag_filter:
                    match = all(q.tags.get(k) == v for k, v in tag_filter.items())
                    if not match:
                        continue

                results.append((q, float(score)))
                if len(results) >= top_k:
                    break

            return results

        # Fallback NumPy Cosine Similarity (when FAISS is disabled or unavailable)
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

            sim = float(np.dot(query_emb[0], candidate_emb))
            results.append((q, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_similar(
        self,
        query_text: str,
        top_k: int = 5,
        tag_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Question, float]]:
        """
        Search for most semantically similar questions using text query.
        """
        if not self._questions:
            return []
        query_vec = self.encoder.encode(query_text)
        return self.search_by_vector(query_vector=query_vec, top_k=top_k, tag_filter=tag_filter)

    def clear(self):
        """Clear all stored questions and indices."""
        self._questions.clear()
        self._hash_index.clear()
        self._token_index.clear()
        self._init_faiss_index()
