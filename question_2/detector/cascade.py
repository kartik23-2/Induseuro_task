import re
import json
import logging
from typing import Optional, List, Tuple
from ..models.question import Question, DetectionResult
from ..models.schema_types import DetectionTier, DuplicateStatus
from ..utils.normalizer import TextNormalizer
from ..storage.vector_store import VectorStore
from ..llm.client import LLMClient
from ..config import settings

logger = logging.getLogger(__name__)


class CascadeDuplicateDetector:
    """
    Independent multi-tier cascading detector that evaluates a candidate question
    against the existing question repository. Operates independently of generation logic.
    """

    def __init__(self, vector_store: VectorStore, llm_client: Optional[LLMClient] = None):
        self.store = vector_store
        self.llm = llm_client or LLMClient()
        self.exact_thresh = 1.0
        self.lexical_thresh = settings.LEXICAL_DUP_THRESHOLD
        self.semantic_thresh = settings.SEMANTIC_DUP_THRESHOLD
        self.unique_thresh = settings.UNIQUE_THRESHOLD

    @staticmethod
    def _has_conflicting_discriminators(text_a: str, text_b: str) -> bool:
        """
        Detects if two questions contain mutually conflicting discriminators:
        - Contrasting numerical values (e.g. 40° vs 65°)
        - Contrasting ordinals (e.g. first vs second vs third)
        - Contrasting scientific terms (e.g. mitosis vs meiosis, reflection vs refraction)
        If present, automated shortcutting is suppressed in favor of Tier 4 LLM intent analysis.
        """
        a_lower = text_a.lower()
        b_lower = text_b.lower()

        # 1. Number extraction
        nums_a = set(re.findall(r'\b\d+(?:\.\d+)?\b', a_lower))
        nums_b = set(re.findall(r'\b\d+(?:\.\d+)?\b', b_lower))
        if nums_a and nums_b and nums_a != nums_b:
            return True

        # 2. Ordinal conflicts
        ordinals = ["first", "second", "third", "fourth", "1st", "2nd", "3rd", "4th"]
        present_a = {o for o in ordinals if re.search(rf'\b{o}\b', a_lower)}
        present_b = {o for o in ordinals if re.search(rf'\b{o}\b', b_lower)}
        if present_a and present_b and present_a != present_b:
            return True

        # 3. Contrasting entity pairs
        contrasting_pairs = [
            ("reflection", "refraction"),
            ("specular", "diffuse"),
            ("concave", "convex"),
            ("mitosis", "meiosis"),
            ("stack", "queue")
        ]
        for term1, term2 in contrasting_pairs:
            if (term1 in a_lower and term2 in b_lower) or (term2 in a_lower and term1 in b_lower):
                return True

        return False

    def detect(self, candidate_text: str, candidate_id: Optional[str] = None) -> DetectionResult:
        """
        Evaluate a candidate question through the 4-tier cascade:
        Tier 1: Canonical Normalization & Hash Match (O(1))
        Tier 2: Lexical Token / Jaccard Overlap Filter (<1ms via Inverted Index)
        Tier 3: Dense Semantic Vector Cosine Match (~1-10ms via FAISS IndexFlatIP)
        Tier 4: LLM Concept & Intent Judge (for borderline/ambiguous cases)
        """
        norm_text = TextNormalizer.normalize(candidate_text)
        hash_key = TextNormalizer.compute_hash(norm_text)

        # -------------------------------------------------------------
        # Tier 1: Canonical Hash Lookup (O(1))
        # -------------------------------------------------------------
        exact_match = self.store.get_by_hash(hash_key)
        if exact_match and (candidate_id is None or exact_match.id != candidate_id):
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=DuplicateStatus.EXACT_DUPLICATE,
                tier=DetectionTier.TIER_1_EXACT_HASH,
                matched_question_id=exact_match.id,
                matched_question_text=exact_match.text,
                similarity_score=1.0,
                explanation="Identical canonical match after case, Unicode, contraction, and punctuation normalization."
            )

        # If store has no questions, candidate is definitely unique
        if self.store.count() == 0:
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=DuplicateStatus.UNIQUE,
                tier=DetectionTier.NONE,
                similarity_score=0.0,
                explanation="Corpus is empty; question is unique."
            )

        # -------------------------------------------------------------
        # Tier 2: Lexical / Token Jaccard Filter (via Inverted Token Index)
        # -------------------------------------------------------------
        # Retrieve candidate questions sharing at least 1 token instead of scanning entire collection
        candidate_subset = self.store.get_lexical_candidates(candidate_text, max_candidates=100)
        # If corpus is small (<50 items), check all items as safety net
        if self.store.count() < 50:
            candidate_subset = self.store.get_all()

        max_jaccard = 0.0
        lexical_match: Optional[Question] = None

        for q in candidate_subset:
            if candidate_id and q.id == candidate_id:
                continue
            j_score = TextNormalizer.jaccard_similarity(candidate_text, q.text)
            if j_score > max_jaccard:
                max_jaccard = j_score
                lexical_match = q

        if max_jaccard >= self.lexical_thresh and lexical_match:
            # Check for conflicting numbers or discriminators before shortcutting
            if not self._has_conflicting_discriminators(candidate_text, lexical_match.text):
                return DetectionResult(
                    candidate_id=candidate_id,
                    candidate_text=candidate_text,
                    status=DuplicateStatus.SEMANTIC_DUPLICATE,
                    tier=DetectionTier.TIER_2_LEXICAL,
                    matched_question_id=lexical_match.id,
                    matched_question_text=lexical_match.text,
                    similarity_score=max_jaccard,
                    explanation=f"High lexical token overlap (Jaccard: {max_jaccard:.2f}) with existing question."
                )

        # -------------------------------------------------------------
        # Tier 3: Dense Semantic Vector Cosine Similarity (via FAISS)
        # -------------------------------------------------------------
        similar_items = self.store.search_similar(candidate_text, top_k=5)
        # Filter out self if evaluating already ingested item
        filtered_items = [item for item in similar_items if item[0].id != candidate_id]

        if not filtered_items:
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=DuplicateStatus.UNIQUE,
                tier=DetectionTier.TIER_3_DENSE_SEMANTIC,
                similarity_score=0.0,
                explanation="No semantic neighbors found in corpus."
            )

        top_match, top_sim = filtered_items[0]

        # Case 3A: Very high semantic similarity -> Duplicate
        # Shortcut ONLY if there are no conflicting discriminators (e.g. first law vs second law)
        if top_sim >= self.semantic_thresh and not self._has_conflicting_discriminators(candidate_text, top_match.text):
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=DuplicateStatus.SEMANTIC_DUPLICATE,
                tier=DetectionTier.TIER_3_DENSE_SEMANTIC,
                matched_question_id=top_match.id,
                matched_question_text=top_match.text,
                similarity_score=top_sim,
                explanation=f"High dense semantic vector similarity ({top_sim:.4f}) indicates a paraphrase."
            )

        # Case 3B: Low similarity -> Definitely unique
        if top_sim < self.unique_thresh:
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=DuplicateStatus.UNIQUE,
                tier=DetectionTier.TIER_3_DENSE_SEMANTIC,
                matched_question_id=top_match.id,
                matched_question_text=top_match.text,
                similarity_score=top_sim,
                explanation=f"Top semantic similarity ({top_sim:.4f}) is below uniqueness threshold ({self.unique_thresh})."
            )

        # -------------------------------------------------------------
        # Tier 4: LLM Intent & Pedagogical Judge
        # Invoked strictly for boundary zone [0.65, 0.88) or ambiguous scopes
        # -------------------------------------------------------------
        return self._evaluate_tier_4(candidate_text, candidate_id, top_match, top_sim)

    def _evaluate_tier_4(
        self,
        candidate_text: str,
        candidate_id: Optional[str],
        top_match: Question,
        top_sim: float
    ) -> DetectionResult:
        """
        Deep pedagogical and intent comparison using the LLM Judge.
        """
        system_prompt = (
            "You are an expert assessment examiner and duplicate question judge. "
            "Your task is to analyze whether two questions are duplicates, test the same concept, "
            "or are pedagogically distinct."
        )

        prompt = f"""[TIER 4 DUPLICATE JUDGE]
Evaluate whether Candidate Question A and Existing Question B are semantic duplicates.

Candidate Question A: "{candidate_text}"
Existing Question B: "{top_match.text}"
Baseline Vector Cosine Similarity: {top_sim:.4f}

Analyze the following:
1. Core Target Knowledge Element: Do they test the exact same concept or definition?
2. Expected Student Answer: Would an acceptable answer to Question A satisfy Question B?
3. Cognitive Task: Do they require the same cognitive level (e.g., recall vs. application vs. numerical calculation)?
4. Ambiguity / Breadth: Is one question so broad or underspecified (e.g., general 'reflection' vs 'optical reflection') that context review is required?

Respond in valid JSON with keys:
- "verdict": One of ["SEMANTIC_DUPLICATE", "UNIQUE", "AMBIGUOUS_REVIEW_NEEDED"]
- "confidence": Float between 0.0 and 1.0
- "target_knowledge_overlap": One of ["IDENTICAL", "PARTIAL", "LOW"]
- "explanation": Brief explanation (max 2 sentences)
"""

        try:
            response_str = self.llm.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
            # Clean JSON markdown fences or formatting if present
            clean_json = response_str.strip()
            if "```" in clean_json:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_json)
                if match:
                    clean_json = match.group(1).strip()

            parsed = json.loads(clean_json)
            verdict_raw = parsed.get("verdict", "UNIQUE").upper()
            explanation = parsed.get("explanation", "Evaluated by Tier 4 LLM Intent Judge.")

            if "EXACT" in verdict_raw:
                status = DuplicateStatus.EXACT_DUPLICATE
            elif "SEMANTIC" in verdict_raw:
                status = DuplicateStatus.SEMANTIC_DUPLICATE
            elif "AMBIGUOUS" in verdict_raw:
                status = DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED
            else:
                status = DuplicateStatus.UNIQUE

            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=status,
                tier=DetectionTier.TIER_4_INTENT_JUDGE,
                matched_question_id=top_match.id,
                matched_question_text=top_match.text,
                similarity_score=top_sim,
                explanation=f"[Tier 4 Judge]: {explanation}"
            )

        except Exception as e:
            logger.error(f"Error in Tier 4 LLM judge: {e}. Defaulting to conservative status.")
            # Fallback based on vector score
            status = DuplicateStatus.SEMANTIC_DUPLICATE if top_sim >= 0.75 else DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED
            return DetectionResult(
                candidate_id=candidate_id,
                candidate_text=candidate_text,
                status=status,
                tier=DetectionTier.TIER_4_INTENT_JUDGE,
                matched_question_id=top_match.id,
                matched_question_text=top_match.text,
                similarity_score=top_sim,
                explanation=f"Tier 4 fallback: {e}"
            )
