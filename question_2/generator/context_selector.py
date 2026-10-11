import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..models.question import Question
from ..storage.vector_store import VectorStore
from ..config import settings

logger = logging.getLogger(__name__)


class ContextSelector:
    """
    Manages generation context to avoid exceeding LLM context limits while providing
    effective negative guidance against existing questions in the collection.
    """

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def build_avoidance_context(
        self,
        topic: str,
        tag_filter: Optional[Dict[str, Any]] = None,
        max_exemplars: int = 5
    ) -> Dict[str, Any]:
        """
        Selects representative existing questions using MMR (Maximal Marginal Relevance)
        and semantic proximity to the target topic, avoiding full-corpus context overflow.

        Returns a dictionary containing:
        - 'topic': the target topic
        - 'negative_exemplars': Top-k diverse questions already in the bank
        - 'existing_count': Total questions matching topic in the bank
        - 'avoidance_instructions': Specific negative constraints for the prompt
        """
        all_topic_questions = self.store.search_similar(
            query_text=topic,
            top_k=settings.TOP_K_RETRIEVAL,
            tag_filter=tag_filter
        )

        total_matching = len(all_topic_questions)
        if total_matching == 0:
            return {
                "topic": topic,
                "negative_exemplars": [],
                "existing_count": 0,
                "avoidance_instructions": "No existing questions on this topic in the bank. You may explore fundamental definitions and core concepts."
            }

        # Select diverse subset using greedy MMR (Maximal Marginal Relevance)
        selected_exemplars = self._select_diverse_subset(all_topic_questions, k=max_exemplars)

        instructions = (
            f"The question bank currently contains {total_matching} questions related to '{topic}'. "
            f"DO NOT generate questions that duplicate the wording, concept, or required answer of the following existing questions:\n"
        )
        for idx, q in enumerate(selected_exemplars, 1):
            instructions += f"  [{idx}] {q.text}\n"

        instructions += (
            "\nRequirements for genuine novelty:\n"
            "- Target an unaddressed subtopic, real-world application, or quantitative numerical problem.\n"
            "- Vary the Bloom's Taxonomy cognitive level (e.g. Analysis or Evaluation if definitions already exist).\n"
            "- Ensure the question cannot be answered by the same statement as any existing question above."
        )

        return {
            "topic": topic,
            "negative_exemplars": [q.text for q in selected_exemplars],
            "existing_count": total_matching,
            "avoidance_instructions": instructions
        }

    def _select_diverse_subset(
        self,
        scored_candidates: List[Any],
        k: int = 5
    ) -> List[Question]:
        """
        Select k diverse questions from candidates to cover distinct concept clusters
        rather than redundant variations of the same definition.
        """
        if len(scored_candidates) <= k:
            return [cand[0] for cand in scored_candidates]

        selected: List[Question] = []
        candidates = [cand[0] for cand in scored_candidates]

        # 1. Pick the most central question first
        selected.append(candidates[0])
        remaining = candidates[1:]

        # 2. Iteratively pick candidates that maximize diversity (minimal cosine similarity to already selected)
        while len(selected) < k and remaining:
            best_cand = None
            lowest_max_sim = float("inf")

            for cand in remaining:
                cand_vec = np.array(cand.embedding, dtype=np.float32) if cand.embedding else None
                if cand_vec is None:
                    continue

                # Max similarity to any already selected
                max_sim_to_selected = -1.0
                for s in selected:
                    s_vec = np.array(s.embedding, dtype=np.float32)
                    sim = float(np.dot(cand_vec, s_vec))
                    if sim > max_sim_to_selected:
                        max_sim_to_selected = sim

                if max_sim_to_selected < lowest_max_sim:
                    lowest_max_sim = max_sim_to_selected
                    best_cand = cand

            if best_cand:
                selected.append(best_cand)
                remaining.remove(best_cand)
            else:
                break

        return selected
