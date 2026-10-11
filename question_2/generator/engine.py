import json
import logging
from typing import Dict, Any, Optional, Tuple, List

from ..models.question import Question, DetectionResult
from ..models.schema_types import DuplicateStatus
from ..storage.vector_store import VectorStore
from ..detector.cascade import CascadeDuplicateDetector
from ..generator.context_selector import ContextSelector
from ..llm.client import LLMClient
from ..config import settings

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generation-Time Avoidance Engine that steers the LLM toward producing genuinely novel
    questions and validates each candidate through the independent duplicate detector.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        detector: Optional[CascadeDuplicateDetector] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.store = vector_store
        self.llm = llm_client or LLMClient()
        self.detector = detector or CascadeDuplicateDetector(vector_store=self.store, llm_client=self.llm)
        self.context_selector = ContextSelector(vector_store=self.store)

    def generate_question(
        self,
        topic: str,
        tags: Optional[Dict[str, Any]] = None,
        desired_cognitive_level: str = "Understand / Apply",
        auto_ingest: bool = False
    ) -> Tuple[Optional[Question], DetectionResult, Dict[str, Any]]:
        """
        Generates a novel question using context avoidance constraints and verifies
        it against the independent duplicate detector with an automated retry loop.
        """
        tags = tags or {}
        attempt_history: List[Dict[str, Any]] = []

        # 1. Build initial context avoiding context limits
        avoidance_ctx = self.context_selector.build_avoidance_context(
            topic=topic,
            tag_filter={"topic": tags.get("topic")} if "topic" in tags else None,
            max_exemplars=5
        )

        feedback_additions = ""

        for attempt in range(1, settings.MAX_GENERATION_RETRIES + 1):
            prompt = self._build_generation_prompt(
                topic=topic,
                tags=tags,
                cognitive_level=desired_cognitive_level,
                avoidance_instructions=avoidance_ctx["avoidance_instructions"],
                feedback_additions=feedback_additions
            )

            response_raw = self.llm.generate(
                prompt=prompt,
                system_prompt="You are an expert curriculum developer who creates rigorous, non-redundant assessment questions.",
                temperature=settings.GENERATION_TEMPERATURE,
                json_mode=True
            )

            candidate_text = self._extract_question_text(response_raw)
            logger.info(f"[Attempt {attempt}] Candidate generated: '{candidate_text}'")

            # 2. Gate through Independent Duplicate Detector
            detection_res = self.detector.detect(candidate_text=candidate_text)
            attempt_history.append({
                "attempt": attempt,
                "candidate_text": candidate_text,
                "status": detection_res.status.value,
                "tier": detection_res.tier.value,
                "matched_question": detection_res.matched_question_text,
                "score": detection_res.similarity_score,
                "explanation": detection_res.explanation
            })

            # 3. Check acceptance criteria
            if not detection_res.is_duplicate and not detection_res.requires_review:
                # Approved!
                new_q = Question(
                    text=candidate_text,
                    tags={**tags, "topic": topic, "cognitive_level": desired_cognitive_level}
                )
                if auto_ingest:
                    self.store.add(new_q)

                metadata = {
                    "attempts": attempt,
                    "attempt_history": attempt_history,
                    "avoidance_context_used": avoidance_ctx["negative_exemplars"],
                    "auto_ingested": auto_ingest
                }
                return new_q, detection_res, metadata

            # Rejected as duplicate: formulate negative feedback for next attempt
            feedback_additions += (
                f"\n[REJECTION FEEDBACK from Attempt {attempt}]:\n"
                f"- Candidate '{candidate_text}' was rejected as a duplicate of: '{detection_res.matched_question_text}'.\n"
                f"- Reason: {detection_res.explanation}\n"
                f"- MANDATORY: You must switch to a completely different subtopic or numerical problem.\n"
            )

        # If all retries failed, return None for question but provide final rejection result
        # pyrefly: ignore [unbound-name]
        return None, detection_res, {
            "attempts": settings.MAX_GENERATION_RETRIES,
            "attempt_history": attempt_history,
            "failed_reason": "Exceeded max generation retries due to persistent duplicate generation."
        }

    def _build_generation_prompt(
        self,
        topic: str,
        tags: Dict[str, Any],
        cognitive_level: str,
        avoidance_instructions: str,
        feedback_additions: str
    ) -> str:
        tags_str = ", ".join(f"{k}: {v}" for k, v in tags.items()) if tags else "None specified"
        return f"""[QUESTION GENERATION TASK]
Topic: {topic}
Target Cognitive Level: {cognitive_level}
Metadata Tags: {tags_str}

{avoidance_instructions}
{feedback_additions}

Output JSON format:
{{
  "question": "Exact text of the newly formulated question",
  "target_concept": "Specific physical concept or principle tested",
  "cognitive_level": "{cognitive_level}",
  "novelty_justification": "Why this question tests a concept distinct from existing questions"
}}
"""

    def _extract_question_text(self, response_str: str) -> str:
        """Extract question text from JSON response or raw string."""
        clean = response_str.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        try:
            data = json.loads(clean)
            if isinstance(data, dict) and "question" in data:
                return data["question"].strip()
        except Exception:
            pass

        return clean.strip().strip('"')
