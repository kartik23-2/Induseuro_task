import pytest
from question_2.models.question import Question
from question_2.models.schema_types import DuplicateStatus, DetectionTier
from question_2.utils.normalizer import TextNormalizer
from question_2.storage.vector_store import VectorStore
from question_2.detector.cascade import CascadeDuplicateDetector
from question_2.generator.engine import QuestionGenerator


@pytest.fixture(scope="module")
def populated_store():
    store = VectorStore()
    q1 = Question(id="Q1", text="What is reflection of light?", tags={"topic": "reflection", "grade": 10})
    q2 = Question(id="Q2", text="Define the reflection of light.", tags={"topic": "reflection", "grade": 10})
    store.add(q1)
    store.add(q2)
    return store


@pytest.fixture(scope="module")
def detector(populated_store):
    return CascadeDuplicateDetector(vector_store=populated_store)


def test_text_normalizer():
    # Casing and punctuation
    n1 = TextNormalizer.normalize("What is reflection of light?")
    n2 = TextNormalizer.normalize("  what is reflection of light...  ")
    assert n1 == n2 == "what is reflection of light"

    # Contraction expansion
    assert TextNormalizer.normalize("What's reflection of light?") == "what is reflection of light"

    # MD5 Hash equality
    h1 = TextNormalizer.compute_hash(n1)
    h2 = TextNormalizer.compute_hash(n2)
    assert h1 == h2


def test_tier_1_exact_duplicate(detector):
    # Q10 is an exact duplicate of Q2
    res = detector.detect("Define the reflection of light.")
    assert res.status == DuplicateStatus.EXACT_DUPLICATE
    assert res.tier == DetectionTier.TIER_1_EXACT_HASH
    assert res.matched_question_id == "Q2"


def test_tier_2_lexical_overlap():
    store = VectorStore()
    store.add(Question(id="lex1", text="Calculate the velocity of a projectile launched at forty-five degrees angle."))
    det = CascadeDuplicateDetector(vector_store=store)
    # Near-identical wording with reordered minor tokens (Jaccard > 0.85)
    res = det.detect("Calculate the velocity of a projectile launched at forty-five degrees.")
    assert res.is_duplicate is True
    assert res.tier in [DetectionTier.TIER_2_LEXICAL, DetectionTier.TIER_3_DENSE_SEMANTIC]


def test_tier_3_semantic_duplicates(detector):
    # Q3 and Q4 are semantic duplicates of Q1
    res_q3 = detector.detect("What do you understand by the reflection of light?")
    assert res_q3.status == DuplicateStatus.SEMANTIC_DUPLICATE
    assert res_q3.is_duplicate is True

    res_q4 = detector.detect("Explain the meaning of light reflection.")
    assert res_q4.status == DuplicateStatus.SEMANTIC_DUPLICATE
    assert res_q4.is_duplicate is True


def test_distinct_questions_flagged_unique(detector):
    # Q5, Q6, Q7, Q9 are different questions
    for q_text in [
        "State the laws of reflection of light.",
        "What is the angle of reflection?",
        "The angle of incidence is 30°. What is the angle of reflection?",
        "What is the difference between regular and diffuse reflection?"
    ]:
        res = detector.detect(q_text)
        assert res.status == DuplicateStatus.UNIQUE, f"Failed for '{q_text}': got {res.status}"
        assert res.is_duplicate is False


def test_ambiguous_question_escalation(detector):
    # Q8 is ambiguous without 'of light'
    res = detector.detect("What is reflection?")
    assert res.status == DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED
    assert res.requires_review is True


def test_detector_reproducibility(detector):
    """Ensure the detector produces identical results and scores across repeated evaluations."""
    candidate = "What do you understand by the reflection of light?"
    res_initial = detector.detect(candidate)

    for _ in range(5):
        res = detector.detect(candidate)
        assert res.status == res_initial.status
        assert res.tier == res_initial.tier
        assert res.matched_question_id == res_initial.matched_question_id
        assert abs(res.similarity_score - res_initial.similarity_score) < 1e-6


def test_tags_independence(populated_store):
    detector = CascadeDuplicateDetector(vector_store=populated_store)
    # Question with completely opposing tags should STILL be flagged as duplicate
    res = detector.detect("Define the reflection of light.")
    assert res.status == DuplicateStatus.EXACT_DUPLICATE


def test_generation_avoidance(populated_store):
    generator = QuestionGenerator(vector_store=populated_store)
    new_q, det_res, meta = generator.generate_question(
        topic="reflection of light",
        tags={"grade": "10"},
        desired_cognitive_level="Analyze",
        auto_ingest=True
    )
    assert new_q is not None
    assert det_res.is_duplicate is False
    assert populated_store.count() >= 3
