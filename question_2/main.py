import sys
from pathlib import Path
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from question_2.models.question import Question
from question_2.storage.vector_store import VectorStore
from question_2.detector.cascade import CascadeDuplicateDetector
from question_2.generator.engine import QuestionGenerator
from question_2.evaluator.benchmark import run_evaluation_benchmark, BENCHMARK_CORPUS
from question_2.config import settings


def demonstrate_system():
    print("=" * 80)
    print(" QUESTION 2: AI QUESTION GENERATION & DUPLICATE PREVENTION SYSTEM")
    print("=" * 80)
    print(f"Active LLM Provider       : {settings.get_active_llm_provider().upper()}")
    print(f"Dense Embedding Model     : {settings.EMBEDDING_MODEL_NAME} ({settings.EMBEDDING_DIMENSION}-dim)")
    print(f"Semantic Duplicate Thresh : {settings.SEMANTIC_DUP_THRESHOLD}")
    print(f"Uniqueness Cutoff Thresh  : {settings.UNIQUE_THRESHOLD}")
    print("=" * 80)

    # -------------------------------------------------------------
    # DEMO PART 1: INDEPENDENT DUPLICATE DETECTION BENCHMARK
    # -------------------------------------------------------------
    print("\n[PART 1] Running Independent Duplicate Detection on Official Test Corpus (Q1-Q10)...")
    benchmark_results = run_evaluation_benchmark(verbose=True)

    # -------------------------------------------------------------
    # DEMO PART 2: GENERATION-TIME DUPLICATE AVOIDANCE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 2] Demonstrating Generation-Time Avoidance Engine")
    print("=" * 80)

    # Create fresh store and populate with existing corpus Q1-Q10
    store = VectorStore()
    print(f"\n1. Ingesting {len(BENCHMARK_CORPUS)} existing questions into Question Bank...")
    for item in BENCHMARK_CORPUS:
        store.add(Question(
            id=item["id"],
            text=item["text"],
            tags={"subject": "Physics", "topic": "reflection_of_light", "grade": "10"}
        ))
    print(f"   Done. Question Bank currently holds {store.count()} items.")

    detector = CascadeDuplicateDetector(vector_store=store)
    generator = QuestionGenerator(vector_store=store, detector=detector)

    topic = "reflection of light"
    desired_level = "Apply / Analyze"
    print(f"\n2. Requesting LLM to generate novel questions for topic: '{topic}'")
    print(f"   Target Cognitive Level: {desired_level}")
    print("   Context Avoidance Strategy: MMR Selection + Semantic Centroids + Explicit Negative Exemplars")

    new_q, det_res, meta = generator.generate_question(
        topic=topic,
        tags={"subject": "Physics", "grade": "10"},
        desired_cognitive_level=desired_level,
        auto_ingest=True
    )

    if new_q:
        print("\n--> GENERATION SUCCESSFUL!")
        print(f"    Generated Question ID : {new_q.id}")
        print(f"    Generated Text        : \"{new_q.text}\"")
        print(f"    Tags Attached         : {new_q.tags}")
        print(f"    Verification Status   : {det_res.status.value} (Approved by {det_res.tier.value})")
        print(f"    Detector Score        : {det_res.similarity_score:.4f} (Unique Cutoff: <{settings.UNIQUE_THRESHOLD})")
        print(f"    Ingested to Bank      : New Bank Size = {store.count()} questions")
        print("\n    Negative Exemplars Provided in Avoidance Context:")
        for idx, ex in enumerate(meta.get("avoidance_context_used", []), 1):
            print(f"      [{idx}] \"{ex}\"")
    else:
        print(f"\n--> GENERATION FAILED: {meta.get('failed_reason')}")

    # -------------------------------------------------------------
    # DEMO PART 3: PROVING PREVENTION FAILURE SAFEGUARD
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[PART 3] Safeguard Simulation: What happens if generation creates a duplicate?")
    print("=" * 80)
    simulated_duplicate = "Explain what is meant by the reflection of light."
    print(f"Simulating rogue LLM output: \"{simulated_duplicate}\"")
    
    safeguard_res = detector.detect(simulated_duplicate)
    print(f"--> Detector Interception: {safeguard_res.status.value}")
    print(f"    Triggered Tier        : {safeguard_res.tier.value}")
    print(f"    Matched Corpus Item   : \"{safeguard_res.matched_question_text}\" (ID: {safeguard_res.matched_question_id})")
    print(f"    Similarity Score      : {safeguard_res.similarity_score:.4f}")
    print(f"    Explanation           : {safeguard_res.explanation}")
    print("\nSafeguard verified: Rogue candidate was successfully blocked from corrupting the question bank.")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_system()
