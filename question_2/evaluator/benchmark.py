import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate

try:
    from ..models.question import Question, DetectionResult
    from ..models.schema_types import DuplicateStatus, DetectionTier
    from ..storage.vector_store import VectorStore
    from ..detector.cascade import CascadeDuplicateDetector
    from ..llm.client import LLMClient
except (ImportError, ValueError):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from question_2.models.question import Question, DetectionResult
    from question_2.models.schema_types import DuplicateStatus, DetectionTier
    from question_2.storage.vector_store import VectorStore
    from question_2.detector.cascade import CascadeDuplicateDetector
    from question_2.llm.client import LLMClient


# Official Test Corpus from Question 2 Specification (Section 6)
BENCHMARK_CORPUS = [
    {
        "id": "Q1",
        "text": "What is reflection of light?",
        "expected": "Original",
        "expected_status": DuplicateStatus.UNIQUE
    },
    {
        "id": "Q2",
        "text": "Define the reflection of light.",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE
    },
    {
        "id": "Q3",
        "text": "What do you understand by the reflection of light?",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE
    },
    {
        "id": "Q4",
        "text": "Explain the meaning of light reflection.",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE
    },
    {
        "id": "Q5",
        "text": "State the laws of reflection of light.",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE
    },
    {
        "id": "Q6",
        "text": "What is the angle of reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE
    },
    {
        "id": "Q7",
        "text": "The angle of incidence is 30°. What is the angle of reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE
    },
    {
        "id": "Q8",
        "text": "What is reflection?",
        "expected": "Potentially ambiguous; review context",
        "expected_status": DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED
    },
    {
        "id": "Q9",
        "text": "What is the difference between regular and diffuse reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE
    },
    {
        "id": "Q10",
        "text": "Define the reflection of light.",
        "expected": "Exact duplicate of Q2",
        "expected_status": DuplicateStatus.EXACT_DUPLICATE
    }
]


def run_evaluation_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes the independent duplicate detection evaluation benchmark across
    all test questions defined in Section 6.
    """
    store = VectorStore()
    llm = LLMClient()
    detector = CascadeDuplicateDetector(vector_store=store, llm_client=llm)

    # 1. Ingest Baseline Questions Q1 and Q2
    # Q1 is the base definition, Q2 is ingested so Q10 can test exact duplicate of Q2
    q1 = Question(id="Q1", text=BENCHMARK_CORPUS[0]["text"], tags={"topic": "reflection_of_light", "source": "baseline"})
    store.add(q1)

    # Ingest Q2 so Q10 can match against Q2 as exact duplicate
    q2 = Question(id="Q2", text=BENCHMARK_CORPUS[1]["text"], tags={"topic": "reflection_of_light", "source": "baseline"})
    store.add(q2)

    rows = []
    correct_count = 0
    total_eval = 0
    tier_counts = {tier.value: 0 for tier in DetectionTier}
    latencies = []

    # Evaluate Q2 through Q10
    # Note: Q2 is tested against Q1 (simulating candidate Q2 arriving when Q1 is in store)
    for item in BENCHMARK_CORPUS[1:]:
        total_eval += 1
        cand_id = item["id"]
        cand_text = item["text"]
        expected_status = item["expected_status"]

        start_t = time.perf_counter()
        # For Q2 evaluation against Q1, ignore match to itself by passing candidate_id="Q2"
        res: DetectionResult = detector.detect(candidate_text=cand_text, candidate_id=cand_id)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        tier_counts[res.tier.value] = tier_counts.get(res.tier.value, 0) + 1

        is_match = (res.status == expected_status)
        if is_match:
            correct_count += 1

        rows.append([
            cand_id,
            cand_text[:40] + ("..." if len(cand_text) > 40 else ""),
            item["expected"][:28],
            res.status.value,
            res.tier.value,
            f"{res.similarity_score:.3f}",
            f"{elapsed_ms:.2f}ms",
            "PASS" if is_match else "FAIL"
        ])

    accuracy = (correct_count / total_eval) * 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    if verbose:
        headers = ["ID", "Question", "Expected", "Detected Status", "Triggered Tier", "Sim Score", "Latency", "Result"]
        print("\n" + "=" * 95)
        print(" INDEPENDENT DUPLICATE DETECTION BENCHMARK (Section 6 Test Corpus)")
        print("=" * 95)
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print("\n--- BENCHMARK SUMMARY ---")
        print(f"Total Evaluated Questions: {total_eval}")
        print(f"Correct Classifications  : {correct_count} / {total_eval} ({accuracy:.1f}%)")
        print(f"Average Detection Latency: {avg_latency:.2f} ms")
        print("Tier Activation Distribution:")
        for t, cnt in tier_counts.items():
            if cnt > 0:
                print(f"  - {t}: {cnt} items")
        print("=" * 95 + "\n")

    return {
        "total": total_eval,
        "correct": correct_count,
        "accuracy": accuracy,
        "average_latency_ms": avg_latency,
        "tier_distribution": tier_counts,
        "details": rows
    }


if __name__ == "__main__":
    run_evaluation_benchmark()
