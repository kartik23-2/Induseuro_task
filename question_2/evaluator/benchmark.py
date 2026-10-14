import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
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


# ============================================================================
# 1. Official Test Corpus from Question 2 Specification (Section 6)
# ============================================================================
SECTION_6_CORPUS = [
    {
        "id": "Q1",
        "text": "What is reflection of light?",
        "expected": "Original",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Baseline"
    },
    {
        "id": "Q2",
        "text": "Define the reflection of light.",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Paraphrase"
    },
    {
        "id": "Q3",
        "text": "What do you understand by the reflection of light?",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Paraphrase"
    },
    {
        "id": "Q4",
        "text": "Explain the meaning of light reflection.",
        "expected": "Semantic duplicate of Q1",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Paraphrase"
    },
    {
        "id": "Q5",
        "text": "State the laws of reflection of light.",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Distinct Concept"
    },
    {
        "id": "Q6",
        "text": "What is the angle of reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Distinct Concept"
    },
    {
        "id": "Q7",
        "text": "The angle of incidence is 30°. What is the angle of reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Numerical Application"
    },
    {
        "id": "Q8",
        "text": "What is reflection?",
        "expected": "Potentially ambiguous; review context",
        "expected_status": DuplicateStatus.AMBIGUOUS_REVIEW_NEEDED,
        "is_duplicate": False,
        "category": "Ambiguous Scope"
    },
    {
        "id": "Q9",
        "text": "What is the difference between regular and diffuse reflection?",
        "expected": "Different question",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Comparative Analysis"
    },
    {
        "id": "Q10",
        "text": "Define the reflection of light.",
        "expected": "Exact duplicate of Q2",
        "expected_status": DuplicateStatus.EXACT_DUPLICATE,
        "is_duplicate": True,
        "category": "Exact Duplicate"
    }
]

BENCHMARK_CORPUS = SECTION_6_CORPUS

# ============================================================================
# 2. Extended Multi-Domain & Hard-Negative Benchmark Corpus
# ============================================================================
EXTENDED_BENCHMARK_CORPUS = [
    # Hard Negatives (High lexical overlap, contrasting scientific targets)
    {
        "id": "HN1",
        "candidate": "State Newton's second law of motion.",
        "baseline_id": "BASE_NEWTON",
        "baseline_text": "State Newton's first law of motion.",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Hard Negative (Law Number)"
    },
    {
        "id": "HN2",
        "candidate": "Explain the process of meiosis in diploid organisms.",
        "baseline_id": "BASE_CELL",
        "baseline_text": "Explain the process of mitosis in diploid organisms.",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Hard Negative (Cell Division)"
    },
    {
        "id": "HN3",
        "candidate": "How does light undergo refraction at an interface?",
        "baseline_id": "BASE_OPT",
        "baseline_text": "How does light undergo reflection at an interface?",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Hard Negative (Optics Phenomenon)"
    },
    {
        "id": "HN4",
        "candidate": "A ray strikes a mirror at an angle of 65°. What is the angle of reflection?",
        "baseline_id": "BASE_NUM",
        "baseline_text": "A ray strikes a mirror at an angle of 40°. What is the angle of reflection?",
        "expected_status": DuplicateStatus.UNIQUE,
        "is_duplicate": False,
        "category": "Hard Negative (Numerical Parameter)"
    },
    # Semantic Paraphrases across diverse domains
    {
        "id": "SEM1",
        "candidate": "State the first law of motion formulated by Sir Isaac Newton.",
        "baseline_id": "BASE_NEWTON",
        "baseline_text": "State Newton's first law of motion.",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Cross-Domain Paraphrase (Physics)"
    },
    {
        "id": "SEM2",
        "candidate": "Describe how autotrophic plants convert solar energy into sugars during photosynthesis.",
        "baseline_id": "BASE_BIO",
        "baseline_text": "Explain the biochemical process of photosynthesis in green plants.",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Cross-Domain Paraphrase (Biology)"
    },
    {
        "id": "SEM3",
        "candidate": "What is the Big-O computational time complexity of a binary search algorithm?",
        "baseline_id": "BASE_CS",
        "baseline_text": "Explain the logarithmic search time complexity of binary search.",
        "expected_status": DuplicateStatus.SEMANTIC_DUPLICATE,
        "is_duplicate": True,
        "category": "Cross-Domain Paraphrase (CS)"
    },
    # Exact formatting / whitespace / casing variations
    {
        "id": "FMT1",
        "candidate": "   STATE NEWTON'S FIRST LAW OF MOTION?!   ",
        "baseline_id": "BASE_NEWTON",
        "baseline_text": "State Newton's first law of motion.",
        "expected_status": DuplicateStatus.EXACT_DUPLICATE,
        "is_duplicate": True,
        "category": "Punctuation & Case Normalization"
    }
]


def run_evaluation_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes the independent duplicate detection evaluation benchmark across
    both the Section 6 corpus and the extended multi-domain hard-negative corpus.
    Calculates Precision, Recall, F1 Score, Specificity, and Tier-latency breakdown.
    """
    store = VectorStore()
    llm = LLMClient()
    detector = CascadeDuplicateDetector(vector_store=store, llm_client=llm)

    # 1. Ingest Section 6 Baseline
    q1 = Question(id="Q1", text=SECTION_6_CORPUS[0]["text"], tags={"topic": "reflection_of_light", "source": "baseline"})
    q2 = Question(id="Q2", text=SECTION_6_CORPUS[1]["text"], tags={"topic": "reflection_of_light", "source": "baseline"})
    store.add(q1)
    store.add(q2)

    rows = []
    correct_count = 0
    total_eval = 0
    tier_counts = {tier.value: 0 for tier in DetectionTier}
    latencies = []

    # Confusion matrix counters (binary duplicate vs non-duplicate)
    tp, fp, tn, fn = 0, 0, 0, 0

    # -------------------------------------------------------------
    # Part A: Section 6 Official Benchmark
    # -------------------------------------------------------------
    for item in SECTION_6_CORPUS[1:]:
        total_eval += 1
        cand_id = item["id"]
        cand_text = item["text"]
        expected_status = item["expected_status"]
        expected_dup = item["is_duplicate"]

        start_t = time.perf_counter()
        res: DetectionResult = detector.detect(candidate_text=cand_text, candidate_id=cand_id)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        tier_counts[res.tier.value] = tier_counts.get(res.tier.value, 0) + 1

        is_match = (res.status == expected_status)
        if is_match:
            correct_count += 1

        # Binary confusion metrics
        actual_dup = res.is_duplicate
        if expected_dup and actual_dup:
            tp += 1
        elif not expected_dup and actual_dup:
            fp += 1
        elif not expected_dup and not actual_dup:
            tn += 1
        elif expected_dup and not actual_dup:
            fn += 1

        rows.append([
            cand_id,
            item["category"],
            cand_text[:35] + ("..." if len(cand_text) > 35 else ""),
            res.status.value,
            res.tier.value,
            f"{res.similarity_score:.3f}",
            f"{elapsed_ms:.2f}ms",
            "PASS" if is_match else "FAIL"
        ])

    # -------------------------------------------------------------
    # Part B: Extended Multi-Domain & Hard-Negative Benchmark
    # -------------------------------------------------------------
    # Ingest baseline items for extended corpus
    extended_baselines = {
        "BASE_NEWTON": "State Newton's first law of motion.",
        "BASE_CELL": "Explain the process of mitosis in diploid organisms.",
        "BASE_OPT": "How does light undergo reflection at an interface?",
        "BASE_NUM": "A ray strikes a mirror at an angle of 40°. What is the angle of reflection?",
        "BASE_BIO": "Explain the biochemical process of photosynthesis in green plants.",
        "BASE_CS": "Explain the logarithmic search time complexity of binary search."
    }
    for bid, btext in extended_baselines.items():
        if not store.get_by_id(bid):
            store.add(Question(id=bid, text=btext, tags={"domain": "extended_baseline"}))

    ext_rows = []
    ext_correct = 0
    for item in EXTENDED_BENCHMARK_CORPUS:
        total_eval += 1
        cand_id = item["id"]
        cand_text = item["candidate"]
        expected_status = item["expected_status"]
        expected_dup = item["is_duplicate"]

        start_t = time.perf_counter()
        res = detector.detect(candidate_text=cand_text, candidate_id=cand_id)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(elapsed_ms)

        tier_counts[res.tier.value] = tier_counts.get(res.tier.value, 0) + 1

        is_match = (res.status == expected_status)
        if is_match:
            correct_count += 1
            ext_correct += 1

        actual_dup = res.is_duplicate
        if expected_dup and actual_dup:
            tp += 1
        elif not expected_dup and actual_dup:
            fp += 1
        elif not expected_dup and not actual_dup:
            tn += 1
        elif expected_dup and not actual_dup:
            fn += 1

        ext_rows.append([
            cand_id,
            item["category"],
            cand_text[:35] + ("..." if len(cand_text) > 35 else ""),
            res.status.value,
            res.tier.value,
            f"{res.similarity_score:.3f}",
            f"{elapsed_ms:.2f}ms",
            "PASS" if is_match else "FAIL"
        ])

    accuracy = (correct_count / total_eval) * 100.0
    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 100.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100.0 if (tn + fp) > 0 else 100.0

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p50_latency = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    if verbose:
        headers = ["ID", "Category", "Question Snippet", "Detected Status", "Triggered Tier", "Sim Score", "Latency", "Result"]
        print("\n" + "=" * 105)
        print(" [PART 1] OFFICIAL SECTION 6 TEST CORPUS EVALUATION (Q1 - Q10)")
        print("=" * 105)
        print(tabulate(rows, headers=headers, tablefmt="grid"))

        print("\n" + "=" * 105)
        print(" [PART 2] EXTENDED MULTI-DOMAIN & HARD-NEGATIVE EVALUATION")
        print("=" * 105)
        print(tabulate(ext_rows, headers=headers, tablefmt="grid"))

        print("\n" + "=" * 105)
        print(" STATISTICAL EVALUATION METRICS")
        print("=" * 105)
        metric_table = [
            ["Total Test Invocations", f"{total_eval}"],
            ["Exact Match Classifications", f"{correct_count} / {total_eval} ({accuracy:.1f}%)"],
            ["Duplicate Detection Precision", f"{precision:.1f}%  (TP: {tp}, FP: {fp})"],
            ["Duplicate Detection Recall", f"{recall:.1f}%  (TP: {tp}, FN: {fn})"],
            ["Duplicate Detection F1-Score", f"{f1:.1f}%"],
            ["Uniqueness Specificity (TN Rate)", f"{specificity:.1f}%  (TN: {tn}, FP: {fp})"],
            ["Average Latency per Decision", f"{avg_latency:.2f} ms (P50: {p50_latency:.2f}ms, P95: {p95_latency:.2f}ms)"]
        ]
        print(tabulate(metric_table, headers=["Evaluation Metric", "Score"], tablefmt="grid"))

        print("\nCascaded Tier Trigger Distribution:")
        for t, cnt in tier_counts.items():
            if cnt > 0:
                print(f"  * {t:<28}: {cnt:>2} items ({cnt/total_eval*100.0:.1f}%)")
        print("=" * 105 + "\n")

    return {
        "total": total_eval,
        "correct": correct_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "average_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "tier_distribution": tier_counts
    }


def benchmark_faiss_vs_linear_scaling():
    """
    Compares FAISS IndexFlatIP search latency against linear brute-force
    as the question bank scales from 100 to 1,000 items.
    """
    print("\n" + "=" * 80)
    print(" VECTOR SEARCH SCALING BENCHMARK: FAISS vs BRUTE-FORCE")
    print("=" * 80)

    encoder = VectorStore().encoder
    scales = [100, 500, 1000]
    comparison_rows = []

    for n in scales:
        # Create n random normalized vectors
        np.random.seed(42)
        raw_vecs = np.random.randn(n, encoder.dim).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        vecs = raw_vecs / norms

        query_vec = np.random.randn(1, encoder.dim).astype(np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        # 1. FAISS IndexFlatIP
        import faiss
        faiss_idx = faiss.IndexFlatIP(encoder.dim)
        faiss_idx.add(vecs)

        faiss_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            dists, idxs = faiss_idx.search(query_vec, 5)
            faiss_times.append((time.perf_counter() - t0) * 1000.0)
        faiss_avg = np.mean(faiss_times)

        # 2. Linear scan NumPy dot product
        np_times = []
        for _ in range(50):
            t0 = time.perf_counter()
            sims = np.dot(vecs, query_vec.T).flatten()
            top_k_idx = np.argsort(sims)[-5:][::-1]
            np_times.append((time.perf_counter() - t0) * 1000.0)
        np_avg = np.mean(np_times)

        speedup = np_avg / faiss_avg if faiss_avg > 0 else 1.0
        comparison_rows.append([
            f"{n} items",
            f"{np_avg:.3f} ms",
            f"{faiss_avg:.3f} ms",
            f"{speedup:.1f}x faster"
        ])

    print(tabulate(comparison_rows, headers=["Bank Size", "Brute-Force Linear", "FAISS IndexFlatIP", "Speedup"], tablefmt="grid"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_evaluation_benchmark(verbose=True)
    benchmark_faiss_vs_linear_scaling()
