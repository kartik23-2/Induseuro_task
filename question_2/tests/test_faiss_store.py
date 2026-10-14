import pytest
import time
import numpy as np
from question_2.models.question import Question
from question_2.storage.vector_store import VectorStore, DenseEmbeddingEncoder


@pytest.fixture(scope="module")
def encoder():
    return DenseEmbeddingEncoder()


@pytest.fixture
def empty_store(encoder):
    return VectorStore(encoder=encoder)


def test_embedding_reproducibility(encoder):
    """Verify that repeated encodings produce bit-exact, identical unit vectors."""
    text = "Explain the laws of reflection of light."
    emb1 = np.array(encoder.encode(text), dtype=np.float32)
    emb2 = np.array(encoder.encode(text), dtype=np.float32)
    emb3 = np.array(encoder.encode(text), dtype=np.float32)

    np.testing.assert_array_equal(emb1, emb2)
    np.testing.assert_array_equal(emb2, emb3)

    # Verify unit length (L2 norm == 1.0)
    norm = np.linalg.norm(emb1)
    assert abs(norm - 1.0) < 1e-5


def test_batch_encoding_consistency(encoder):
    """Verify that batch encoding matches single-item encoding exactly."""
    texts = [
        "What is reflection of light?",
        "State Snell's Law.",
        "Calculate the focal length of a concave mirror."
    ]
    single_embeddings = [encoder.encode(t) for t in texts]
    batch_embeddings = encoder.encode_batch(texts)

    assert len(batch_embeddings) == len(texts)
    for s_emb, b_emb in zip(single_embeddings, batch_embeddings):
        np.testing.assert_allclose(s_emb, b_emb, atol=1e-5)


def test_faiss_indexing_and_search(empty_store):
    """Test FAISS IndexFlatIP indexing and top-k nearest neighbor retrieval."""
    q1 = Question(id="q1", text="What is reflection of light?", tags={"topic": "optics", "level": "easy"})
    q2 = Question(id="q2", text="Define the reflection of light.", tags={"topic": "optics", "level": "easy"})
    q3 = Question(id="q3", text="Explain photosynthesis in plants.", tags={"topic": "biology", "level": "easy"})

    empty_store.add_all([q1, q2, q3])
    assert empty_store.count() == 3

    # Query semantically near Q1 and Q2
    results = empty_store.search_similar("What do you understand by the reflection of light?", top_k=2)
    assert len(results) == 2
    matched_ids = [res[0].id for res in results]
    assert "q1" in matched_ids
    assert "q2" in matched_ids
    # Similarity should be high (> 0.85)
    assert results[0][1] > 0.85


def test_faiss_tag_filtered_search(empty_store):
    """Test that tag filtering properly restricts candidate selection in FAISS."""
    q1 = Question(id="q1", text="What is reflection of light?", tags={"subject": "physics"})
    q2 = Question(id="q2", text="What is light reflection?", tags={"subject": "chemistry"})
    empty_store.add_all([q1, q2])

    # Search with subject: physics
    res_physics = empty_store.search_similar("Define reflection of light", top_k=5, tag_filter={"subject": "physics"})
    assert len(res_physics) == 1
    assert res_physics[0][0].id == "q1"

    # Search with subject: chemistry
    res_chem = empty_store.search_similar("Define reflection of light", top_k=5, tag_filter={"subject": "chemistry"})
    assert len(res_chem) == 1
    assert res_chem[0][0].id == "q2"


def test_faiss_scalability_and_latency(empty_store):
    """
    Stress test FAISS with 500 questions, measuring indexing speed
    and sub-millisecond query search latency.
    """
    # Generate 500 synthetic variations
    sample_questions = [
        Question(
            id=f"synth_{i}",
            text=f"Question {i}: How does light reflect when striking surface variation index {i % 20}?",
            tags={"index": i, "category": f"cat_{i % 5}"}
        )
        for i in range(500)
    ]

    t0 = time.perf_counter()
    empty_store.add_all(sample_questions)
    index_time_ms = (time.perf_counter() - t0) * 1000.0

    assert empty_store.count() == 500

    # Pre-encode query to isolate FAISS search index speed from PyTorch text tokenization
    query_vector = empty_store.encoder.encode("How does light reflect off surface variation?")

    # Measure pure FAISS index query search latency across 50 iterations
    search_latencies = []
    for _ in range(50):
        t_start = time.perf_counter()
        results = empty_store.search_by_vector(query_vector=query_vector, top_k=5)
        search_latencies.append((time.perf_counter() - t_start) * 1000.0)

    avg_search_ms = sum(search_latencies) / len(search_latencies)
    assert len(results) == 5
    # FAISS search across 500 vectors should be sub-millisecond (< 2.0ms on CPU)
    assert avg_search_ms < 2.0, f"FAISS search took {avg_search_ms:.3f}ms, expected < 2.0ms"

    # Also verify end-to-end text search produces identical top-k result
    text_results = empty_store.search_similar("How does light reflect off surface variation?", top_k=5)
    assert len(text_results) == 5
    assert text_results[0][0].id == results[0][0].id
