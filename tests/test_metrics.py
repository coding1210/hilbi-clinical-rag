"""Metrics are hand-checkable — these lock the numbers to known values."""
import math

from clinical_rag import metrics


def test_recall_and_precision():
    ranked = ["d1", "d2", "d3", "d4"]
    relevant = {"d2", "d4"}
    assert metrics.recall_at_k(ranked, relevant, 2) == 0.5     # only d2 in top-2
    assert metrics.recall_at_k(ranked, relevant, 4) == 1.0
    assert metrics.precision_at_k(ranked, relevant, 4) == 0.5  # 2 hits / 4


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert metrics.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_matches_manual():
    # relevant doc sits at rank 2 -> DCG = 1/log2(3); ideal at rank 1 -> 1/log2(2)=1
    ndcg = metrics.ndcg_at_k(["x", "hit", "y"], {"hit"}, 3)
    assert math.isclose(ndcg, (1 / math.log2(3)) / 1.0, rel_tol=1e-9)


def test_token_f1():
    assert metrics.token_f1("aspirin reduces risk", "aspirin reduces risk") == 1.0
    assert metrics.token_f1("", "something") == 0.0
    assert 0.0 < metrics.token_f1("aspirin lowers stroke", "aspirin reduces stroke") < 1.0


def test_groundedness_flags_unsupported():
    context = "Metformin is first-line therapy for type 2 diabetes."
    grounded = metrics.groundedness("Metformin is first-line therapy for diabetes. [S1]", context)
    hallucinated = metrics.groundedness("The patient should undergo cardiac surgery immediately.", context)
    assert grounded > hallucinated
    assert grounded >= 0.5


def test_has_citation():
    assert metrics.has_citation("Give aspirin [S1].")
    assert not metrics.has_citation("Give aspirin.")
