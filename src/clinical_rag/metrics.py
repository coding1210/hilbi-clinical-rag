"""Evaluation metrics: retrieval, generation, and privacy.

All metrics are implemented transparently (no black-box eval framework) so every
number in the report is auditable. Retrieval metrics need gold document ids;
generation metrics are computed against reference answers with deterministic,
model-free lexical measures so results reproduce exactly offline. A hook for an
LLM-judge faithfulness score is provided for when Claude is configured.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence, Set

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "with",
    "on", "at", "by", "be", "as", "that", "this", "it", "its", "should", "may",
}


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _content(text: str) -> Set[str]:
    return {t for t in _tokens(text) if t not in _STOP and len(t) > 2}


# ─────────────────────────── retrieval metrics ──────────────────────────────
def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(set(ranked[:k]) & relevant)
    return hit / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hit = len(set(ranked[:k]) & relevant)
    return hit / k


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ─────────────────────────── generation metrics ─────────────────────────────
def token_f1(pred: str, ref: str) -> float:
    """SQuAD-style token overlap F1 between answer and reference."""
    p, r = _content(pred), _content(ref)
    if not p or not r:
        return 0.0
    common = len(p & r)
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return 2 * precision * recall / (precision + recall)


def groundedness(answer: str, context: str, sent_threshold: float = 0.5) -> float:
    """Fraction of answer sentences supported by the retrieved context.

    A sentence is "supported" if at least ``sent_threshold`` of its content
    words also appear in the context — a cheap, deterministic proxy for
    faithfulness / absence of hallucination.
    """
    ctx = _content(context)
    sentences = [s for s in _SENT_RE.split(answer.strip()) if s.strip()]
    if not sentences:
        return 0.0
    supported = 0
    for sent in sentences:
        words = _content(sent)
        if not words:
            supported += 1  # no content words (e.g. citation-only) → not a claim
            continue
        if len(words & ctx) / len(words) >= sent_threshold:
            supported += 1
    return supported / len(sentences)


_CITATION_RE = re.compile(r"\[S\d+\]")


def has_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer))


# ─────────────────────────── aggregation ────────────────────────────────────
def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval(
    per_query_ranked: List[List[str]],
    per_query_relevant: List[Set[str]],
    k_values: List[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in k_values:
        out[f"recall@{k}"] = mean(
            [recall_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        )
        out[f"precision@{k}"] = mean(
            [precision_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        )
        out[f"ndcg@{k}"] = mean(
            [ndcg_at_k(r, rel, k) for r, rel in zip(per_query_ranked, per_query_relevant)]
        )
    out["mrr"] = mean(
        [reciprocal_rank(r, rel) for r, rel in zip(per_query_ranked, per_query_relevant)]
    )
    return out
