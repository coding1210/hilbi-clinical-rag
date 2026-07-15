"""Cross-encoder re-ranking.

The first-stage retriever (bi-encoder + BM25) is optimised for recall over the
whole corpus. A cross-encoder jointly encodes the (query, chunk) pair and scores
true relevance far more precisely, at a cost that is only acceptable on the small
candidate set. We retrieve ``candidate_k`` then re-rank down to ``top_k`` — this
is where precision@k is won for clinical QA.

Default model: BAAI/bge-reranker-base. Swap to ncbi/MedCPT-Cross-Encoder for a
biomedical-tuned reranker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .chunking import Chunk
from .config import RerankCfg


@dataclass
class Scored:
    chunk: Chunk
    score: float


class Reranker:
    def __init__(self, cfg: RerankCfg):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.model)
        return self._model

    def rerank(self, query: str, candidates: List[Chunk]) -> List[Scored]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        scored = [Scored(chunk=c, score=float(s)) for c, s in zip(candidates, scores)]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored
