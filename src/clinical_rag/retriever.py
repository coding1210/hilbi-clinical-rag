"""End-to-end retrieval: candidate generation -> cross-encoder re-rank -> top-k.

This is the single entry point the pipeline and the eval harness call. ``mode``
and ``use_rerank`` can be overridden per call so the eval script can run the
dense / hybrid / hybrid+rerank ablation against one prebuilt index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .chunking import Chunk
from .config import Config
from .embeddings import Embedder
from .index import HybridIndex
from .rerank import Reranker


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    reranked: bool


class Retriever:
    def __init__(
        self,
        index: HybridIndex,
        embedder: Embedder,
        reranker: Optional[Reranker],
        cfg: Config,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg

    def retrieve(
        self,
        query: str,
        *,
        mode: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        cfg = self.cfg
        if mode is not None:
            cfg = cfg.model_copy(deep=True)
            cfg.retrieval.mode = mode

        candidates = self.index.search(query, self.embedder, cfg)
        cand_chunks = [self.index.chunks[i] for i, _ in candidates]
        cand_scores = {i: s for i, s in candidates}

        rerank_on = self.cfg.rerank.enabled if use_rerank is None else use_rerank
        k = top_k if top_k is not None else self.cfg.rerank.top_k

        if rerank_on and self.reranker is not None and cand_chunks:
            scored = self.reranker.rerank(query, cand_chunks)
            kept = [
                RetrievedChunk(chunk=s.chunk, score=s.score, reranked=True)
                for s in scored
                if s.score >= self.cfg.rerank.score_threshold
            ]
            # Never return an empty context solely due to the threshold.
            if not kept and scored:
                kept = [
                    RetrievedChunk(chunk=scored[0].chunk, score=scored[0].score, reranked=True)
                ]
            return kept[:k]

        # No re-ranking: keep first-stage fused order.
        ordered = [
            RetrievedChunk(
                chunk=self.index.chunks[i], score=cand_scores[i], reranked=False
            )
            for i, _ in candidates
        ]
        return ordered[:k]
