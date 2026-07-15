"""Hybrid index: dense (FAISS) + lexical (BM25), fused with Reciprocal Rank Fusion.

Why hybrid?
-----------
Dense embeddings capture paraphrase and semantic similarity but can miss exact
clinical tokens (drug names, abbreviations like COPD/DKA, lab thresholds).
BM25 nails those exact matches but misses paraphrase. Fusing the two rank lists
with RRF gives robust recall without having to normalise heterogeneous score
scales — RRF only uses ranks, which is why it is the pragmatic default.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .chunking import Chunk
from .config import Config
from .embeddings import Embedder

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class HybridIndex:
    """Holds chunks + a dense FAISS index + a BM25 model over the same chunks."""

    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._faiss = None
        self._bm25 = None
        self._embeddings: np.ndarray | None = None

    # ---- build -------------------------------------------------------------
    def build(self, embedder: Embedder) -> "HybridIndex":
        import faiss
        from rank_bm25 import BM25Okapi

        texts = [c.text for c in self.chunks]
        self._embeddings = embedder.encode_documents(texts)
        dim = self._embeddings.shape[1]
        # Inner product on normalised vectors == cosine similarity.
        self._faiss = faiss.IndexFlatIP(dim)
        self._faiss.add(self._embeddings)
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts])
        return self

    # ---- persistence -------------------------------------------------------
    def save(self, out_dir: Path) -> None:
        import faiss

        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss, str(out_dir / "dense.faiss"))
        np.save(out_dir / "embeddings.npy", self._embeddings)
        with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c.__dict__) + "\n")

    @classmethod
    def load(cls, out_dir: Path) -> "HybridIndex":
        import faiss
        from rank_bm25 import BM25Okapi

        chunks: List[Chunk] = []
        with open(out_dir / "chunks.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                chunks.append(Chunk(**json.loads(line)))
        idx = cls(chunks)
        idx._faiss = faiss.read_index(str(out_dir / "dense.faiss"))
        idx._embeddings = np.load(out_dir / "embeddings.npy")
        idx._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        return idx

    # ---- search primitives -------------------------------------------------
    def dense_search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        scores, ids = self._faiss.search(query_vec.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    # ---- fusion ------------------------------------------------------------
    def search(
        self, query: str, embedder: Embedder, cfg: Config
    ) -> List[Tuple[int, float]]:
        """Return candidate (chunk_index, fused_score) pairs per retrieval mode."""
        k = cfg.retrieval.candidate_k
        mode = cfg.retrieval.mode

        if mode == "dense":
            qv = embedder.encode_queries([query])[0]
            return self.dense_search(qv, k)
        if mode == "bm25":
            return self.bm25_search(query, k)
        if mode == "hybrid":
            qv = embedder.encode_queries([query])[0]
            dense = self.dense_search(qv, k)
            lexical = self.bm25_search(query, k)
            return _reciprocal_rank_fusion([dense, lexical], cfg.retrieval.rrf_k, k)
        raise ValueError(f"Unknown retrieval mode: {mode}")


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]], rrf_k: int, k: int
) -> List[Tuple[int, float]]:
    """Standard RRF: score(d) = sum over lists of 1 / (rrf_k + rank(d))."""
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(idx, score) for idx, score in ordered[:k]]
