"""Embedding model wrapper.

Defaults to a strong, fast general encoder (BAAI/bge-small-en-v1.5). Swapping
``embeddings.model`` in config to a biomedical encoder (e.g.
``ncbi/MedCPT-Query-Encoder`` or a PubMedBERT model) demonstrates domain
adaptation with no code change.

Note the asymmetry: bge-style models want a short instruction prefix on the
*query* side only, never on documents. We honour that split explicitly.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import EmbeddingsCfg


class Embedder:
    def __init__(self, cfg: EmbeddingsCfg):
        self.cfg = cfg
        self._model = None  # lazy-loaded so importing this module is cheap

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model)
        return self._model

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)

    def _encode(self, texts: List[str], is_query: bool) -> np.ndarray:
        if is_query and self.cfg.query_prefix:
            texts = [self.cfg.query_prefix + t for t in texts]
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
