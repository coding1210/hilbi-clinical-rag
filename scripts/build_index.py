"""Build and persist the hybrid retrieval index from the corpus.

    python -m scripts.build_index
"""
from __future__ import annotations

import scripts  # noqa: F401  (path bootstrap)

from clinical_rag.config import load_config
from clinical_rag.data_loader import load_corpus
from clinical_rag.pipeline import build_index, index_dir


def main() -> None:
    cfg = load_config()
    docs = load_corpus(cfg)
    print(f"Loaded {len(docs)} documents from {cfg.data.corpus_path}")
    print(f"Embedding model: {cfg.embeddings.model}")
    print("Chunking + embedding + indexing (first run downloads the model)...")
    index = build_index(cfg)
    print(f"Indexed {len(index.chunks)} chunks.")
    print(f"Index written to {index_dir(cfg)}")


if __name__ == "__main__":
    main()
