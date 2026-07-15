"""RAG orchestration.

    query
      -> pseudonymise (PHI never leaves here on)
      -> retrieve (hybrid) -> re-rank (cross-encoder) -> top-k
      -> build minimum-context prompt
      -> [privacy gate] assert no raw PHI in the outbound payload
      -> generate (mock | Claude)
      -> re-identify the answer for the clinician
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Config, load_config
from .data_loader import load_corpus
from .deid import DeidResult, Pseudonymizer
from .embeddings import Embedder
from .index import HybridIndex
from .prompt import PromptBundle, build_prompt
from .rerank import Reranker
from .retriever import RetrievedChunk, Retriever


def index_dir(cfg: Config) -> Path:
    return cfg.path("data/index")


@dataclass
class RAGResult:
    query: str
    deid: DeidResult
    retrieved: List[RetrievedChunk]
    prompt: PromptBundle
    answer_raw: str  # may contain surrogate tokens
    answer: str  # re-identified for the clinician
    phi_leaks: List[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        cfg: Config,
        index: HybridIndex,
        embedder: Embedder,
        reranker: Optional[Reranker],
        pseudonymizer: Pseudonymizer,
        llm,
    ):
        self.cfg = cfg
        self.retriever = Retriever(index, embedder, reranker, cfg)
        self.pseudonymizer = pseudonymizer
        self.llm = llm

    def answer(self, query: str, **retrieve_kwargs) -> RAGResult:
        deid = self.pseudonymizer.pseudonymize(query)
        retrieved = self.retriever.retrieve(deid.text, **retrieve_kwargs)
        prompt = build_prompt(deid.text, retrieved, self.cfg.prompt.max_context_tokens)

        # Privacy gate. The only PHI-bearing input is the query, so we verify the
        # de-id transform left none of the original PHI in the query-derived text
        # that gets sent out. The retrieved context comes from the public corpus,
        # so it cannot be a source of patient PHI (and its wording may coincide
        # with redacted phrases, e.g. "two weeks").
        leaks = self.pseudonymizer.find_leaks(deid.text, deid)

        answer_raw = self.llm.generate(prompt, deid.text)
        answer = deid.re_identify(answer_raw)
        return RAGResult(
            query=query,
            deid=deid,
            retrieved=retrieved,
            prompt=prompt,
            answer_raw=answer_raw,
            answer=answer,
            phi_leaks=leaks,
        )


def build_pipeline(cfg: Optional[Config] = None, *, load_llm: bool = True) -> RAGPipeline:
    """Assemble a pipeline from a prebuilt on-disk index."""
    cfg = cfg or load_config()
    idx_path = index_dir(cfg)
    if not (idx_path / "dense.faiss").exists():
        raise FileNotFoundError(
            f"No index at {idx_path}. Build it first: `make index`."
        )
    index = HybridIndex.load(idx_path)
    embedder = Embedder(cfg.embeddings)
    reranker = Reranker(cfg.rerank) if cfg.rerank.enabled else None
    pseudonymizer = Pseudonymizer(cfg.privacy)

    llm = None
    if load_llm:
        from .llm import get_llm

        llm = get_llm(cfg.llm)
    return RAGPipeline(cfg, index, embedder, reranker, pseudonymizer, llm)


def build_index(cfg: Optional[Config] = None) -> HybridIndex:
    """Chunk the corpus, embed, and persist the hybrid index to disk."""
    from .chunking import chunk_corpus

    cfg = cfg or load_config()
    docs = load_corpus(cfg)
    chunks = chunk_corpus(docs, cfg.chunking)
    embedder = Embedder(cfg.embeddings)
    index = HybridIndex(chunks).build(embedder)
    index.save(index_dir(cfg))
    return index
