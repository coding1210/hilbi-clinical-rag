"""Sentence-window chunking.

Design rationale
----------------
Clinical answers are information-dense; a single passage often mixes diagnosis,
thresholds, and treatment. We chunk so that:
  * chunks never split mid-sentence (retrieval returns coherent, quotable spans);
  * chunk size is bounded (large chunks dilute the embedding signal and blow the
    minimum-context budget) with a small overlap (facts spanning a boundary are
    not lost between adjacent chunks).

Token counting uses a lightweight word+punctuation approximation rather than a
model tokenizer, keeping chunking decoupled from any specific embedding model.
The embedding model still truncates internally, so the approximation only needs
to be in the right ballpark. Size/overlap are config-driven for ablations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .config import ChunkingCfg
from .data_loader import Document

# Split on sentence-ending punctuation followed by whitespace. Simple and
# dependency-free; good enough for well-formed clinical prose.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int  # index of this chunk within its document


def approx_tokens(text: str) -> int:
    """Approximate token count (words + punctuation marks)."""
    return len(_TOKEN_RE.findall(text))


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def chunk_document(doc: Document, cfg: ChunkingCfg) -> List[Chunk]:
    """Greedily pack sentences into chunks up to ``max_tokens`` with overlap."""
    sentences = _split_sentences(doc.text) if cfg.respect_sentences else [doc.text]
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = approx_tokens(sent)
        # If a single sentence already exceeds the budget, let it stand alone.
        if current and current_tokens + sent_tokens > cfg.max_tokens:
            chunks.append(" ".join(current))
            # Start next chunk with a tail overlap from the previous one.
            current = _overlap_tail(current, cfg.overlap_tokens)
            current_tokens = approx_tokens(" ".join(current))
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}::c{i}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=chunk_text,
            position=i,
        )
        for i, chunk_text in enumerate(chunks)
    ]


def _overlap_tail(sentences: List[str], overlap_tokens: int) -> List[str]:
    """Return the trailing sentences whose combined size ~= overlap_tokens."""
    if overlap_tokens <= 0:
        return []
    tail: List[str] = []
    total = 0
    for sent in reversed(sentences):
        tail.insert(0, sent)
        total += approx_tokens(sent)
        if total >= overlap_tokens:
            break
    return tail


def chunk_corpus(docs: List[Document], cfg: ChunkingCfg) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, cfg))
    return chunks
