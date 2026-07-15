"""Load the clinical corpus and the evaluation set.

The corpus is a vendored MedQuAD-style JSONL slice so the repo runs fully offline
and reproducibly. Setting ``data.use_huggingface: true`` in config pulls the full
MedQuAD dataset instead (network + `datasets` package required).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import Config


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    topic: str = ""
    source: str = ""


@dataclass
class EvalItem:
    query_id: str
    query: str
    relevant_doc_ids: List[str]
    reference_answer: str = ""
    metadata: Dict = field(default_factory=dict)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(cfg: Config) -> List[Document]:
    if cfg.data.use_huggingface:
        return _load_medquad_hf()
    rows = _read_jsonl(cfg.path(cfg.data.corpus_path))
    return [
        Document(
            doc_id=r["doc_id"],
            title=r.get("title", ""),
            text=r["text"],
            topic=r.get("topic", ""),
            source=r.get("source", ""),
        )
        for r in rows
    ]


def load_eval_set(cfg: Config) -> List[EvalItem]:
    rows = _read_jsonl(cfg.path(cfg.data.eval_path))
    return [
        EvalItem(
            query_id=r["query_id"],
            query=r["query"],
            relevant_doc_ids=list(r["relevant_doc_ids"]),
            reference_answer=r.get("reference_answer", ""),
            metadata=r.get("metadata", {}),
        )
        for r in rows
    ]


def _load_medquad_hf(limit: int = 400) -> List[Document]:
    """Optional path: pull MedQuAD from HuggingFace. Kept minimal on purpose."""
    from datasets import load_dataset  # local import; only needed on this path

    ds = load_dataset("lavita/MedQuAD", split="train")
    docs: List[Document] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        answer = (row.get("answer") or "").strip()
        if not answer:
            continue
        docs.append(
            Document(
                doc_id=f"medquad-{i:05d}",
                title=(row.get("question") or "").strip(),
                text=answer,
                topic=row.get("focus_area", "") or "",
                source="MedQuAD (HuggingFace: lavita/MedQuAD)",
            )
        )
    return docs
