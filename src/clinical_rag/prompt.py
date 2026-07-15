"""Prompt construction with *minimum context*.

Two things happen here, both deliberate:

1. Minimum context — even after re-ranking to top_k, we greedily add chunks only
   until a token budget is hit. Less context = lower cost, less distraction for
   the model, and a smaller privacy surface. This is a safety lever, not just a
   cost one.

2. Grounding contract — the prompt forces the model to answer *only* from the
   supplied sources, cite them by label ([S1], [S2], ...), and say when the
   answer is not present. Citation labels map back to document ids so the eval
   can measure citation coverage.

The query passed in here is already pseudonymised: no raw PHI reaches the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .chunking import approx_tokens
from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a careful clinical assistant. Answer the question using ONLY the "
    "numbered sources provided. Cite the sources you use with their labels in "
    "square brackets, e.g. [S1]. If the sources do not contain the answer, say "
    "you do not have enough information. Do not invent facts, doses, or values. "
    "Patient identifiers may appear as placeholder tokens such as [PERSON_1]; "
    "keep them as-is."
)


@dataclass
class ContextItem:
    label: str  # S1, S2, ...
    doc_id: str
    text: str


@dataclass
class PromptBundle:
    system: str
    user: str
    context: List[ContextItem]

    @property
    def source_labels(self) -> List[str]:
        return [c.label for c in self.context]


def select_min_context(
    retrieved: List[RetrievedChunk], max_context_tokens: int
) -> List[ContextItem]:
    items: List[ContextItem] = []
    budget = 0
    for i, r in enumerate(retrieved, start=1):
        cost = approx_tokens(r.chunk.text)
        if items and budget + cost > max_context_tokens:
            break
        items.append(ContextItem(label=f"S{i}", doc_id=r.chunk.doc_id, text=r.chunk.text))
        budget += cost
    return items


def build_prompt(
    pseudonymised_query: str,
    retrieved: List[RetrievedChunk],
    max_context_tokens: int,
) -> PromptBundle:
    context = select_min_context(retrieved, max_context_tokens)
    blocks = [f"[{c.label}] (source: {c.doc_id})\n{c.text}" for c in context]
    sources_text = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    user = (
        f"Sources:\n{sources_text}\n\n"
        f"Question: {pseudonymised_query}\n\n"
        f"Answer using only the sources above and cite them by label."
    )
    return PromptBundle(system=SYSTEM_PROMPT, user=user, context=context)
