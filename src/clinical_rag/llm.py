"""Pluggable answer generation.

Two providers behind one interface:

* ``mock`` (default) — a deterministic, offline, extractive generator. It selects
  the context sentences most relevant to the query and returns them with source
  citations. No network, no API key → the eval is fully reproducible and CI-safe.
  Because it only ever emits text lifted from the retrieved sources, it is a
  faithful-by-construction baseline that isolates *retrieval* quality.

* ``claude`` — Anthropic Claude via the official SDK (needs ANTHROPIC_API_KEY).
* ``openai`` — OpenAI chat completions via the official SDK (needs OPENAI_API_KEY).

For the API providers, only pseudonymised text is ever sent — the PHI mapping
stays local and the answer is re-identified after generation.
"""
from __future__ import annotations

import os
import re
from typing import List

from .config import LLMCfg
from .prompt import PromptBundle

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "with",
    "on", "at", "by", "be", "as", "what", "how", "should", "his", "her", "he",
    "she", "was", "has", "have", "this", "that", "patient", "mr", "mrs", "ms", "dr",
}


def _keywords(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}


class MockLLM:
    """Deterministic extractive generator."""

    def generate(self, bundle: PromptBundle, query: str) -> str:
        q_terms = _keywords(query)
        scored = []
        for item in bundle.context:
            for sent in _SENT_RE.split(item.text):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(q_terms & _keywords(sent))
                if overlap:
                    scored.append((overlap, item.label, sent))
        if not scored:
            return "I do not have enough information in the provided sources to answer."
        # Stable ordering: by overlap desc, then original appearance.
        scored.sort(key=lambda x: (-x[0]))
        picked = scored[:3]
        # Preserve source order for readability.
        picked.sort(key=lambda x: x[1])
        parts = [f"{sent} [{label}]" for _, label, sent in picked]
        return " ".join(parts)


class ClaudeLLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        from anthropic import Anthropic

        self._client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def generate(self, bundle: PromptBundle, query: str) -> str:
        msg = self._client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=bundle.system,
            messages=[{"role": "user", "content": bundle.user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")


class OpenAILLM:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, bundle: PromptBundle, query: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.cfg.openai_model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": bundle.system},
                {"role": "user", "content": bundle.user},
            ],
        )
        return resp.choices[0].message.content or ""


def get_llm(cfg: LLMCfg):
    if cfg.provider == "claude":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "llm.provider=claude but ANTHROPIC_API_KEY is not set. "
                "Export it, or set llm.provider=mock in config.yaml."
            )
        return ClaudeLLM(cfg)
    if cfg.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "llm.provider=openai but OPENAI_API_KEY is not set. "
                "Put it in .env, or set llm.provider=mock in config.yaml."
            )
        return OpenAILLM(cfg)
    return MockLLM()
