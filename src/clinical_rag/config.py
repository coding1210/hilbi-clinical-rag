"""Typed configuration loaded from ``config.yaml``.

A single pydantic model tree mirrors the YAML so the rest of the codebase gets
autocomplete + validation and never reaches into raw dicts. Load once via
``load_config()`` and pass the object around.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

# Repo root = two levels up from this file (src/clinical_rag/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]


class DataCfg(BaseModel):
    corpus_path: str
    eval_path: str
    use_huggingface: bool = False


class ChunkingCfg(BaseModel):
    max_tokens: int = 320
    overlap_tokens: int = 48
    respect_sentences: bool = True


class EmbeddingsCfg(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = ""
    normalize: bool = True
    batch_size: int = 32


class RetrievalCfg(BaseModel):
    mode: str = "hybrid"  # dense | bm25 | hybrid
    candidate_k: int = 20
    rrf_k: int = 60


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-base"
    top_k: int = 4
    score_threshold: float = -6.0


class PromptCfg(BaseModel):
    max_context_tokens: int = 900


class PrivacyCfg(BaseModel):
    entities: List[str] = []
    score_threshold: float = 0.4
    enable_mrn_recognizer: bool = True


class LLMCfg(BaseModel):
    provider: str = "mock"  # mock | claude | openai
    claude_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    max_tokens: int = 512
    temperature: float = 0.0


class EvalCfg(BaseModel):
    k_values: List[int] = [1, 3, 5]
    report_md: str = "results/eval_report.md"
    report_json: str = "results/eval_results.json"


class Config(BaseModel):
    data: DataCfg
    chunking: ChunkingCfg
    embeddings: EmbeddingsCfg
    retrieval: RetrievalCfg
    rerank: RerankCfg
    prompt: PromptCfg
    privacy: PrivacyCfg
    llm: LLMCfg
    eval: EvalCfg

    def path(self, relative: str) -> Path:
        """Resolve a repo-relative path from config to an absolute Path."""
        return REPO_ROOT / relative


def _load_dotenv() -> None:
    """Populate os.environ from a repo-root .env (git-ignored), if present.

    Dependency-free and non-destructive: existing environment variables win, so
    secrets are never written to disk by us and never overridden accidentally.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> Config:
    _load_dotenv()
    path = Path(config_path) if config_path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(**raw)
    # Env override lets you switch providers without editing the committed
    # default (which stays `mock` so the repo runs offline for reviewers).
    provider = os.environ.get("LLM_PROVIDER")
    if provider:
        cfg.llm.provider = provider
    return cfg
