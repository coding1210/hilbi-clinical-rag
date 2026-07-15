# Clinical RAG Prototype + Eval

A minimal, **fully offline-runnable** retrieval-augmented-generation pipeline for
clinical question answering, with **PHI pseudonymisation**, **minimum-context**
prompting, and a **transparent evaluation harness**.

Built for the "Clinical RAG prototype + eval" assignment. The emphasis is on
*clear, defensible engineering decisions* over scale — every knob is in
[`config.yaml`](config.yaml) and every metric in the report is hand-auditable.

```
query → pseudonymise → retrieve (hybrid) → re-rank (cross-encoder) → top-k
      → minimum-context prompt → [privacy gate] → generate → re-identify answer
```

---

## Quickstart

```bash
make setup      # create venv, install deps, download spaCy model (~one-time)
make index      # chunk + embed the corpus, build the hybrid index
make eval       # run the full eval → results/eval_report.md
make ask Q="Mr. John Smith MRN 4839201 with chest pain, what workup?"
make test       # unit tests (hermetic; no model downloads needed)
```

Everything runs **offline with a deterministic mock LLM** — no API key required.
To use Claude for generation instead, set `llm.provider: claude` in `config.yaml`
and export `ANTHROPIC_API_KEY`. **PHI is pseudonymised before anything is sent.**

> Python 3.12 recommended (`make setup` uses `uv` if present, else `venv`).

---

## Dataset

- **Knowledge base:** a vendored MedQuAD-style clinical Q&A slice
  ([`data/corpus/medquad_slice.jsonl`](data/corpus/medquad_slice.jsonl), ~36
  documents across hypertension, diabetes, asthma/COPD, cardiology, ID, etc.).
  Committed so the repo is reproducible with zero network. Set
  `data.use_huggingface: true` to pull the full [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD)
  instead.
- **Eval set:** [`data/eval/eval_set.jsonl`](data/eval/eval_set.jsonl) — 25
  patient-context queries, each with **synthetic PHI** (names, MRNs, DOBs, phone
  numbers), gold `relevant_doc_ids`, and a reference answer.

**Design choice — where the PHI lives.** In real clinical RAG the *knowledge
base* (guidelines/literature) is not sensitive; the **query / patient context**
carries the PHI. So the public corpus stays clean and pseudonymisation is
exercised on the incoming queries — the realistic shape, and it makes the de-id
step genuinely load-bearing rather than decorative.

---

## Design decisions

### Chunking — `src/clinical_rag/chunking.py`
Sentence-window packing to a token budget (`max_tokens=320`, `overlap=48`), never
splitting mid-sentence. Rationale: oversized chunks dilute the embedding signal
and blow the context budget; undersized chunks lose clinical context; overlap
keeps facts that straddle a boundary retrievable. Size/overlap are config-driven.

### Retrieval — `src/clinical_rag/index.py`
Three selectable modes:
- **dense** — bi-encoder embeddings (`bge-small-en-v1.5`) over a FAISS cosine index.
- **bm25** — lexical; matters for exact clinical tokens (drug names, abbreviations
  like *DKA/COPD*, lab thresholds) that dense retrieval blurs.
- **hybrid (default)** — **Reciprocal Rank Fusion** of dense + BM25. RRF fuses on
  *rank*, so it needs no fragile score normalisation across the two scales.

### Re-ranking — `src/clinical_rag/rerank.py`
Retrieve `candidate_k=20`, then a **cross-encoder** (`bge-reranker-base`) scores
each `(query, chunk)` pair jointly and we keep the top `k=4`. This is where
precision@k / nDCG at the very top are won. Swap to `ncbi/MedCPT-Cross-Encoder`
(and `ncbi/MedCPT-Query-Encoder` for embeddings) for a biomedical-tuned stack —
config only, no code change.

### Minimum context — `src/clinical_rag/prompt.py`
Even after top-k, chunks are added greedily only until a token budget
(`max_context_tokens=900`) is reached. Less context → lower cost, less model
distraction, and a **smaller privacy surface**. The prompt enforces a grounding
contract: answer *only* from the numbered sources, cite them `[S1]`, and admit
when the answer isn't present.

### Pseudonymisation — `src/clinical_rag/deid.py`
[Microsoft Presidio](https://microsoft.github.io/presidio/) (spaCy NER +
recognizers) with a **custom MRN recognizer**. PHI is not blanked — each entity
becomes a **consistent, reversible surrogate** (`[PERSON_1]`, `[DATE_1]`, …):
- *consistent*: the same value always maps to the same token, so the model can
  still reason about "the same patient";
- *reversible*: a **local-only** map re-identifies the final answer for the
  clinician; the map never leaves the process.

Pseudonymisation runs **before** the query is embedded, prompted, or sent to any
LLM. A dependency-free regex backend kicks in if Presidio/spaCy are unavailable
so the pipeline still runs; the active backend is logged.

> **HIPAA nuance:** under Safe Harbor only ages **over 89** are PHI, so ordinary
> ages (62, 74) are intentionally left intact — correct behaviour, not a miss.

### Generation — `src/clinical_rag/llm.py`
Pluggable. `mock` (default) is a deterministic extractive generator that only
emits sentences lifted from the retrieved sources — faithful by construction, so
the eval isolates *retrieval* quality with no API key. `claude` uses the
Anthropic SDK; only pseudonymised text is sent.

---

## Evaluation — `scripts/run_eval.py`

Three axes, written to [`results/eval_report.md`](results/eval_report.md) and
`results/eval_results.json`:

1. **Retrieval (ablation):** Recall@k, Precision@k, nDCG@k, MRR for
   *dense vs bm25 vs hybrid vs hybrid+rerank* — so each design decision's
   contribution is visible, not asserted.
2. **Generation:** groundedness (faithfulness proxy), answer token-F1 vs
   reference, citation coverage.
3. **Privacy:** count of PHI entities pseudonymised and **raw-PHI-leak count into
   the outbound prompt (target: 0)**.

Metrics are implemented from scratch in `src/clinical_rag/metrics.py` (no
black-box eval library) so every number is auditable.

**Headline results** (25 queries, mock LLM, Presidio de-id — full tables in
[`results/eval_report.md`](results/eval_report.md)):

| axis | result |
|---|---|
| retrieval (hybrid) | recall@5 = 1.00, nDCG@5 = 1.00, MRR = 1.00 |
| generation | groundedness = 1.00, citation coverage = 1.00, token-F1 = 0.41 |
| privacy | 68 PHI entities pseudonymised, **0 leaks** |

*Honest reading:* this curated corpus is small and topically separable, so dense
retrieval already saturates — hybrid adds no headroom and re-ranking even trades
a hair of nDCG by reordering an already-correct top hit. That is the expected
shape; the ablation exists precisely to expose where these choices *do* matter
(larger, more ambiguous corpora). token-F1 is moderate because the mock generator
extracts source sentences verbatim while references are paraphrased — that is a
property of the deterministic baseline, not a retrieval failure.

---

## Repository layout

```
src/clinical_rag/   config, data_loader, chunking, embeddings, index, rerank,
                    retriever, deid, prompt, llm, pipeline, metrics
scripts/            build_index.py, ask.py, run_eval.py
data/               corpus slice + eval set
results/            committed eval_report.md
tests/              hermetic unit tests (metrics, de-id, chunking/RRF)
```

---

## Limitations & next steps

- **Corpus is a curated slice** for offline reproducibility; flip
  `use_huggingface` for full MedQuAD, or point the loader at real (de-identified)
  discharge notes.
- **Groundedness is a lexical proxy.** With Claude configured, an LLM-judge
  faithfulness score is the natural upgrade; RAGAS could cross-check.
- **De-id precision/recall is not perfect.** Presidio F1 on clinical text is
  ~0.4–0.85 in the literature. It also *over-redacts* here: `DATE_TIME` catches
  durations and ages ("age 62", "two weeks") that are not HIPAA identifiers,
  stripping clinically useful context. Production use needs a tuned recognizer
  suite, an explicit age>89 rule, allow-listing of clinical durations, and
  human-in-the-loop review — the custom-recognizer hook shows the extension path.
- **Single-vector dense retrieval.** ColBERT-style late interaction or a
  biomedical encoder would likely lift recall further; both are config swaps.
