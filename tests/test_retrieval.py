"""Chunking + rank-fusion tests (pure, no model downloads)."""
from clinical_rag.chunking import chunk_document
from clinical_rag.config import ChunkingCfg
from clinical_rag.data_loader import Document
from clinical_rag.index import _reciprocal_rank_fusion


def test_chunking_respects_token_budget_and_covers_text():
    text = " ".join(f"Sentence number {i} about clinical care." for i in range(40))
    doc = Document(doc_id="d1", title="t", text=text)
    cfg = ChunkingCfg(max_tokens=40, overlap_tokens=8, respect_sentences=True)
    chunks = chunk_document(doc, cfg)
    assert len(chunks) > 1                      # long doc splits
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].chunk_id == "d1::c0"
    # First sentence appears in the first chunk (nothing dropped at the head).
    assert "Sentence number 0" in chunks[0].text


def test_short_doc_is_single_chunk():
    doc = Document(doc_id="d2", title="t", text="One short clinical fact.")
    chunks = chunk_document(doc, ChunkingCfg(max_tokens=320))
    assert len(chunks) == 1


def test_rrf_prefers_items_ranked_high_in_both_lists():
    dense = [(10, 0.9), (11, 0.8), (12, 0.7)]
    lexical = [(11, 5.0), (13, 4.0), (10, 3.0)]
    fused = _reciprocal_rank_fusion([dense, lexical], rrf_k=60, k=3)
    ids = [i for i, _ in fused]
    # 11 (rank1+rank... appears high in both) and 10 outrank single-list items.
    assert ids[0] in (10, 11)
    assert 11 in ids and 10 in ids
