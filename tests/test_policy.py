"""Regulation retrieval: corpus chunking, BM25, and RRF fusion (stub embedder, no model download)."""

import hashlib

import numpy as np

from vocalis.rights.policy import PolicyIndex, load_corpus


class _HashEmbedder:
    """Deterministic bag-of-words vectors: enough to exercise the dense and hybrid code paths offline."""

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return v + 1e-6

    def passage_embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        return (self._vec(t) for t in texts)

    def query_embed(self, texts: list[str]):  # type: ignore[no-untyped-def]
        return (self._vec(t) for t in texts)


def test_corpus_is_article_paragraphs_with_citations() -> None:
    ps = load_corpus()
    ids = [p.id for p in ps]
    assert len(ids) == len(set(ids)) and len(ps) > 60
    art7 = next(p for p in ps if p.id == "UK261 Art. 7(1)")
    assert (
        "£520" in art7.text
        and art7.jurisdiction == "UK"
        and art7.url.startswith("https://www.legislation.gov.uk")
    )
    assert next(p for p in ps if p.id == "EU261 Art. 7(1)").text.count("EUR") == 3


def test_bm25_finds_exact_clause_words() -> None:
    idx = PolicyIndex(embedder=_HashEmbedder())
    top = idx.search("extraordinary circumstances which could not have been avoided", k=2, mode="bm25")
    assert {h.passage.id for h in top} <= {"EU261 Art. 5(3)", "UK261 Art. 5(3)", "UK261 Art. 6(4)"}


def test_hybrid_fuses_and_filters_by_jurisdiction() -> None:
    idx = PolicyIndex(embedder=_HashEmbedder())
    hits = idx.search("meals and refreshments while waiting", k=3, jurisdiction="UK")
    assert hits and all(h.passage.jurisdiction == "UK" for h in hits)
    assert hits[0].passage.id == "UK261 Art. 9(1)"
    assert [h.rank for h in hits] == [1, 2, 3]
