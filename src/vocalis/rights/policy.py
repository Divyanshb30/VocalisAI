"""Hybrid retrieval over regulation and airline policy text, with citations.

Passages are article paragraphs ("UK261 Art. 7(1)") from `data/policy/*.txt`. Ranking fuses BM25
(exact clause words: "extraordinary circumstances", "great circle") with dense embeddings
(paraphrases: "my flight got scrapped") by reciprocal rank fusion. Entitlement amounts still come
from the rules in code; retrieval supplies the wording to cite. See docs/adr/0003.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rank_bm25 import BM25Okapi

CORPUS_DIR = Path("data/policy")
CACHE = Path(".cache/policy")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60
# Dense ranks count double in the fusion: chosen on the hand-written gold set (1, 2, 3 tried) and
# checked on the LLM-reworded set, which plain words match less often. See evals/retrieval.py.
DENSE_WEIGHT = 2.0
# A small local cross-encoder re-orders the top fused candidates (no API, ~130 MB ONNX on CPU). Chosen
# over bge-reranker-base and Gemini embeddings on the same sets; see evals/retrieval.py.
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-12-v2"
RERANK_TOP = 20
Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]

_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "for",
        "and",
        "or",
        "is",
        "are",
        "be",
        "by",
        "with",
        "at",
        "from",
        "as",
        "that",
        "this",
        "it",
        "its",
        "if",
        "any",
        "all",
        "their",
        "they",
        "shall",
        "which",
        "such",
        "been",
        "has",
        "have",
        "was",
        "were",
        "when",
        "where",
        "who",
        "whom",
        "what",
        "into",
        "than",
        "then",
    ]
)


@dataclass(frozen=True)
class Passage:
    id: str  # citation, e.g. "UK261 Art. 7(1)"
    jurisdiction: str  # UK | EU | airline code
    title: str
    text: str
    source: str
    url: str

    @property
    def indexed(self) -> str:
        return f"{self.id} {self.title}. {self.text}"


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    rank: int


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def load_corpus(directory: Path = CORPUS_DIR) -> list[Passage]:
    """Passages from each corpus file; '# key: value' header lines give source and url.

    Two layouts: "Article N / title / 1. ..." (EU/UK legislation, one passage per numbered paragraph),
    and "§ <id> | <title>" followed by the text (one passage per clause, as cut by
    scripts/build_policy_corpus.py). A passage id is the file's code and the clause: "DGCA para 3.3.2".
    """
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        meta: dict[str, str] = {}
        for line in lines:
            if m := re.match(r"#\s*(\w+):\s*(.*)", line):
                meta[m.group(1)] = m.group(2)
        code = path.stem.upper().replace("_", "-")
        jurisdiction = meta.get("jurisdiction", code[:2])
        body = "\n".join(line for line in lines if not line.startswith("#"))
        if body.lstrip().startswith("§"):
            for m in re.finditer(r"(?m)^§ (.+?) \| (.+)\n(.+)$", body):
                passages.append(
                    Passage(
                        f"{code} {m.group(1)}",
                        jurisdiction,
                        m.group(2).strip(),
                        m.group(3).strip(),
                        meta.get("source", ""),
                        meta.get("url", ""),
                    )
                )
            continue
        for art in re.split(r"(?m)^(?=Article \d+\s*$)", body):
            m = re.match(r"Article (\d+)\s*\n(.*?)\n(.*)", art, re.S)
            if not m:
                continue
            num, title, rest = m.group(1), m.group(2).strip(), m.group(3)
            paras = re.split(r"(?m)^(\d+)\.\s*", rest)
            chunks = [(p, " ".join(t.split())) for p, t in zip(paras[1::2], paras[2::2], strict=True)]
            if not chunks:
                chunks = [("", " ".join(rest.split()))]
            for para, text in chunks:
                if not text:
                    continue
                cite = f"{code} Art. {num}" + (f"({para})" if para else "")
                passages.append(
                    Passage(cite, jurisdiction, title, text, meta.get("source", ""), meta.get("url", ""))
                )
    return passages


class PolicyIndex:
    def __init__(
        self, passages: list[Passage] | None = None, embedder: Any = None, reranker: Any = None
    ) -> None:
        self.passages = passages if passages is not None else load_corpus()
        self.bm25 = BM25Okapi([_tokens(p.indexed) for p in self.passages])
        self._embedder = embedder
        self._own_model = embedder is None
        self._vectors: np.ndarray | None = None
        # an injected embedder (tests) gets no reranker unless one is injected too: no model download
        self._reranker = reranker
        self._can_rerank = reranker is not None or embedder is None

    def _rerank_model(self) -> Any:
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(RERANK_MODEL)
        return self._reranker

    def rerank(self, query: str, order: list[int]) -> list[int]:
        """Re-order the first RERANK_TOP candidates by the cross-encoder; the rest keep their order."""
        if not self._can_rerank or not order:
            return order
        top = order[:RERANK_TOP]
        scores = list(self._rerank_model().rerank(query, [self.passages[i].indexed for i in top]))
        return [top[k] for k in np.argsort(scores)[::-1]] + order[RERANK_TOP:]

    # ------------------------------------------------------------------ dense
    def _model(self) -> Any:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(EMBED_MODEL)
        return self._embedder

    def _passage_vectors(self) -> np.ndarray:
        if self._vectors is None:
            text = EMBED_MODEL + "\n" + "\n".join(p.indexed for p in self.passages)
            path = CACHE / f"{hashlib.sha1(text.encode()).hexdigest()[:16]}.npy"
            own_model = self._own_model  # an injected embedder never reads or writes the disk cache
            if own_model and path.exists():
                self._vectors = np.load(path)
            else:
                vecs = np.array(list(self._model().passage_embed([p.indexed for p in self.passages])))
                self._vectors = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                if own_model:
                    CACHE.mkdir(parents=True, exist_ok=True)
                    np.save(path, self._vectors)
        return self._vectors

    def _query_vector(self, query: str) -> np.ndarray:
        q: np.ndarray = np.array(next(iter(self._model().query_embed([query]))))
        return q / float(np.linalg.norm(q))

    # ---------------------------------------------------------------- ranking
    def rank(self, query: str, mode: Mode = "hybrid") -> list[tuple[int, float]]:
        """All passage indices, best first, with the mode's score."""
        if mode == "hybrid_rerank":
            candidates = self.rank(query, "hybrid")
            by_index = dict(candidates)
            return [(i, by_index[i]) for i in self.rerank(query, [i for i, _ in candidates])]
        if mode == "bm25":
            scores = self.bm25.get_scores(_tokens(query))
            return sorted(enumerate(map(float, scores)), key=lambda x: -x[1])
        if mode == "dense":
            scores = self._passage_vectors() @ self._query_vector(query)
            return sorted(enumerate(map(float, scores)), key=lambda x: -x[1])
        fused: dict[int, float] = {}
        subs: tuple[tuple[Mode, float], ...] = (("bm25", 1.0), ("dense", DENSE_WEIGHT))
        for sub, weight in subs:
            for r, (i, _) in enumerate(self.rank(query, sub)):
                fused[i] = fused.get(i, 0.0) + weight / (RRF_K + r + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid_rerank",
        jurisdiction: str | None = None,
        scope: set[str] | None = None,
    ) -> list[Hit]:
        """Top passages; ``scope`` keeps only those jurisdictions (the case's applicable rules, e.g.
        {"IN", "INTL"}), ``jurisdiction`` only one. The scope is applied before reranking, so the
        cross-encoder re-orders candidates that can actually be returned."""
        keep = {j.upper() for j in scope} if scope else {jurisdiction.upper()} if jurisdiction else None
        base = "hybrid" if mode == "hybrid_rerank" else mode
        ranked = [
            (i, s) for i, s in self.rank(query, base) if not keep or self.passages[i].jurisdiction in keep
        ]
        if mode == "hybrid_rerank":
            by_index = dict(ranked)
            ranked = [(i, by_index[i]) for i in self.rerank(query, [i for i, _ in ranked])]
        hits: list[Hit] = []
        for i, score in ranked:
            hits.append(Hit(self.passages[i], score, len(hits) + 1))
            if len(hits) == k:
                break
        return hits


REGIME_SCOPE = {"DGCA": "IN", "GCAA": "AE", "UK261": "UK", "EU261": "EU", "MONTREAL": "INTL"}


def scope_for(regimes: list[str]) -> set[str]:
    """Corpus jurisdictions for the regimes that apply to a case (international rules always included)."""
    return {REGIME_SCOPE[r] for r in regimes if r in REGIME_SCOPE} | {"INTL"}


_SHARED: PolicyIndex | None = None


def shared_index() -> PolicyIndex:
    """One index per process: the corpus and its vectors load once, on first use."""
    global _SHARED
    if _SHARED is None:
        _SHARED = PolicyIndex()
    return _SHARED
