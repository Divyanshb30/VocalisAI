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
Mode = Literal["bm25", "dense", "hybrid"]

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
    """Article paragraphs from each corpus file; '# key: value' header lines give source and url."""
    passages: list[Passage] = []
    for path in sorted(directory.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        meta: dict[str, str] = {}
        for line in lines:
            if m := re.match(r"#\s*(\w+):\s*(.*)", line):
                meta[m.group(1)] = m.group(2)
        code = path.stem.upper()
        jurisdiction = meta.get("jurisdiction", code[:2])
        body = "\n".join(line for line in lines if not line.startswith("#"))
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
    def __init__(self, passages: list[Passage] | None = None, embedder: Any = None) -> None:
        self.passages = passages if passages is not None else load_corpus()
        self.bm25 = BM25Okapi([_tokens(p.indexed) for p in self.passages])
        self._embedder = embedder
        self._own_model = embedder is None
        self._vectors: np.ndarray | None = None

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
        if mode == "bm25":
            scores = self.bm25.get_scores(_tokens(query))
            return sorted(enumerate(map(float, scores)), key=lambda x: -x[1])
        if mode == "dense":
            scores = self._passage_vectors() @ self._query_vector(query)
            return sorted(enumerate(map(float, scores)), key=lambda x: -x[1])
        fused: dict[int, float] = {}
        subs: tuple[Mode, Mode] = ("bm25", "dense")
        for sub in subs:
            for r, (i, _) in enumerate(self.rank(query, sub)):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def search(
        self, query: str, k: int = 5, mode: Mode = "hybrid", jurisdiction: str | None = None
    ) -> list[Hit]:
        hits: list[Hit] = []
        for i, score in self.rank(query, mode):
            p = self.passages[i]
            if jurisdiction and p.jurisdiction != jurisdiction.upper():
                continue
            hits.append(Hit(p, score, len(hits) + 1))
            if len(hits) == k:
                break
        return hits
