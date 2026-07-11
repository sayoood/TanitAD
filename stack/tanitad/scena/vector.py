"""Self-contained local semantic search over parsed scenarios.

No network and no external API at **query time**. Two embedder paths, chosen
automatically:

* **MiniLM** — ``sentence-transformers`` ``all-MiniLM-L6-v2`` *iff* it imports
  in the venv (a ~80 MB one-time model download the first time it is built).
  Best semantic quality when present.
* **Hashing TF-IDF fallback** — a pure-numpy hashing vectorizer (word 1/2-grams
  + char 3/4/5-grams, md5-bucketed so it is **deterministic across processes**,
  sublinear TF × corpus IDF, L2-normalised). Always available, needs nothing
  but numpy, so the feature and its tests stay green fully offline.

Ranking is cosine similarity over an in-memory ``float32`` matrix (rows are
unit-normalised, so a dot product *is* the cosine). The index persists to a
single ``vectors.npz`` (ids + matrix + embedder metadata + the fitted IDF for
the fallback) next to ``scenarios.json`` and reloads without re-embedding.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9_]+")


def doc_text(scenario: dict[str, Any]) -> str:
    """The text embedded for a scenario: title + description + correct-behavior
    + tags (per the TanitScena spec)."""
    parts = [scenario.get("title", ""),
             scenario.get("description", ""),
             scenario.get("correct_behavior", "")]
    parts += [str(t) for t in scenario.get("tags", [])]
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# deterministic hashing TF-IDF (pure numpy, no sklearn)
# --------------------------------------------------------------------------- #

class HashingTfidf:
    """A tiny, deterministic hashing TF-IDF vectoriser.

    Determinism matters: Python's builtin ``hash`` is per-process salted, so we
    bucket features with md5 — identical vectors on every run and every machine.
    """

    def __init__(self, dim: int = 2048,
                 word_ngrams: tuple[int, ...] = (1, 2),
                 char_ngrams: tuple[int, ...] = (3, 4, 5)):
        self.dim = int(dim)
        self.word_ngrams = tuple(word_ngrams)
        self.char_ngrams = tuple(char_ngrams)
        self.idf: np.ndarray | None = None      # (dim,), set by fit()

    def _features(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        feats: list[str] = []
        for n in self.word_ngrams:
            for i in range(len(words) - n + 1):
                feats.append(f"w{n}:" + "_".join(words[i:i + n]))
        for w in words:
            s = f"#{w}#"
            for n in self.char_ngrams:
                for i in range(len(s) - n + 1):
                    feats.append(f"c{n}:" + s[i:i + n])
        return feats

    def _bucket(self, feat: str) -> int:
        return int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16) % self.dim

    def _counts(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for f in self._features(text):
            v[self._bucket(f)] += 1.0
        return v

    def _weight(self, counts: np.ndarray) -> np.ndarray:
        tf = np.zeros_like(counts)               # sublinear TF, no log(0) warn
        pos = counts > 0.0
        tf[pos] = 1.0 + np.log(counts[pos])
        v = tf * (self.idf if self.idf is not None else 1.0)
        nrm = float(np.linalg.norm(v))
        return v / nrm if nrm > 0.0 else v

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        counts = [self._counts(t) for t in texts]
        n = len(counts)
        df = np.zeros(self.dim, dtype=np.float64)
        for c in counts:
            df += (c > 0.0)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        if not counts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._weight(c) for c in counts])

    def transform(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._weight(self._counts(t)) for t in texts])


# --------------------------------------------------------------------------- #
# MiniLM (optional)
# --------------------------------------------------------------------------- #

def _minilm_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _load_minilm():
    """Return a loaded ``all-MiniLM-L6-v2`` or ``None`` (import/download fail
    is soft — we drop to the hashing fallback)."""
    if not _minilm_available():
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# the index
# --------------------------------------------------------------------------- #

class VectorIndex:
    """Cosine-ranked semantic index over scenario ids."""

    def __init__(self, ids: list[str], matrix: np.ndarray, meta: dict[str, Any],
                 hasher: HashingTfidf | None = None):
        self.ids = list(ids)
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.meta = dict(meta)
        self.hasher = hasher
        self._qmodel = None                      # lazy MiniLM query encoder

    # -- properties -------------------------------------------------------- #
    @property
    def embedder(self) -> str:
        return self.meta.get("embedder", "unknown")

    def __len__(self) -> int:
        return len(self.ids)

    # -- build ------------------------------------------------------------- #
    @classmethod
    def build(cls, scenarios: list[dict[str, Any]], prefer: str = "auto",
              dim: int = 2048) -> "VectorIndex":
        """Embed ``scenarios``. ``prefer`` ∈ {auto, minilm, hashing}.

        ``auto`` uses MiniLM if importable else the hashing fallback; ``minilm``
        forces MiniLM (raises if unavailable); ``hashing`` forces the
        deterministic fallback.
        """
        ids = [s["id"] for s in scenarios]
        docs = [doc_text(s) for s in scenarios]

        model = None
        if prefer in ("auto", "minilm"):
            model = _load_minilm()
            if model is None and prefer == "minilm":
                raise RuntimeError(
                    "prefer='minilm' but sentence-transformers is unavailable")

        if model is not None:
            emb = np.asarray(model.encode(docs, normalize_embeddings=True),
                             dtype=np.float32) if docs \
                else np.zeros((0, 384), dtype=np.float32)
            meta = {"embedder": "minilm", "model": "all-MiniLM-L6-v2",
                    "dim": int(emb.shape[1]) if emb.size else 384}
            idx = cls(ids, emb, meta, hasher=None)
            idx._qmodel = model
            return idx

        hasher = HashingTfidf(dim=dim)
        mat = hasher.fit_transform(docs).astype(np.float32)
        meta = {"embedder": "hashing-tfidf", "dim": hasher.dim,
                "word_ngrams": list(hasher.word_ngrams),
                "char_ngrams": list(hasher.char_ngrams)}
        return cls(ids, mat, meta, hasher=hasher)

    @classmethod
    def build_and_save(cls, scenarios: list[dict[str, Any]], path: str | Path,
                       prefer: str = "auto", dim: int = 2048) -> "VectorIndex":
        idx = cls.build(scenarios, prefer=prefer, dim=dim)
        idx.save(path)
        return idx

    # -- query ------------------------------------------------------------- #
    def _encode_query(self, query: str) -> np.ndarray | None:
        if self.embedder == "minilm":
            if self._qmodel is None:
                self._qmodel = _load_minilm()
            if self._qmodel is None:
                return None
            v = np.asarray(self._qmodel.encode([query],
                                               normalize_embeddings=True),
                           dtype=np.float32)[0]
            return v
        if self.hasher is None:
            return None
        return self.hasher.transform([query]).astype(np.float32)[0]

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-``k`` ``(scenario_id, cosine_score)`` for ``query`` (desc)."""
        if not query or not query.strip() or self.matrix.size == 0:
            return []
        q = self._encode_query(query)
        if q is None:
            return []
        sims = self.matrix @ q                    # unit rows · unit q = cosine
        k = max(1, min(int(k), len(self.ids)))
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], round(float(sims[i]), 6)) for i in order]

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        idf = self.hasher.idf if (self.hasher is not None
                                  and self.hasher.idf is not None) \
            else np.zeros(0, dtype=np.float64)
        np.savez(str(path),
                 ids=np.array(self.ids, dtype=object).astype("U16"),
                 matrix=self.matrix,
                 meta=np.array(json.dumps(self.meta)),
                 idf=idf)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        z = np.load(str(path), allow_pickle=False)
        ids = [str(x) for x in z["ids"].tolist()]
        matrix = z["matrix"].astype(np.float32)
        meta = json.loads(str(z["meta"]))
        hasher = None
        if meta.get("embedder") == "hashing-tfidf":
            hasher = HashingTfidf(
                dim=int(meta["dim"]),
                word_ngrams=tuple(meta.get("word_ngrams", (1, 2))),
                char_ngrams=tuple(meta.get("char_ngrams", (3, 4, 5))))
            hasher.idf = z["idf"].astype(np.float64)
        return cls(ids, matrix, meta, hasher=hasher)
