"""Match Skills Framework job roles to SSOC 2020 5-digit occupations.

Methods (all unsupervised; nothing is fitted on the labels):
  * tfidf_word  - word 1-2 gram TF-IDF, cosine similarity
  * tfidf_char  - character 3-5 gram TF-IDF (robust to plurals/abbreviations)
  * embed       - BAAI/bge-small-en-v1.5 sentence embeddings (optional dependency)
  * hybrid      - 0.5 * tfidf_word + 0.5 * embed (weights fixed a priori, not tuned)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def role_text(roles: pd.DataFrame, with_description: bool = True) -> list[str]:
    if not with_description:
        return [f"{t}" for t in roles["job_role"]]
    return [f"{t}. {t}. {d}" for t, d in zip(roles["job_role"], roles["description"])]


def occ_text(occ: pd.DataFrame) -> list[str]:
    return [
        f"{t}. {t}. {d} {e} {a}"
        for t, d, e, a in zip(occ["title"], occ["definition"], occ["examples"], occ["alpha_titles"])
    ]


def tfidf_scores(queries: list[str], docs: list[str], analyzer: str = "word") -> np.ndarray:
    if analyzer == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    vec.fit(list(queries) + list(docs))
    q = vec.transform(queries)
    d = vec.transform(docs)
    return (q @ d.T).toarray()


def embed_scores(queries: list[str], docs: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # optional

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    q = model.encode([BGE_QUERY + x for x in queries], batch_size=64, normalize_embeddings=True)
    d = model.encode(list(docs), batch_size=64, normalize_embeddings=True)
    return np.asarray(q) @ np.asarray(d).T


def topk(scores: np.ndarray, codes: list[str], k: int = 5) -> list[list[str]]:
    """Indices of the k best docs per row, best first (stable on ties)."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    codes = np.asarray(codes)
    return [list(codes[row]) for row in order]


def hybrid(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    return w * a + (1.0 - w) * b


def hit_at_k(preds: list[list[str]], gold: list[set[str]], k: int, level: int = 5) -> np.ndarray:
    """1 if any of the first k predictions agrees with any gold code at `level` digits.

    A gold code shorter than `level` (a 4-digit unit group) matches by prefix.
    """
    out = np.zeros(len(preds), dtype=int)
    for i, (p, g) in enumerate(zip(preds, gold)):
        gl = {c[:level] for c in g}
        out[i] = int(any(any(c[: min(level, len(gc))] == gc[:level] for gc in gl) for c in p[:k]))
    return out


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
