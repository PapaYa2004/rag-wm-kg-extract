"""Offline stand-in for sentence_transformers, used ONLY by the end-to-end test
when the real package is not installed. Character-trigram hashing embeddings:
similar strings get similar vectors, deterministic, no downloads."""

import hashlib

import numpy as np

DIM = 256


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    t = f"  {str(text).lower()}  "
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class SentenceTransformer:
    def __init__(self, name=None, *a, **k):
        self.name = name

    def encode(self, sentences, *a, **k):
        if isinstance(sentences, str):
            return _vec(sentences)
        return np.stack([_vec(s) for s in sentences]) if len(sentences) else np.zeros((0, DIM), np.float32)
