from __future__ import annotations
from typing import List, Tuple
import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)

def rerank(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_items: List[dict], top_k: int) -> List[dict]:
    """
    Lightweight reranker using cosine similarity on embedding vectors.
    (This mimics a cross-encoder reranker slot in the architecture without extra deps.)
    """
    scores = [(i, cosine(query_vec, cand_vecs[i])) for i in range(cand_vecs.shape[0])]
    scores.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, s in scores[:top_k]:
        item = dict(cand_items[i])
        item["rerank_score"] = s
        out.append(item)
    return out
