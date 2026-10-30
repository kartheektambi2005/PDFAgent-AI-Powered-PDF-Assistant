"""Retrieval: embed the question (cached), cosine top-k over stored chunk vectors, drop near-duplicates."""
from dataclasses import dataclass

import numpy as np

from . import config
from .store import Store, question_key

DUP_COSINE = 0.95


@dataclass
class Hit:
    idx: int
    page: int
    text: str
    score: float


def retrieve(store: Store, client, doc_id: str, question: str, k: int | None = None,
             min_score: float | None = None) -> tuple[list[Hit], float]:
    """Return (hits sorted by document order, best_score). Hits are empty when nothing clears min_score."""
    k = config.TOP_K if k is None else k
    min_score = config.MIN_SCORE if min_score is None else min_score
    rows, mat = store.load_chunks(doc_id)
    if not rows:
        return [], 0.0

    key = question_key(question)
    q = store.get_query_emb(key)
    if q is None:
        q = client.embed([question], "RETRIEVAL_QUERY")[0]
        store.put_query_emb(key, q)
        store.log_usage("embed_query", 1, len(question) // 4, 0, estimated=True)
    q = q / (np.linalg.norm(q) or 1.0)

    scores = mat @ q
    order = np.argsort(-scores)
    best = float(scores[order[0]])
    if best < min_score:
        return [], best

    picked: list[int] = []
    for i in order:
        if len(picked) == k or scores[i] < min_score:
            break
        text = rows[i][2]
        dup = any(float(mat[i] @ mat[j]) > DUP_COSINE or text in rows[j][2] or rows[j][2] in text for j in picked)
        if not dup:
            picked.append(int(i))
    picked.sort()  # document order reads more naturally in the prompt
    return [Hit(rows[i][0], rows[i][1], rows[i][2], float(scores[i])) for i in picked], best
