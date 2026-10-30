"""Local persistence: SQLite for docs/chunks/caches/usage, embeddings as float32 blobs."""
import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import closing

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(id TEXT PRIMARY KEY, name TEXT, n_pages INTEGER, n_chunks INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, idx INTEGER, page INTEGER, text TEXT, emb BLOB, PRIMARY KEY(doc_id, idx));
CREATE TABLE IF NOT EXISTS query_cache(key TEXT PRIMARY KEY, emb BLOB);
CREATE TABLE IF NOT EXISTS answer_cache(doc_id TEXT, key TEXT, payload TEXT, PRIMARY KEY(doc_id, key));
CREATE TABLE IF NOT EXISTS usage(ts REAL, kind TEXT, calls INTEGER, in_tokens INTEGER, out_tokens INTEGER, estimated INTEGER);
"""


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def question_key(q: str) -> str:
    return hashlib.sha256(normalize_question(q).encode()).hexdigest()


def unit_vectors(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        # One short-lived connection per operation keeps Flask's threads safe.
        return closing(sqlite3.connect(self.path, timeout=10))

    # --- documents -------------------------------------------------------
    def has_doc(self, doc_id: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM docs WHERE id=?", (doc_id,)).fetchone() is not None

    def get_doc(self, doc_id: str):
        with self._conn() as c:
            r = c.execute("SELECT id,name,n_pages,n_chunks FROM docs WHERE id=?", (doc_id,)).fetchone()
        return dict(zip(("id", "name", "pages", "chunks"), r)) if r else None

    def add_doc(self, doc_id, name, n_pages, chunks, embeddings):
        emb = unit_vectors(embeddings)
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?)", (doc_id, name, n_pages, len(chunks), time.time()))
            c.executemany("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?)",
                          [(doc_id, ch.idx, ch.page, ch.text, emb[i].tobytes()) for i, ch in enumerate(chunks)])
            c.commit()

    def load_chunks(self, doc_id: str):
        """-> (rows [(idx, page, text)], matrix of unit vectors)"""
        with self._conn() as c:
            rows = c.execute("SELECT idx,page,text,emb FROM chunks WHERE doc_id=? ORDER BY idx", (doc_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        return [r[:3] for r in rows], np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows])

    # --- caches ----------------------------------------------------------
    def get_query_emb(self, key: str):
        with self._conn() as c:
            r = c.execute("SELECT emb FROM query_cache WHERE key=?", (key,)).fetchone()
        return np.frombuffer(r[0], dtype=np.float32) if r else None

    def put_query_emb(self, key: str, emb):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO query_cache VALUES(?,?)", (key, unit_vectors(emb).tobytes()))
            c.commit()

    def get_answer(self, doc_id: str, key: str):
        with self._conn() as c:
            r = c.execute("SELECT payload FROM answer_cache WHERE doc_id=? AND key=?", (doc_id, key)).fetchone()
        return json.loads(r[0]) if r else None

    def put_answer(self, doc_id: str, key: str, payload: dict):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO answer_cache VALUES(?,?,?)", (doc_id, key, json.dumps(payload)))
            c.commit()

    # --- usage -----------------------------------------------------------
    def log_usage(self, kind: str, calls: int, in_tokens: int, out_tokens: int, estimated: bool = False):
        with self._conn() as c:
            c.execute("INSERT INTO usage VALUES(?,?,?,?,?,?)",
                      (time.time(), kind, calls, in_tokens, out_tokens, int(estimated)))
            c.commit()

    def usage_totals(self) -> dict:
        with self._conn() as c:
            rows = c.execute("SELECT kind,SUM(calls),SUM(in_tokens),SUM(out_tokens),MAX(estimated) "
                             "FROM usage GROUP BY kind").fetchall()
        return {k: {"calls": n, "input_tokens": i, "output_tokens": o, "estimated": bool(e)} for k, n, i, o, e in rows}
