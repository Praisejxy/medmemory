"""
Vector store — lightweight, dependency-free semantic memory search.

Stores an embedding per fact so we can do semantic similarity search,
e.g. matching "I get headaches" against a stored fact "reports migraines
triggered by screen time" even without exact keyword overlap.

Also used for contradiction detection: before inserting a new fact, we
search for existing facts of the same type that are semantically close,
which flags candidates for the contradiction-resolution step in manager.py.

DESIGN NOTE: earlier versions of this used Chroma as the vector DB, but
Chroma pulls in onnxruntime, tokenizers (which requires a Rust
toolchain to build), fastapi/uvicorn, and other heavy transitive
dependencies — none of which are needed for a personal-scale memory
store (a few hundred facts per user, not billions of documents). That
combination broke Netlify's build (no Rust/Cargo available) and would
very likely hit the same wall — plus package-size limits — on
Alibaba Cloud Function Compute.

Instead: embeddings are computed with a pure-Python offline hashing
function (see decay.py-adjacent _hash_embed below) and stored directly
in SQLite as JSON. Search does cosine similarity in plain Python. At
this scale (single-user or small-cohort memory, not a general-purpose
document corpus) a linear scan is fast, has zero compiled dependencies,
and keeps the deployment package tiny and cold-start fast on serverless.
"""

import re
import math
import json
import time
import hashlib
import sqlite3
from collections import Counter
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

EMBED_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id);
"""


def _hash_embed(text: str) -> List[float]:
    """
    Deterministic, fully offline embedding using the classic feature-
    hashing trick (bag-of-words hashed into a fixed-size vector,
    L2-normalized). No model download, no external API call, no
    cold-start latency. Swap for the Qwen Cloud text-embedding API in
    production for higher semantic quality; this is the free,
    zero-dependency default so the whole pipeline runs offline.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    vec = [0.0] * EMBED_DIM
    counts = Counter(tokens)
    for token, count in counts.items():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % EMBED_DIM
        sign = 1.0 if (h // EMBED_DIM) % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    # Both vectors are already L2-normalized at embed time, so cosine
    # similarity is just the dot product.
    return sum(x * y for x, y in zip(a, b))


class VectorMemory:
    def __init__(self, persist_path: str = "./chroma_store"):
        # persist_path kept as a directory for backward-compatible call
        # sites; the actual file lives inside it.
        import os
        os.makedirs(persist_path, exist_ok=True)
        self.db_path = os.path.join(persist_path, "embeddings.db")
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def add(self, fact_id: str, user_id: str, content: str, fact_type: str):
        embedding = _hash_embed(content)
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embeddings
                   (id, user_id, fact_type, content, embedding, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (fact_id, user_id, fact_type, content, json.dumps(embedding), time.time()),
            )

    def deactivate(self, fact_id: str):
        """Remove a superseded fact from active semantic search results."""
        with self._conn() as conn:
            conn.execute("DELETE FROM embeddings WHERE id = ?", (fact_id,))

    def search(self, user_id: str, query: str, top_k: int = 8,
               fact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query_vec = _hash_embed(query)

        with self._conn() as conn:
            if fact_type:
                rows = conn.execute(
                    "SELECT * FROM embeddings WHERE user_id = ? AND fact_type = ?",
                    (user_id, fact_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM embeddings WHERE user_id = ?", (user_id,),
                ).fetchall()

        scored = []
        for row in rows:
            vec = json.loads(row["embedding"])
            sim = _cosine(query_vec, vec)
            scored.append({
                "id": row["id"],
                "content": row["content"],
                "similarity": sim,
                "fact_type": row["fact_type"],
            })

        scored.sort(key=lambda h: h["similarity"], reverse=True)
        return scored[:top_k]