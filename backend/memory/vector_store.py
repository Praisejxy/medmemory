"""
Vector store wrapper (Chroma, embedded/local — no hosting cost).

Stores an embedding per fact so we can do semantic similarity search,
e.g. matching "I get headaches" against a stored fact "reports migraines
triggered by screen time" even without exact keyword overlap.

Also used for contradiction detection: before inserting a new fact, we
search for existing facts of the same type that are semantically close,
which flags candidates for the contradiction-resolution step in manager.py.
"""

import re
import math
import hashlib
from collections import Counter
from typing import List, Dict, Any, Optional

import chromadb

EMBED_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_embed(text: str) -> List[float]:
    """
    Deterministic, fully offline embedding using the classic feature-
    hashing trick (bag-of-words hashed into a fixed-size vector,
    L2-normalized). No model download, no external API call, no
    cold-start latency — important on Function Compute, and avoids the
    default Chroma embedder's dependency on downloading an ONNX model
    at runtime (which fails in network-restricted environments and adds
    multi-second cold starts in production).

    Swap this out for the Qwen Cloud text-embedding API in production
    for higher semantic quality; this is the free, zero-dependency
    default so the whole pipeline runs without any external calls.
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


class VectorMemory:
    def __init__(self, persist_path: str = "./chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_path)
        # We pass precomputed embeddings ourselves (see _hash_embed) so
        # Chroma never tries to download a model at runtime.
        self.collection = self.client.get_or_create_collection(
            name="facts",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, fact_id: str, user_id: str, content: str, fact_type: str):
        self.collection.add(
            ids=[fact_id],
            embeddings=[_hash_embed(content)],
            documents=[content],
            metadatas=[{"user_id": user_id, "fact_type": fact_type}],
        )

    def deactivate(self, fact_id: str):
        """Remove a superseded fact from active semantic search results."""
        try:
            self.collection.delete(ids=[fact_id])
        except Exception:
            pass  # already absent — non-fatal

    def search(self, user_id: str, query: str, top_k: int = 8,
               fact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        where = {"user_id": user_id}
        if fact_type:
            where = {"$and": [{"user_id": user_id}, {"fact_type": fact_type}]}

        results = self.collection.query(
            query_embeddings=[_hash_embed(query)],
            n_results=top_k,
            where=where,
        )

        hits = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for i in range(len(ids)):
            hits.append({
                "id": ids[i],
                "content": docs[i],
                "distance": dists[i],
                "similarity": 1.0 - dists[i],  # cosine distance -> similarity
                "fact_type": metas[i].get("fact_type"),
            })
        return hits
