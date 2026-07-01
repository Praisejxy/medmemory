"""
MemoryManager — ties together the structured fact store (SQLite), the
vector store (Chroma), and the decay/priority scorer into one coherent
memory system.

Two capabilities beyond standard RAG-over-history:

  1. CONTRADICTION RESOLUTION: when a new fact conflicts with an existing
     one of the same type (e.g. "switched from Panadol to Ibuprofen"),
     the old fact is explicitly superseded rather than left to sit
     alongside the new one as equally-true. The agent is told about the
     change so it can acknowledge it ("Got it, updating your medication
     from Panadol to Ibuprofen") instead of silently forgetting.

  2. BUDGETED RETRIEVAL: retrieval never dumps the whole memory store
     into the prompt. It ranks candidates by similarity x confidence x
     importance and greedily fills a fixed token budget, per decay.py.
"""

import json
import time
from typing import List, Dict, Any, Optional

from backend.memory.database import FactStore
from backend.memory.vector_store import VectorMemory
from backend.memory import decay
from backend.api.qwen_client import QwenClient

FACT_EXTRACTION_PROMPT = """You are a clinical fact extraction module for a
personal health memory agent. Given a user's message, extract discrete,
atomic facts worth remembering long-term (medications, allergies,
diagnoses, symptoms, preferences). Ignore small talk.

Respond ONLY with JSON of the form:
{"facts": [
  {"type": "medication|allergy|diagnosis|symptom|preference|other",
   "content": "short atomic fact, third person, e.g. 'takes Metformin 500mg daily'",
   "importance": 0.0-1.0,
   "permanent": true|false}
]}

Rules:
- "permanent": true ONLY for allergies and chronic/lifelong diagnoses.
- If the message contains no rememberable facts, return {"facts": []}.
- Keep each fact atomic (one claim per fact).
"""

CONTRADICTION_SIMILARITY_THRESHOLD = 0.55  # above this, treat as "same topic"


class MemoryManager:
    def __init__(self, db_path: str = "medmemory.db",
                 chroma_path: str = "./chroma_store",
                 qwen_client: Optional[QwenClient] = None):
        self.store = FactStore(db_path)
        self.vectors = VectorMemory(chroma_path)
        self.qwen = qwen_client or QwenClient()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def extract_and_store(self, user_id: str, message: str) -> List[Dict[str, Any]]:
        """Extract facts from a raw user message, resolve contradictions
        against existing memory, and persist. Returns a list of change
        events (added / updated) so the agent can acknowledge them."""
        raw = self.qwen.chat(
            system_prompt=FACT_EXTRACTION_PROMPT,
            user_prompt=message,
            json_mode=True,
        )
        try:
            parsed = json.loads(raw)
            facts = parsed.get("facts", [])
        except (json.JSONDecodeError, AttributeError):
            facts = []

        events = []
        for f in facts:
            event = self._store_fact(
                user_id=user_id,
                fact_type=f.get("type", "other"),
                content=f.get("content", "").strip(),
                importance=float(f.get("importance", 0.5)),
                permanent=bool(f.get("permanent", False)),
            )
            if event:
                events.append(event)
        return events

    def _store_fact(self, user_id: str, fact_type: str, content: str,
                     importance: float, permanent: bool) -> Optional[Dict[str, Any]]:
        if not content:
            return None

        # Check for a near-duplicate / contradicting fact of the same type.
        candidates = self.vectors.search(user_id, content, top_k=3, fact_type=fact_type)
        best = candidates[0] if candidates else None

        if best and best["similarity"] >= CONTRADICTION_SIMILARITY_THRESHOLD:
            old_fact = self.store.get_fact(best["id"])
            if old_fact and old_fact["active"]:
                if old_fact["content"].strip().lower() == content.strip().lower():
                    # Exact same fact mentioned again -> reinforce, don't duplicate.
                    self.store.reinforce_fact(old_fact["id"])
                    return {"action": "reinforced", "fact_type": fact_type, "content": content}
                else:
                    # Same topic, different content -> treat as an update /
                    # contradiction. Supersede old, insert new.
                    new_id = self.store.insert_fact(user_id, fact_type, content, importance, permanent)
                    self.vectors.add(new_id, user_id, content, fact_type)
                    self.store.supersede_fact(old_fact["id"], new_id)
                    self.vectors.deactivate(old_fact["id"])
                    return {
                        "action": "updated",
                        "fact_type": fact_type,
                        "old_content": old_fact["content"],
                        "new_content": content,
                    }

        # No close match -> brand new fact.
        new_id = self.store.insert_fact(user_id, fact_type, content, importance, permanent)
        self.vectors.add(new_id, user_id, content, fact_type)
        return {"action": "added", "fact_type": fact_type, "content": content}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, user_id: str, query: str, token_budget: int = 300) -> List[Dict[str, Any]]:
        """Return the highest-value active facts for this query, packed
        into a fixed token budget rather than the full memory store."""
        # Pull a generous candidate pool from the vector store, then
        # re-rank with decay-aware scoring before budgeting.
        candidates = self.vectors.search(user_id, query, top_k=20)

        scored = []
        now = time.time()
        for c in candidates:
            fact = self.store.get_fact(c["id"])
            if not fact or not fact["active"]:
                continue  # superseded facts never resurface in normal retrieval
            score = decay.retrieval_score(fact, c["similarity"], now=now)
            scored.append({**fact, "similarity": c["similarity"], "score": score})

        # Always guarantee permanent, high-importance facts (e.g. allergies)
        # are considered even if the vector search didn't surface them for
        # this particular query — safety-critical facts shouldn't depend
        # on phrasing luck.
        for fact in self.store.get_active_facts(user_id):
            if fact.get("permanent") and fact["id"] not in {f["id"] for f in scored}:
                scored.append({**fact, "similarity": 0.3,
                                "score": decay.retrieval_score(fact, 0.3, now=now)})

        return decay.pack_into_budget(scored, token_budget)

    def get_full_history(self, user_id: str) -> List[Dict[str, Any]]:
        return self.store.get_history(user_id)
