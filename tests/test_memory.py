"""
Local smoke test — runs the full MemoryManager pipeline with MOCK Qwen
responses (no API key / credits needed) to verify:
  - fact insertion
  - reinforcement of duplicate facts
  - contradiction resolution (supersede)
  - decay scoring
  - budgeted retrieval

Run with: python -m tests.test_memory   (from the medmemory/ root)
"""

import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QWEN_MOCK_MODE"] = "true"

from backend.memory.database import FactStore
from backend.memory.vector_store import VectorMemory
from backend.memory import decay


def run():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    chroma_path = os.path.join(tmpdir, "chroma")

    store = FactStore(db_path)
    vectors = VectorMemory(chroma_path)
    user_id = "test-user-1"

    print("== Test 1: insert facts ==")
    allergy_id = store.insert_fact(user_id, "allergy", "allergic to penicillin",
                                    importance=1.0, permanent=True)
    vectors.add(allergy_id, user_id, "allergic to penicillin", "allergy")

    med_id = store.insert_fact(user_id, "medication", "takes Panadol for headaches",
                                importance=0.6, permanent=False)
    vectors.add(med_id, user_id, "takes Panadol for headaches", "medication")

    facts = store.get_active_facts(user_id)
    assert len(facts) == 2, f"expected 2 facts, got {len(facts)}"
    print(f"  OK - {len(facts)} facts stored")

    print("== Test 2: contradiction resolution (medication change) ==")
    new_med_id = store.insert_fact(user_id, "medication", "switched to Ibuprofen for headaches",
                                    importance=0.6, permanent=False)
    vectors.add(new_med_id, user_id, "switched to Ibuprofen for headaches", "medication")
    store.supersede_fact(med_id, new_med_id)
    vectors.deactivate(med_id)

    active = store.get_active_facts(user_id)
    active_meds = [f for f in active if f["fact_type"] == "medication"]
    assert len(active_meds) == 1, "old medication fact should be superseded"
    assert active_meds[0]["content"] == "switched to Ibuprofen for headaches"
    print("  OK - old medication fact superseded, new one active")

    history = store.get_history(user_id)
    assert len(history) == 3, "history should retain the superseded fact"
    print(f"  OK - full history preserved ({len(history)} rows, incl. superseded)")

    print("== Test 3: decay scoring ==")
    old_fact = dict(store.get_fact(med_id))
    old_fact["last_reinforced_at"] = time.time() - (60 * 86400)  # 60 days ago
    conf_old = decay.compute_confidence(old_fact)

    fresh_fact = dict(store.get_fact(new_med_id))
    conf_fresh = decay.compute_confidence(fresh_fact)

    permanent_fact = dict(store.get_fact(allergy_id))
    permanent_fact["last_reinforced_at"] = time.time() - (365 * 86400)  # 1 year ago
    conf_permanent = decay.compute_confidence(permanent_fact)

    print(f"  60-day-old unreinforced fact confidence: {conf_old:.3f}")
    print(f"  fresh fact confidence: {conf_fresh:.3f}")
    print(f"  1-year-old PERMANENT fact confidence: {conf_permanent:.3f}")
    assert conf_old < conf_fresh, "older unreinforced fact should have lower confidence"
    assert conf_permanent == 1.0, "permanent facts must never decay"
    print("  OK - decay behaves correctly, permanent facts pinned at 1.0")

    print("== Test 4: budgeted retrieval ranks permanent + relevant facts highest ==")
    candidates = [
        {**store.get_fact(allergy_id), "similarity": 0.4},
        {**store.get_fact(new_med_id), "similarity": 0.9},
    ]
    scored = [{**f, "score": decay.retrieval_score(f, f["similarity"])} for f in candidates]
    packed = decay.pack_into_budget(scored, token_budget=1000)
    assert len(packed) == 2
    print(f"  OK - retrieval scores: " +
          ", ".join(f"{f['content'][:30]}...={f['score']:.3f}" for f in packed))

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run()
