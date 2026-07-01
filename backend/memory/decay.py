"""
Decay & priority scoring — the core differentiating logic of MedMemory.

Design goal (per the hackathon brief): "efficient memory storage and
retrieval, timely forgetting of outdated information, and recalling
critical memories within limited context windows."

Naive RAG-over-chat-history treats every past message as equally
retrievable forever. That's wrong for a health-memory agent: a headache
from three months ago that never recurred should fade from relevance,
while a documented drug allergy must NEVER fade, no matter how old.

This module implements:
  1. Exponential confidence decay, with per-fact half-life modulated by
     importance and how many times the fact has been reinforced
     (mentioned again in later sessions).
  2. A hard floor for `permanent` facts (allergies, chronic diagnoses) —
     confidence is pinned at 1.0 regardless of elapsed time.
  3. A combined retrieval score blending semantic similarity (from the
     vector store) with decayed confidence and importance, so retrieval
     is not just "closest match" but "closest match that still matters."
"""

import math
import time
from typing import Dict, Any

BASE_HALF_LIFE_DAYS = 14.0   # how fast an average, unreinforced fact fades
MIN_CONFIDENCE = 0.05        # facts are deprioritized, never fully erased
SECONDS_PER_DAY = 86400.0


def compute_confidence(fact: Dict[str, Any], now: float = None) -> float:
    """
    Returns a confidence value in [MIN_CONFIDENCE, 1.0] representing how
    'current' this fact still is.
    """
    if fact.get("permanent"):
        return 1.0

    now = now or time.time()
    elapsed_days = max(0.0, (now - fact["last_reinforced_at"]) / SECONDS_PER_DAY)

    importance = fact.get("importance", 0.5)
    reinforce_count = max(1, fact.get("reinforce_count", 1))

    # Higher importance and more reinforcement both slow decay.
    # importance in [0,1] -> multiplier in [1, 3]
    # reinforcement gives diminishing-returns extension via log.
    effective_half_life = BASE_HALF_LIFE_DAYS * (1 + 2 * importance) * (1 + math.log(reinforce_count))

    confidence = 0.5 ** (elapsed_days / effective_half_life)
    return max(MIN_CONFIDENCE, confidence)


def retrieval_score(fact: Dict[str, Any], similarity: float, now: float = None) -> float:
    """
    Combined ranking score used to select which facts make it into the
    limited context window. Blends:
      - similarity: how relevant this fact is to the current query
      - confidence: how 'current'/undecayed the fact still is
      - importance: base weight assigned at extraction time
    """
    confidence = compute_confidence(fact, now=now)
    importance = fact.get("importance", 0.5)
    # weighted geometric-ish blend so a fact needs to be decent on ALL
    # three axes, not just dominant on one
    return (similarity ** 0.5) * (confidence ** 0.35) * (0.5 + 0.5 * importance)


def pack_into_budget(scored_facts, token_budget: int, avg_tokens_per_fact: int = 25):
    """
    Greedily select highest-scoring facts until the token budget is
    exhausted. This directly answers the brief's requirement to recall
    critical memories within a limited context window rather than
    dumping the full memory store into every prompt.
    """
    max_facts = max(1, token_budget // avg_tokens_per_fact)
    ranked = sorted(scored_facts, key=lambda f: f["score"], reverse=True)
    return ranked[:max_facts]
