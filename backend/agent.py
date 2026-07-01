"""
MedMemory agent — the conversational layer.

Per-turn flow:
  1. Retrieve a budgeted set of relevant memories for the incoming message.
  2. Build a system prompt that includes ONLY those memories (not full
     history) plus an instruction on how to handle any contradiction
     events surfaced this turn.
  3. Extract + store new facts from the message (runs in parallel with
     step 2 conceptually; here sequential for simplicity/clarity).
  4. Generate the reply.

This keeps every turn to exactly 2 Qwen calls (extract, respond),
regardless of how long the user's history is — the memory budget is
what scales, not the token cost.
"""

from typing import Dict, Any, List

from backend.memory.manager import MemoryManager
from backend.api.qwen_client import QwenClient

SYSTEM_TEMPLATE = """You are MedMemory, a persistent-memory personal health
assistant. You are NOT a doctor and must not diagnose — encourage
professional care for anything serious. Use the memory below naturally,
the way a doctor who has seen this patient before would, without
narrating that you are "retrieving memories."

Known facts about this patient (ranked by relevance, may be incomplete):
{memory_block}

{change_note}

Respond warmly, briefly, and specifically — reference relevant facts
naturally rather than listing them.
"""


class MedMemoryAgent:
    def __init__(self, memory_manager: MemoryManager = None, qwen_client: QwenClient = None):
        self.qwen = qwen_client or QwenClient()
        self.memory = memory_manager or MemoryManager(qwen_client=self.qwen)

    def _format_memory_block(self, facts: List[Dict[str, Any]]) -> str:
        if not facts:
            return "(no relevant memories yet — this may be a new patient or new topic)"
        lines = []
        for f in facts:
            tag = "[PERMANENT]" if f.get("permanent") else f"[conf={f['score']:.2f}]"
            lines.append(f"- {tag} ({f['fact_type']}) {f['content']}")
        return "\n".join(lines)

    def _format_change_note(self, events: List[Dict[str, Any]]) -> str:
        updates = [e for e in events if e["action"] == "updated"]
        adds = [e for e in events if e["action"] == "added"]
        if not updates and not adds:
            return ""
        parts = []
        if updates:
            for u in updates:
                parts.append(
                    f"Note: the patient just updated a fact — previously "
                    f"'{u['old_content']}', now '{u['new_content']}'. "
                    f"Acknowledge this change explicitly in your reply."
                )
        if adds:
            parts.append(f"New facts just recorded this turn: " +
                         "; ".join(a["content"] for a in adds))
        return "\n".join(parts)

    def chat(self, user_id: str, message: str, token_budget: int = 300) -> Dict[str, Any]:
        # 1) Retrieve budgeted memory relevant to this message
        relevant_facts = self.memory.retrieve(user_id, message, token_budget=token_budget)

        # 2) Extract + store new facts from this message (with contradiction resolution)
        events = self.memory.extract_and_store(user_id, message)

        # 3) Build prompt and generate reply
        memory_block = self._format_memory_block(relevant_facts)
        change_note = self._format_change_note(events)
        system_prompt = SYSTEM_TEMPLATE.format(memory_block=memory_block, change_note=change_note)

        reply = self.qwen.chat(system_prompt=system_prompt, user_prompt=message, temperature=0.4)

        return {
            "reply": reply,
            "memories_used": relevant_facts,
            "memory_events": events,
        }
