"""
Qwen Cloud API client.

Wraps calls to Qwen Cloud's OpenAI-compatible chat completions endpoint.
Supports a MOCK_MODE so the rest of the system (memory manager, decay logic,
contradiction resolution) can be built and tested end-to-end WITHOUT spending
any API credits. Flip QWEN_MOCK_MODE=false once real credentials are in place.
"""

import os
import json
import requests
from typing import Optional


QWEN_API_BASE = os.environ.get(
    "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
MOCK_MODE = os.environ.get("QWEN_MOCK_MODE", "true").lower() == "true"


class QwenClient:
    """Thin wrapper around the Qwen Cloud chat completions API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or QWEN_API_KEY
        self.model = model or QWEN_MODEL
        self.mock_mode = MOCK_MODE or not self.api_key

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2,
              max_tokens: int = 512, json_mode: bool = False) -> str:
        """
        Send a single-turn (system + user) request to Qwen.
        Returns the raw text content of the model's reply.
        """
        if self.mock_mode:
            return self._mock_response(system_prompt, user_prompt, json_mode)

        url = f"{QWEN_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Mock responses — deterministic-ish stand-ins so the pipeline can be
    # developed/tested for free. These are intentionally simple; swap
    # MOCK_MODE off to exercise the real model.
    # ------------------------------------------------------------------
    def _mock_response(self, system_prompt: str, user_prompt: str, json_mode: bool) -> str:
        if json_mode:
            # Fact-extraction mock: pretend to pull out a plausible fact.
            return json.dumps({
                "facts": [
                    {
                        "type": "symptom",
                        "content": "mock-extracted fact from: " + user_prompt[:60],
                        "importance": 0.5,
                        "permanent": False,
                    }
                ]
            })
        return (
            "[MOCK RESPONSE — set QWEN_API_KEY and QWEN_MOCK_MODE=false for real output] "
            f"Acknowledged: {user_prompt[:80]}"
        )
