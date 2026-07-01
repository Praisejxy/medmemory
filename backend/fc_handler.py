"""
Alibaba Cloud Function Compute (FC) entry point.

This file is the required "Proof of Alibaba Cloud Deployment" artifact:
it demonstrates the MedMemory backend running as a Function Compute
HTTP-triggered function, using Alibaba Cloud's FC Python runtime handler
signature (`handler(environ, start_response)` for the HTTP trigger, or
the simpler `handler(event, context)` for an event trigger — both shown
below so this can be deployed either way depending on the trigger type
chosen in the FC console).

Deploy notes (see docs/DEPLOY.md for the full walkthrough):
  1. `fun deploy` or the FC console, runtime: python3.10, handler:
     backend.fc_handler.handler
  2. Set QWEN_API_KEY as an FC environment variable (Function Compute
     -> Configuration -> Environment Variables) — never hardcode it here.
  3. SQLite + Chroma persist to /tmp (FC's writable scratch space) or,
     for durability across cold starts, to a mounted NAS/OSS volume —
     see docs/DEPLOY.md.
"""

import json
import os

from backend.agent import MedMemoryAgent

# Function Compute's /tmp is the only writable path in the default
# sandbox; mount NAS for persistence across invocations in production.
DB_PATH = os.environ.get("MEDMEMORY_DB_PATH", "/tmp/medmemory.db")
CHROMA_PATH = os.environ.get("MEDMEMORY_CHROMA_PATH", "/tmp/chroma_store")

_agent = None


def _get_agent() -> MedMemoryAgent:
    global _agent
    if _agent is None:
        from backend.memory.manager import MemoryManager
        _agent = MedMemoryAgent(memory_manager=MemoryManager(db_path=DB_PATH, chroma_path=CHROMA_PATH))
    return _agent


def _handle_request(body: dict) -> dict:
    user_id = body.get("user_id", "anonymous")
    message = body.get("message", "")
    if not message:
        return {"error": "missing 'message' field"}

    agent = _get_agent()
    result = agent.chat(user_id=user_id, message=message)
    return {
        "reply": result["reply"],
        "memory_events": result["memory_events"],
        "memories_used": [
            {"type": m["fact_type"], "content": m["content"], "score": round(m["score"], 3)}
            for m in result["memories_used"]
        ],
    }


# --- Event-trigger style handler (simplest; used by API Gateway / DashScope-style triggers) ---
def handler(event, context):
    """Standard FC event handler signature."""
    try:
        if isinstance(event, (bytes, str)):
            body = json.loads(event)
        else:
            body = event
        result = _handle_request(body)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except Exception as e:  # noqa: BLE001 - top-level FC handler must not raise
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


# --- HTTP-trigger style handler (WSGI-compatible, for direct HTTP invocation) ---
def http_handler(environ, start_response):
    # CORS: the frontend is hosted separately on Netlify, so the FC
    # endpoint must allow cross-origin requests from it.
    cors_headers = [
        ("Access-Control-Allow-Origin", os.environ.get("CORS_ALLOW_ORIGIN", "*")),
        ("Access-Control-Allow-Methods", "POST, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
    ]

    if environ.get("REQUEST_METHOD") == "OPTIONS":
        start_response("204 No Content", cors_headers)
        return [b""]

    try:
        length = int(environ.get("CONTENT_LENGTH", 0) or 0)
        raw_body = environ["wsgi.input"].read(length) if length else b"{}"
        body = json.loads(raw_body or b"{}")
        result = _handle_request(body)
        status = "200 OK"
        response_body = json.dumps(result).encode("utf-8")
    except Exception as e:  # noqa: BLE001
        status = "500 Internal Server Error"
        response_body = json.dumps({"error": str(e)}).encode("utf-8")

    headers = cors_headers + [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(response_body))),
    ]
    start_response(status, headers)
    return [response_body]
