"""
Local development server.

Runs the exact same WSGI handler used in Alibaba Cloud Function Compute
(backend/fc_handler.py:http_handler), so testing locally is a faithful
preview of the deployed behavior — no separate "local mode" code path
to drift out of sync.

Usage:
    python -m backend.local_server            # serves on :8000
    QWEN_MOCK_MODE=false python -m backend.local_server   # hits real Qwen
"""

import os
from wsgiref.simple_server import make_server
from backend.fc_handler import http_handler

PORT = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    print(f"MedMemory backend (local dev) running on http://localhost:{PORT}")
    print(f"QWEN_MOCK_MODE={os.environ.get('QWEN_MOCK_MODE', 'true')}")
    print("POST {\"user_id\": \"...\", \"message\": \"...\"} to / to chat")
    with make_server("", PORT, http_handler) as httpd:
        httpd.serve_forever()
