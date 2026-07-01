# MedMemory

**A persistent-memory health assistant agent, built for the Global AI Hackathon Series with Qwen Cloud — Track 1: MemoryAgent.**

MedMemory remembers a patient's medications, allergies, diagnoses, and symptoms across sessions — not by dumping full chat history into every prompt, but through a memory system that **decays outdated facts, resolves contradictions when information changes, and retrieves only what's relevant within a fixed token budget.**

## Why this exists

Patients (especially in low-resource healthcare systems) often re-explain their history every time they see a provider — there's no continuity. A memory agent that persists context across sessions, forgets what's no longer relevant, and never loses safety-critical facts (like allergies) is a real, practical gap this project addresses.

## What makes the memory system non-trivial

Most "agent with memory" projects do naive RAG over raw chat history: embed every message, retrieve top-k by similarity, done. MedMemory goes further, addressing the three things the hackathon brief calls out explicitly:

| Requirement | How MedMemory addresses it |
|---|---|
| **Efficient storage & retrieval** | Facts are extracted into atomic, typed records (not raw messages), embedded with a fast offline hashing embedder (no external model download / API call needed for retrieval), and indexed in a vector store for semantic search. |
| **Timely forgetting of outdated information** | Every fact has a confidence score that **decays exponentially over time** unless reinforced (mentioned again). Half-life scales with importance and reinforcement count. `permanent` facts (allergies, chronic diagnoses) are pinned at full confidence forever — safety-critical info never fades. |
| **Recalling critical memories within limited context windows** | Retrieval never returns "everything relevant." It ranks candidates by `similarity × decayed_confidence × importance` and **greedily packs the highest-value facts into a fixed token budget** before they ever reach the prompt. |

On top of that: **contradiction resolution**. When a new fact conflicts with an existing one of the same type (e.g. "switched from Panadol to Ibuprofen"), MedMemory doesn't just add a second, contradicting fact — it explicitly supersedes the old one, keeps full history for auditability, and tells the agent to acknowledge the change in its reply.

## Architecture

```
┌─────────────────┐        HTTPS         ┌──────────────────────────────────────┐
│   Frontend       │ ───────────────────▶ │  Alibaba Cloud Function Compute       │
│  (static HTML/JS, │                     │  backend/fc_handler.py                │
│   Netlify)        │ ◀─────────────────── │                                       │
└─────────────────┘        JSON            │  ┌─────────────────────────────────┐  │
                                            │  │ MedMemoryAgent (backend/agent.py)│  │
                                            │  │  1. retrieve budgeted memories   │  │
                                            │  │  2. extract + store new facts    │  │
                                            │  │  3. generate reply via Qwen      │  │
                                            │  └──────────────┬────────────────────┘  │
                                            │                 │                       │
                                            │  ┌──────────────▼────────────────────┐  │
                                            │  │      MemoryManager                │  │
                                            │  │  (backend/memory/manager.py)      │  │
                                            │  │                                    │  │
                                            │  │  ┌──────────────┐ ┌──────────────┐│  │
                                            │  │  │ FactStore     │ │ VectorMemory ││  │
                                            │  │  │ (SQLite)      │ │ (Chroma,     ││  │
                                            │  │  │ structured     │ │  offline     ││  │
                                            │  │  │ facts +        │ │  hashing     ││  │
                                            │  │  │ history        │ │  embeddings) ││  │
                                            │  │  └──────────────┘ └──────────────┘│  │
                                            │  │            │                       │  │
                                            │  │   ┌────────▼─────────┐             │  │
                                            │  │   │ decay.py           │             │  │
                                            │  │   │ confidence scoring │             │  │
                                            │  │   │ + budgeted packing │             │  │
                                            │  │   └────────────────────┘             │  │
                                            │  └────────────────────────────────────┘  │
                                            │                 │                       │
                                            │        ┌────────▼─────────┐             │
                                            │        │  Qwen Cloud API   │             │
                                            │        │  (chat + fact     │             │
                                            │        │   extraction)     │             │
                                            │        └───────────────────┘             │
                                            └──────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data-flow write-up and [`docs/DEPLOY.md`](docs/DEPLOY.md) for Alibaba Cloud deployment steps.

## Project structure

```
medmemory/
├── backend/
│   ├── agent.py            # conversation loop: retrieve → extract → respond
│   ├── fc_handler.py        # Alibaba Cloud Function Compute entry point (proof of deployment)
│   ├── local_server.py      # local dev server, same handler as production
│   ├── api/
│   │   └── qwen_client.py   # Qwen Cloud API wrapper (with free mock mode)
│   └── memory/
│       ├── database.py      # SQLite structured fact store
│       ├── vector_store.py  # Chroma vector store, offline hashing embedder
│       ├── decay.py         # confidence decay + budgeted retrieval scoring
│       └── manager.py       # orchestration + contradiction resolution
├── frontend/
│   ├── index.html           # chat UI (no framework, deploys to Netlify)
│   └── config.js            # backend URL config
├── tests/
│   └── test_memory.py       # smoke tests for decay/contradiction/retrieval logic
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEPLOY.md
├── requirements.txt
├── netlify.toml
└── .env.example
```

## Running locally (free — mock mode by default)

No API key needed to explore the memory logic:

```bash
pip install -r requirements.txt --break-system-packages
python -m tests.test_memory        # verify decay/contradiction/retrieval logic
python -m backend.local_server     # runs backend on http://localhost:8000
```

Then open `frontend/index.html` in a browser (backend URL defaults to `localhost:8000` in `frontend/config.js`).

## Running with the real Qwen model

1. Copy `.env.example` to `.env` and fill in `QWEN_API_KEY` (from `home.qwencloud.com` → API Keys)
2. Set `QWEN_MOCK_MODE=false`
3. Re-run `python -m backend.local_server`

## Deployment

- **Backend**: Alibaba Cloud Function Compute — see [`docs/DEPLOY.md`](docs/DEPLOY.md). Proof-of-deployment file: [`backend/fc_handler.py`](backend/fc_handler.py).
- **Frontend**: Netlify, static publish of `frontend/`.

## Track & submission

Submitted to **Track 1: MemoryAgent**, Global AI Hackathon Series with Qwen Cloud.

## License

MIT — see [`LICENSE`](LICENSE).
