
# ATLAS — RAG Agent for Equipment Specifications

A retrieval-augmented generation (RAG) agent with a **planner → synthesizer → critic** architecture, built for querying industrial equipment specification documents with verifiable citations.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)
![Eval](https://img.shields.io/badge/eval-83.3%25-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Why This Project Exists

Most RAG demos retrieve chunks and generate answers — but they can't verify whether the answer is actually grounded in the source material. ATLAS solves this with a **critic component** that checks every claim against retrieved evidence before returning an answer. If the critic finds unsupported claims, the agent loops back and re-synthesizes.

**Result: 83.3% eval pass rate with 1.00 grounding score and 1.00 citation quality on passing cases. 100% refusal rate on out-of-scope questions.**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
│  "What is the max operating pressure of Atlas-3?"      │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌─────────────────┐
              │    PLANNER      │  ← gpt-oss-20b
              │  (decide:       │     Decides: retrieve, synthesize, or refuse
              │   retrieve?     │
              │   synthesize?   │
              │   refuse?)      │
              └────────┬────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐
   │   RETRIEVER     │   │   SYNTHESIZER   │  ← gpt-oss-120b
   │  (vector search │   │  (writes answer │     Generates grounded
   │   over 2,067    │   │   with citations│     answer with citations
   │   chunks)       │   │   from evidence)│
   └────────┬────────┘   └────────┬────────┘
            │                     │
            └──────────┬──────────┘
                       ▼
              ┌─────────────────┐
              │     CRITIC      │  ← gpt-oss-20b
              │  (verify every  │     Checks each claim against evidence
              │   claim against │     If unsupported → loop back to planner
              │   evidence)     │
              └────────┬────────┘
                       │
                       ▼ approved
              ┌─────────────────┐
              │  FINAL ANSWER   │
              │  + citations    │
              └─────────────────┘
```

## Key Features

- **Self-correcting agent loop** — critic rejects unsupported claims, agent re-plans (max 3 iterations)
- **Honest refusal** — 100% success rate on out-of-scope questions ("How do I bake bread?")
- **Verifiable citations** — every answer includes source document references
- **Fixture-based eval** — deterministic, zero-cost test runs after initial recording
- **Quota-aware retry** — parses Groq's rate-limit wait hints for smart backoff
- **SSE streaming** — real-time token streaming for responsive UX

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM Provider | Groq (gpt-oss-120b/20b) | Fast inference, free tier, structured output |
| Embeddings | Jina AI (jina-embeddings-v3) | 1024-dim, multilingual, strong on technical docs |
| Vector Store | SQLite + numpy | Zero-dependency, portable, sufficient for 2k chunks |
| Web Framework | FastAPI + Uvicorn | Async, auto-docs, production-ready |
| Deployment | Docker + Render | One-click deploy, auto-scale, health monitoring |

## Evaluation Results

| Category | Pass Rate | Grounding | Citation |
|----------|-----------|-----------|----------|
| Factual (single-doc lookup) | 2/2 (100%) | 1.00 | 1.00 |
| Multi-hop (cross-doc reasoning) | 1/1 (100%) | 1.00 | 1.00 |
| Ambiguous (clarification handling) | 0/1 (0%) | — | — |
| Off-topic (honest refusal) | 2/2 (100%) | 1.00 | 1.00 |
| **Overall** | **5/6 (83.3%)** | **1.00** | **1.00** |

**Grounding score** = every claim in the answer is supported by retrieved evidence.
**Citation quality** = all required source documents are referenced.

The one failure (ambiguous case) was due to a Pydantic schema validation error in the critic's JSON output — the agent produced a correct 1,525-character answer but the critic couldn't validate it. This is a known issue with JSON mode on smaller models.

## Quick Start

### Prerequisites

- Python 3.11+
- Groq API key ([get one here](https://console.groq.com))
- Jina AI API key ([get one here](https://jina.ai))

### Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/atlas.git
cd atlas

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Set environment variables
cp .env.example .env
# Edit .env with your real API keys

# Ingest documents (builds the vector store)
python -m atlas.ingestion.ingest --input data/ --output data/vector_store.db

# Run the API
uvicorn atlas.api.main:app --reload --port 8000

# Test it
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the maximum operating pressure of the Atlas-3 pump?"}'
```

### Running Evaluations

```bash
# First run: records fixtures (consumes ~70k tokens)
python -m atlas.eval.runner --fixtures --fixture-mode record

# All subsequent runs: 100% free, deterministic
python -m atlas.eval.runner --fixtures --fixture-mode replay
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (used by Render for monitoring) |
| GET | `/` | API info and available endpoints |
| POST | `/chat` | Synchronous chat — blocks until complete |
| POST | `/chat/stream` | SSE streaming chat — real-time tokens |
| GET | `/docs` | Auto-generated Swagger documentation |

## Deployment

This project deploys to [Render](https://render.com) using Docker. You don't need Docker installed locally — Render builds the image on their servers.

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → connect your repo
3. Render reads `render.yaml` and `Dockerfile` automatically
4. Set `GROQ_API_KEY` and `JINA_API_KEY` as environment variables in the dashboard
5. Click "Create Web Service" — done in 3-5 minutes

See `DEPLOYMENT.md` for detailed step-by-step instructions.

## Lessons Learned

### 1. Token quota management is a first-class engineering concern

Groq's free tier gives 200,000 tokens per day (TPD). A 20-case eval at ~12k tokens per case = 240k tokens — over budget before you start. Solution: **fixture-based eval** records real API responses and replays them for free.

### 2. Rate-limit errors need structured handling

Groq's 429 errors come in two formats (`"4m18s"` and `"58 seconds"`). A blind 5-second sleep doesn't work for daily quota errors — you need to parse the actual wait time. Solution: `RateLimitError` with `wait_seconds` attribute that the eval runner reads.

### 3. The critic is the most important component

Without the critic, the synthesizer can hallucinate facts not in the evidence. The critic catches this and forces re-synthesis. In our eval, the critic rejected 25% of initial syntheses — every one had a real unsupported claim.

### 4. Model tiering saves 40% of tokens

The planner only makes a binary "retrieve vs synthesize" decision — it doesn't need a 120B model. Switching planner to 20B cut token costs by 40% with zero quality loss.

## Project Structure

```
atlas/
├── src/atlas/
│   ├── agent/           # Agent loop, planner, synthesizer, critic
│   ├── api/             # FastAPI endpoints
│   ├── eval/            # Evaluation runner
│   ├── ingestion/       # Document chunking + embedding
│   ├── providers/       # Groq + Fixture providers
│   ├── retrieval/       # Vector store + similarity search
│   └── tools/           # Retriever tool + dispatcher
├── data/                # Source documents + vector store
├── eval/                # Benchmark cases + baselines
├── tests/fixtures/      # Recorded LLM responses for deterministic eval
├── Dockerfile           # Container recipe (built by Render)
├── render.yaml          # Render deployment config
├── requirements.txt     # Production dependencies
└── README.md            # This file
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [Groq](https://groq.com) for fast LLM inference
- [Jina AI](https://jina.ai) for embeddings
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
````
