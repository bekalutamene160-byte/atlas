# ATLAS

> **A**gentic research and **T**ools for **L**ookup, **A**nalysis, and **S**ynthesis.
> An evidence-grounded research agent for industrial maintenance documentation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

ATLAS takes a complex research request over a bounded corpus of industrial
maintenance documentation, plans an investigation, retrieves evidence, analyzes
it, and produces a cited answer with a traceable evidence chain — refusing
when evidence is insufficient.

## What makes ATLAS different

- **Grounded**: every claim in every answer is tied to a specific evidence chunk with a clickable citation.
- **Honest**: the agent refuses when the corpus does not address the question, instead of hallucinating.
- **Verifiable**: every run produces a JSONL trace that can be replayed for debugging.
- **Measurable**: a 20-case eval suite with 4 metrics runs in CI; regressions block deploys.

## Quick start

```powershell
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/atlas.git
cd atlas
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Configure (mock mode works with no API keys)
Copy-Item .env.example .env
# Edit .env to add your Groq and HuggingFace tokens

# 3. Run tests (mock mode — no internet needed)
pytest tests/unit -v

# 4. Run server (Day 5+)
# make dev
```

## Architecture

ATLAS is a single-agent system with a bounded loop:

```
User query
    ↓
Planner → {retrieve | synthesize | refuse}
    ↓
Retriever → evidence chunks
    ↓
Synthesizer → cited draft answer
    ↓
Critic → verify each claim
    ↓
Done | Refuse | Re-retrieve
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.

## Evaluation

The 20-case eval suite covers four categories:

| Category | Count | Tests |
|---|---|---|
| Factual (single-hop) | 8 | Basic retrieval + citation |
| Multi-hop | 4 | Synthesis across chunks |
| Ambiguous | 4 | Clarification behavior |
| Off-topic (must refuse) | 4 | Refusal behavior |

## Tech stack

- **LLM**: Groq Llama 3.3 70B (planner, synthesizer) + Llama 3.1 8B (critic)
- **Embeddings**: HuggingFace BGE-small-en-v1.5 (primary) + sentence-transformers MiniLM (fallback)
- **Vector store**: in-memory numpy with SQLite persistence
- **API**: FastAPI + uvicorn
- **Deploy**: Render free tier (Docker) + local fallback

## Project structure

```
atlas/
├── src/atlas/         # source code
│   ├── config.py      # settings
│   ├── models.py      # Pydantic schemas
│   ├── providers/     # LLM provider abstraction
│   ├── ingestion/     # document parsing + chunking + embeddings
│   ├── retrieval/     # vector store + retriever
│   ├── tools/         # tool protocol + dispatcher
│   ├── agent/         # loop, planner, synthesizer, critic
│   ├── observability/ # tracer + cost tracker
│   └── api/           # FastAPI app + UI
├── tests/             # unit, integration, eval, security tests
├── eval/              # benchmark cases + baselines
└── data/              # corpus, SQLite DB, embedding cache (gitignored)
```

## License

MIT — see [LICENSE](LICENSE).