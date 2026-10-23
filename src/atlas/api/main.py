"""
FastAPI application — the ATLAS web service.

Endpoints:
- GET  /          → Web UI (HTML)
- GET  /health    → Health check (JSON)
- POST /ask       → Synchronous query (JSON response)
- POST /stream    → Streaming query (SSE events)
- GET  /runs/{id} → Fetch a past run's trace (JSON)

Design decisions:
1. The Agent is a singleton, created once at startup via get_agent().
   If real providers fail (e.g., Groq key missing), falls back to mock.

2. All endpoints are async. No blocking calls. The agent loop runs
   in the same event loop as the HTTP server.

3. Errors are NEVER exposed to the client. Internal errors return
   a generic 500 with error_code. The actual error is logged.

4. The /stream endpoint uses Server-Sent Events (SSE). Each event is
   a JSON object on a line prefixed with "data: ". The browser
   parses these and updates the UI in real-time.

5. CORS is enabled for all origins. For production, tighten this to
   your specific domain.

Pro tip for advanced projects: add rate limiting (slowapi), authentication
(JWT), and request validation (already handled by Pydantic + FastAPI).
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlas.agent.loop import Agent
from atlas.config import Settings, settings as default_settings
from atlas.models import AskRequest
from atlas.observability.tracer import Tracer

logger = logging.getLogger(__name__)

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="ATLAS",
    version="0.1.0",
    description="Agentic Research & Knowledge Intelligence System for industrial maintenance",
)

# CORS — allow all origins for development. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Static files (CSS, JS for the UI)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Tracer for run traces
tracer = Tracer(default_settings.logs_path)

# =============================================================================
# Agent Singleton (with graceful fallback)
# =============================================================================

_agent: Optional[Agent] = None
_agent_error: Optional[str] = None


def get_agent() -> Agent:
    """Get the singleton Agent instance.

    Tries to create with real providers first. If that fails (e.g.,
    Groq key missing, embedding model not downloaded), falls back
    to mock mode so the server always starts.
    """
    global _agent, _agent_error
    if _agent is None and _agent_error is None:
        try:
            _agent = Agent.from_default()
            logger.info("Agent created with real providers")
        except Exception as e:
            _agent_error = str(e)
            logger.warning(
                f"Failed to create agent with real providers: {e}. "
                f"Falling back to mock mode."
            )
            try:
                mock_settings = Settings(
                    llm_provider="mock",
                    embedding_provider="mock",
                    db_path=default_settings.db_path,
                    corpus_path=default_settings.corpus_path,
                )
                _agent = Agent.from_default(mock_settings)
                logger.info("Agent created with mock providers (fallback)")
            except Exception as e2:
                _agent_error = f"Mock also failed: {e2}"
                logger.error(f"Could not create agent even in mock mode: {e2}")

    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not available: {_agent_error}"
        )
    return _agent


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def index():
    """Serve the web UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "ATLAS API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns 200 if the service is running. Reports which providers
    are active.
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "llm_provider": default_settings.llm_provider,
        "embedding_provider": default_settings.embedding_provider,
        "mock_mode": default_settings.is_mock_mode(),
    }


@app.post("/ask")
async def ask(req: AskRequest, agent: Agent = Depends(get_agent)):
    """Synchronous query endpoint.

    Runs the full agent loop and returns the complete result as JSON.
    For streaming (real-time updates), use /stream instead.
    """
    run_id = str(uuid.uuid4())
    tracer.start_run(run_id, req.query)

    try:
        result = await agent.run(req.query)

        # Log completion
        tracer.log_completion(
            run_id=run_id,
            status=result.get("status", "unknown"),
            steps=result.get("steps_taken", 0),
            tokens=result.get("cost_tokens", 0),
            evidence_count=result.get("evidence_count", 0),
        )

        # Add run_id to the result (the agent generates its own,
        # but we use the tracer's for correlation)
        result["trace_run_id"] = run_id
        return result

    except Exception as e:
        logger.error(f"/ask failed: {e}", exc_info=True)
        tracer.log_completion(run_id, "error", 0, 0, error=str(e))
        raise HTTPException(status_code=500, detail="internal_error")


@app.post("/stream")
async def stream(req: AskRequest, agent: Agent = Depends(get_agent)):
    """Streaming query endpoint (Server-Sent Events).

    Returns a text/event-stream. Each event is a JSON object:
    - {"type": "start", "run_id": "...", "query": "..."}
    - {"type": "step", "step": 1, "status": "planning"}
    - {"type": "evidence", "count": 5}
    - {"type": "done", "result": {...}}
    - {"type": "refused", "result": {...}}
    - {"type": "error", "message": "..."}

    The browser parses these and updates the UI in real-time.
    """
    trace_run_id = str(uuid.uuid4())
    tracer.start_run(trace_run_id, req.query)

    async def event_stream():
        """Generate SSE events from the agent's streaming output."""
        try:
            async for event in agent.run_streaming(req.query):
                # Log to tracer (filter out non-serializable and duplicate keys)
                log_kwargs = {
                    k: v for k, v in event.items()
                    if k not in ("type", "step", "run_id")
                    and isinstance(v, (str, int, float, bool))
                }
                tracer.log_event(
                    run_id=trace_run_id,
                    event_type=event.get("type", "unknown"),
                    step=event.get("step", 0),
                    **log_kwargs,
                )

                # Yield as SSE
                yield f"data: {json.dumps(event)}\n\n"

            # Log completion
            tracer.log_completion(trace_run_id, "stream_complete", 0, 0)

        except Exception as e:
            logger.error(f"/stream failed: {e}", exc_info=True)
            error_event = {"type": "error", "message": "internal_error"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable Nginx buffering
            "Connection": "keep-alive",
        },
    )


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Fetch the trace of a past run.

    Returns all events logged during the run. Returns 404 if the
    run doesn't exist.
    """
    events = tracer.get_trace(run_id)
    if not events:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "events": events}


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the server: python -m atlas.api.main or just `atlas`"""
    import uvicorn
    uvicorn.run(
        "atlas.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # auto-reload on code changes (development only)
    )


if __name__ == "__main__":
    main()
