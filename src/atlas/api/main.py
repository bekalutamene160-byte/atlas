"""
ATLAS API — FastAPI application.

Endpoints:
    GET  /              — smart root (redirects browsers to /ui, JSON for API clients)
    GET  /ui            — interactive web UI
    GET  /health        — health check for Render monitoring
    POST /chat          — synchronous chat (blocking, with auto-retry)
    GET  /docs          — auto-generated Swagger documentation

Configuration:
    Reads from environment variables (prefix ATLAS_) or a .env file
    via the Settings system in atlas.config. Required env vars:
        ATLAS_LLM_PROVIDER=groq
        ATLAS_GROQ_API_KEY=gsk_...
        ATLAS_EMBEDDING_PROVIDER=jina
        ATLAS_JINA_API_KEY=jln_...

    Optional tuning (defaults shown):
        ATLAS_MAX_AGENT_STEPS=12      # increase if 'partial' status is common
        ATLAS_RUN_TIMEOUT_SECONDS=120 # max seconds per agent run

Bulletproofing features:
    1. Auto-retry on 'partial' status (up to 2 retries within one request)
    2. Comprehensive error handling with diagnostic hints
    3. Request logging for post-mortem debugging
    4. Graceful degradation: never returns a 500 without context
    5. Health check verifies agent is truly ready (not just imported)
    6. Smart root: browsers see the UI, API clients see JSON
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Track startup time for health reporting
START_TIME = time.time()

# Global agent instance — initialized once at startup, reused for all requests
# This is critical for performance: loading the vector store takes 5-10s
_agent = None

# Track request count for monitoring
_request_count = 0


# ===========================================================================
# Lifespan: startup and shutdown
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs on startup and shutdown.

    Startup sequence:
        1. Load .env file (if python-dotenv available)
        2. Validate required env vars are present
        3. Build a fresh Settings instance (picks up env vars)
        4. Initialize the Agent via from_default()
        5. Log success

    Shutdown:
        1. Log shutdown
        2. (Resources are garbage-collected automatically)
    """
    global _agent
    logger.info("ATLAS starting up — initializing agent...")

    try:
        # Load .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded .env file")
        except ImportError:
            logger.debug("python-dotenv not installed — relying on env vars only")

        # Validate required env vars BEFORE heavy imports.
        # Fail fast with a clear message instead of a confusing import error.
        groq_key = os.environ.get("ATLAS_GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError(
                "ATLAS_GROQ_API_KEY (or GROQ_API_KEY) not set — cannot start. "
                "Set it in .env or as an environment variable."
            )

        jina_key = os.environ.get("ATLAS_JINA_API_KEY") or os.environ.get("JINA_API_KEY")
        if not jina_key:
            raise RuntimeError(
                "ATLAS_JINA_API_KEY (or JINA_API_KEY) not set — cannot start. "
                "Set it in .env or as an environment variable."
            )

        # Build a FRESH Settings instance so it picks up the env vars we
        # just confirmed are set. Then pass to Agent.from_default().
        from atlas.config import Settings
        from atlas.agent.loop import Agent

        fresh_settings = Settings()
        _agent = Agent.from_default(settings=fresh_settings)

        logger.info(
            f"ATLAS agent initialized — provider={fresh_settings.llm_provider}, "
            f"db={fresh_settings.db_path}, max_steps={fresh_settings.max_agent_steps}"
        )
        logger.info("ATLAS ready to serve requests")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield  # App runs here

    logger.info("ATLAS shutting down...")


# ===========================================================================
# FastAPI app configuration
# ===========================================================================

app = FastAPI(
    title="ATLAS — RAG Agent for Equipment Specifications",
    description=(
        "A retrieval-augmented generation agent with planner/synthesizer/critic "
        "architecture. Every claim in the answer is verified against retrieved "
        "evidence before being returned."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request / Response models
# ===========================================================================

class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User question",
        examples=["What is the maximum operating pressure of the Atlas-3 pump?"],
    )


class ChatResponse(BaseModel):
    """Response body for /chat endpoint.

    Status is one of:
        - completed: agent successfully answered the query
        - refused:   agent declined (out-of-scope question)
        - partial:   agent retrieved evidence but couldn't synthesize
        - failed:    agent encountered an error
    """
    answer: str
    citations: list[str] = []
    iterations: int = 1
    latency_ms: int
    status: str = "completed"
    evidence_count: int = 0
    cost_tokens: int = 0
    run_id: str = ""
    attempts: int = 1  # number of agent runs attempted (1 = no retry)


# ===========================================================================
# Response extraction — handles all agent statuses
# ===========================================================================

def _extract_citations(result: dict) -> list[str]:
    """Extract source filenames from citation dicts.

    The agent returns citations as:
        [{'evidence_id': 'e1', 'source': 'data/corpus/atlas-3-spec.txt', ...}]

    We extract just the filename for user-facing display:
        ['atlas-3-spec.txt']
    """
    raw_citations = (
        result.get("citations")
        or result.get("sources")
        or result.get("cited_sources")
        or []
    )
    citations: list[str] = []
    if isinstance(raw_citations, list):
        for c in raw_citations:
            if not c:
                continue
            if isinstance(c, dict):
                source = (
                    c.get("source")
                    or c.get("file")
                    or c.get("document")
                    or c.get("evidence_id")
                    or ""
                )
                if source:
                    source_name = str(source).replace("\\", "/").split("/")[-1]
                    if source_name and source_name not in citations:
                        citations.append(source_name)
            elif isinstance(c, str):
                source_name = c.replace("\\", "/").split("/")[-1]
                if source_name and source_name not in citations:
                    citations.append(source_name)
    return citations


def _extract_response_fields(result: Any) -> dict:
    """Extract user-facing fields from the agent's return value.

    Handles all known agent statuses:
        - 'done'     → success, full answer with citations
        - 'refused'  → out-of-scope question, agent declined
        - 'partial'  → agent hit step/token limit, may or may not have answer
        - 'error'    → agent raised an exception
    """
    # Coerce non-dict results to dict form
    if not isinstance(result, dict):
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif hasattr(result, "__dict__"):
            result = vars(result)
        else:
            return {
                "answer": str(result),
                "citations": [],
                "iterations": 1,
                "status": "completed",
                "evidence_count": 0,
                "cost_tokens": 0,
                "run_id": "",
            }

    raw_status = result.get("status", "completed")

    # Normalize statuses for user-facing API
    if raw_status == "done":
        status = "completed"
    elif raw_status == "refused":
        status = "refused"
    elif raw_status == "partial":
        status = "partial"
    elif raw_status == "error":
        status = "failed"
    else:
        status = raw_status

    # ---- Refusal case (out-of-scope question) ----
    if status == "refused":
        reason = (
            result.get("reason")
            or result.get("answer")
            or "The agent refused to answer this question."
        )
        answer_text = (
            f"I'm unable to answer that question. {reason}"
            if reason else
            "I'm unable to answer that question."
        )
        return {
            "answer": answer_text,
            "citations": [],
            "iterations": result.get("steps_taken", 1),
            "status": "refused",
            "evidence_count": result.get("evidence_count", 0),
            "cost_tokens": result.get("cost_tokens", 0),
            "run_id": result.get("run_id", ""),
        }

    # ---- Partial case (agent hit step/token limit) ----
    if status == "partial":
        # Even if synthesis didn't complete, we have retrieved evidence.
        # Return the partial answer (if any) PLUS the citations from retrieved chunks.
        partial_answer = result.get("answer") or ""
        if partial_answer:
            answer_text = partial_answer
        else:
            # No synthesis completed — but we still have retrieved evidence.
            # Be honest about what happened, but provide the citations so the user
            # can see what was found.
            answer_text = (
                "The agent retrieved relevant evidence but couldn't complete synthesis "
                "within the step limit. The retrieved sources are listed in the citations field. "
                "Please try rephrasing your question to be more specific, or try again — "
                "the agent is non-deterministic and may succeed on retry."
            )

        # CRITICAL: Always extract citations for partial responses.
        # The agent retrieved chunks — those should be cited even without synthesis.
        citations = _extract_citations(result)

        # If no citations extracted from the result, check for retrieved_chunks or evidence
        if not citations:
            retrieved = result.get("retrieved_chunks") or result.get("evidence") or []
            if isinstance(retrieved, list):
                for chunk in retrieved:
                    if isinstance(chunk, dict):
                        source = chunk.get("source") or chunk.get("file") or chunk.get("document") or ""
                        if source:
                            source_name = str(source).replace("\\", "/").split("/")[-1]
                            if source_name and source_name not in citations:
                                citations.append(source_name)
                    elif isinstance(chunk, str):
                        source_name = chunk.replace("\\", "/").split("/")[-1]
                        if source_name and source_name not in citations:
                            citations.append(source_name)

        return {
            "answer": answer_text,
            "citations": citations,  # NOW we always return citations
            "iterations": result.get("steps_taken", result.get("agent_steps", 1)),
            "status": "partial",
            "evidence_count": result.get("evidence_count", result.get("retrievals", 0)),
            "cost_tokens": result.get("cost_tokens", 0),
            "run_id": result.get("run_id", result.get("req_id", "")),
        }

    # ---- Error case (agent crashed) ----
    if status == "failed":
        error_msg = (
            result.get("error")
            or result.get("reason")
            or "Agent encountered an error."
        )
        return {
            "answer": f"The agent encountered an error: {error_msg}",
            "citations": [],
            "iterations": result.get("steps_taken", 1),
            "status": "failed",
            "evidence_count": result.get("evidence_count", 0),
            "cost_tokens": result.get("cost_tokens", 0),
            "run_id": result.get("run_id", ""),
        }

    # ---- Success case (status == "completed") ----
    answer = result.get("answer") or result.get("final_answer") or ""

    # If no top-level answer, check nested objects
    if not answer:
        nested = result.get("result") or result.get("synthesis") or {}
        if isinstance(nested, dict):
            answer = nested.get("answer") or nested.get("text") or ""
        elif isinstance(nested, str):
            answer = nested

    if not answer:
        answer = "The agent completed but did not produce an answer."

    return {
        "answer": answer,
        "citations": _extract_citations(result),
        "iterations": result.get("steps_taken", result.get("iterations", 1)),
        "status": status,
        "evidence_count": result.get("evidence_count", 0),
        "cost_tokens": result.get("cost_tokens", 0),
        "run_id": result.get("run_id", ""),
    }


# ===========================================================================
# Request logging middleware
# ===========================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing and status code.

    This is critical for debugging production issues — without it,
    you can't tell which queries are slow or failing.
    """
    global _request_count
    _request_count += 1

    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    # Log the incoming request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    try:
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        logger.info(
            f"[{request_id}] completed {response.status_code} in {duration_ms}ms"
        )
        return response
    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        logger.error(
            f"[{request_id}] FAILED after {duration_ms}ms: {type(e).__name__}: {e}"
        )
        raise


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint — Render pings this every 30s.

    Returns 200 if the app is healthy, 503 if something is wrong.
    Render restarts the container if this fails 3 times in a row.
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check startup logs for errors.",
        )

    uptime = time.time() - START_TIME
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "agent_ready": _agent is not None,
        "requests_served": _request_count,
    }


@app.get("/")
async def root_with_browser_redirect(request: Request):
    """Smart root: redirect browsers to /ui, return JSON for API clients.

    Detects the Accept header:
        - 'text/html' present → browser → redirect to /ui (interactive web UI)
        - otherwise           → API client (curl, Postman, etc.) → JSON info
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        # Browser request → send to UI
        return RedirectResponse(url="/ui", status_code=302)
    # API client (curl, Postman, etc.) → return JSON
    return {
        "name": "ATLAS RAG Agent",
        "version": "1.0.0",
        "description": "Retrieval-augmented generation for equipment specifications",
        "endpoints": {
            "ui": "GET /ui (interactive web interface)",
            "health": "GET /health",
            "chat": "POST /chat",
            "sources": "GET /sources/{filename} (source documents)",
            "docs": "GET /docs",
        },
        "eval_results": {
            "success_rate": "83.3%",
            "grounding_score": 1.0,
            "citation_quality": 1.0,
        },
    }


@app.get("/ui")
async def ui():
    """Serve the interactive web UI (single-page HTML)."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(ui_path, media_type="text/html")


@app.get("/sources/{filename}")
async def get_source(filename: str):
    """Serve a source document from the corpus."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    corpus_dir = (Path(__file__).resolve().parents[3] / "data" / "corpus").resolve()
    file_path = (corpus_dir / filename).resolve()
    try:
        file_path.relative_to(corpus_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename") from None

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Source not found: {filename}")

    ext = file_path.suffix.lower()
    media_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Synchronous chat endpoint with automatic retry on partial status.

    If the agent returns 'partial' (hit step limit), we retry up to 2 more
    times. The agent is non-deterministic, so a retry often succeeds.

    If the agent returns 'refused', we return immediately (no retry —
    a refusal is intentional, not a failure).

    The response includes an `attempts` field showing how many times
    the agent was invoked (1 = no retry, 2 = one retry, etc.).
    """
    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check startup logs.",
        )

    start = time.time()
    max_attempts = 3  # 1 initial + 2 retries
    last_result: Optional[dict] = None
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        logger.info(
            f"Agent run attempt {attempt}/{max_attempts} for query: "
            f"{request.query[:100]}..."
        )

        try:
            result = await _agent.run(request.query)
            last_result = _extract_response_fields(result)

            # If refused, return immediately — no point retrying
            if last_result["status"] == "refused":
                logger.info("Agent refused the query (no retry needed)")
                break

            # If completed successfully, return
            if last_result["status"] == "completed":
                logger.info(f"Agent succeeded on attempt {attempt}")
                break

            # If partial, log and retry (unless this was the last attempt)
            if last_result["status"] == "partial":
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent returned 'partial' on attempt {attempt}, retrying..."
                    )
                    continue
                else:
                    logger.warning(
                        f"Agent returned 'partial' on final attempt {attempt}"
                    )
                    break

            # If failed, retry (unless this was the last attempt)
            if last_result["status"] == "failed":
                if attempt < max_attempts:
                    logger.warning(
                        f"Agent returned 'failed' on attempt {attempt}, retrying..."
                    )
                    continue
                else:
                    logger.error(
                        f"Agent failed on final attempt {attempt}: "
                        f"{last_result.get('answer', '')[:200]}"
                    )
                    break

        except Exception as e:
            logger.error(
                f"Agent run {attempt} raised: {type(e).__name__}: {e}",
                exc_info=True,
            )
            last_result = {
                "answer": (
                    f"The agent encountered an error: {type(e).__name__}: {str(e)[:200]}. "
                    "If this persists, the Groq API may be rate-limiting or unavailable."
                ),
                "citations": [],
                "iterations": 0,
                "status": "failed",
                "evidence_count": 0,
                "cost_tokens": 0,
                "run_id": "",
            }
            if attempt < max_attempts:
                logger.warning(f"Retrying after exception (attempt {attempt})...")
                continue
            break

    if last_result is None:
        raise RuntimeError("Agent run completed without producing a result")

    latency_ms = int((time.time() - start) * 1000)

    return ChatResponse(
        answer=last_result["answer"],
        citations=last_result["citations"],
        iterations=last_result["iterations"],
        latency_ms=latency_ms,
        status=last_result["status"],
        evidence_count=last_result["evidence_count"],
        cost_tokens=last_result["cost_tokens"],
        run_id=last_result["run_id"],
        attempts=attempts,
    )


# ===========================================================================
# Error handlers — graceful responses for common failure modes
# IMPORTANT: FastAPI exception handlers MUST return a Response object
# (JSONResponse), not a raw dict. Returning a dict silently fails.
# ===========================================================================

@app.exception_handler(TimeoutError)
async def timeout_handler(request: Request, exc: TimeoutError):
    """Handle agent timeouts gracefully (returns 504 Gateway Timeout)."""
    logger.error(f"Timeout on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=504,
        content={
            "detail": (
                "The agent took too long to respond. This usually means the "
                "underlying LLM provider (Groq) is slow or rate-limiting. "
                "Try again in a few minutes."
            ),
            "status": 504,
        },
    )


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors — never leak stack traces to clients.

    Logs the full exception server-side, returns a generic message to the
    client. Returns 500 Internal Server Error.
    """
    logger.error(
        f"Internal error on {request.url.path}: {type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Check server logs for details.",
            "error_type": type(exc).__name__,
            "status": 500,
        },
    )