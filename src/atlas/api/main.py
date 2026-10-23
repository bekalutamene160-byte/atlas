"""
ATLAS API — FastAPI application.

Endpoints:
    GET  /health        — health check for Render monitoring
    GET  /              — API info + available endpoints
    POST /chat          — synchronous chat (blocking)
    POST /chat/stream   — SSE streaming chat (non-blocking)

Pro tip: The /health endpoint is the most important one for production.
Render pings it every 30s. If it fails 3 times, Render restarts your
container. Without it, Render can't tell if your app is alive.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Track startup time for health reporting
START_TIME = time.time()

# Global agent instance — initialized once at startup, reused for all requests
# This is critical for performance: loading the vector store takes 5-10s
_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs on startup and shutdown.
    
    Startup: initialize the agent (loads vector store, connects to Groq)
    Shutdown: cleanup resources (close DB connections, etc.)
    """
    global _agent
    logger.info("ATLAS starting up — initializing agent...")
    
    try:
        from atlas.agent.loop import Agent
        from atlas.config import settings

        # Accept both direct env names and the app's ATLAS_ prefixed settings.
        groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("ATLAS_GROQ_API_KEY")
        if groq_key:
            settings.groq_api_key = groq_key

        jina_key = os.environ.get("JINA_API_KEY") or os.environ.get("ATLAS_JINA_API_KEY")
        if jina_key:
            settings.jina_api_key = jina_key

        if settings.llm_provider == "groq" and not settings.groq_api_key:
            raise RuntimeError("ATLAS_GROQ_API_KEY not set — cannot start")
        if settings.embedding_provider == "jina" and not settings.jina_api_key:
            raise RuntimeError("ATLAS_JINA_API_KEY not set — cannot start")

        _agent = Agent.from_default(settings)

        logger.info("ATLAS agent initialized — ready to serve requests")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # app runs here
    
    # Shutdown
    logger.info("ATLAS shutting down...")


# Create the FastAPI app with lifespan
app = FastAPI(
    title="ATLAS — RAG Agent for Equipment Specifications",
    description="A retrieval-augmented generation agent with planner/synthesizer/critic architecture",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for portfolio demo
# In production, restrict to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for /chat endpoints."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Max agent iterations")


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""
    answer: str
    citations: list[str] = []
    iterations: int
    latency_ms: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint — Render pings this every 30s.
    
    Returns 200 if the app is healthy, 503 if something is wrong.
    Render restarts the container if this fails 3 times in a row.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    uptime = time.time() - START_TIME
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "agent_ready": _agent is not None,
    }


@app.get("/")
async def root():
    """API info — shows available endpoints."""
    return {
        "name": "ATLAS RAG Agent",
        "version": "1.0.0",
        "description": "Retrieval-augmented generation for equipment specifications",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "docs": "GET /docs",
        },
        "eval_results": {
            "success_rate": "83.3%",
            "grounding_score": 1.0,
            "citation_quality": 1.0,
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Synchronous chat endpoint — blocks until the agent completes.
    
    For long-running queries, use /chat/stream instead.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start = time.time()
    
    try:
        result = await _agent.run(request.query)
        latency_ms = int((time.time() - start) * 1000)
        
        return ChatResponse(
            answer=getattr(result, "answer", str(result)),
            citations=getattr(result, "citations", []),
            iterations=getattr(result, "iterations", 1),
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {type(e).__name__}: {str(e)[:200]}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat — yields tokens as they're generated.
    
    Returns a StreamingResponse with Server-Sent Events.
    The client receives partial answers in real-time.
    """
    from fastapi.responses import StreamingResponse
    import json
    
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def event_generator():
        try:
            # Stream the synthesizer's output
            async for chunk in _agent.stream(request.query):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Send final event
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)[:200]})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )