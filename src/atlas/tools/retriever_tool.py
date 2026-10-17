"""
RetrieverTool — the agent's primary sensor.

Wraps the VectorStore as a Tool. When the planner emits
action="retrieve", the dispatcher calls this tool with the query.

The tool:
1. Embeds the query using the embedder
2. Searches the vector store for similar chunks
3. Returns the top-K chunks as ToolResult

Why this is a Tool (not called directly):
- The dispatcher validates arguments before execution
- The dispatcher logs every call for tracing
- The agent loop can swap tools without changing the planner
- Future tools (web_search, doc_analyzer) follow the same pattern

Design decision: the retriever returns RetrievedChunks, but the agent
loop converts them to Evidence objects (with evidence_id "e1", "e2"...)
This separation keeps the tool pure (returns data) and the loop as
the single place where state is mutated.
"""

from __future__ import annotations

import time
import logging

from pydantic import field_validator

from atlas.models import ToolArgs, ToolResult, RetrievedChunk
from atlas.tools.base import Tool
from atlas.ingestion.embedder import Embedder
from atlas.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrieveArgs(ToolArgs):
    """Arguments for the retriever tool.

    Validated by Pydantic BEFORE execution. If the LLM emits an empty
    query or a non-positive top_k, the dispatcher rejects it without
    calling execute().

    Bug fixes (Day 3 audit):
    - Empty/whitespace queries now rejected (was: silently accepted)
    - top_k <= 0 now rejected (was: produced empty/negative-slice results)
    """

    query: str
    top_k: int = 5

    @field_validator("query")
    @classmethod
    def query_must_be_non_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only queries.

        Without this, the LLM could emit query="" and the retriever
        would embed an empty string, get a meaningless vector, and
        return random chunks. Better to fail at the boundary.
        """
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_positive(cls, v: int) -> int:
        """Reject top_k <= 0.

        Without this, top_k=0 returns no chunks (useless), and
        top_k=-1 triggers Python's negative-slice behavior (returns
        the LAST chunk, which is wrong and confusing).
        """
        if v < 1:
            raise ValueError(f"top_k must be at least 1, got {v}")
        return v


class RetrieverTool(Tool):
    """Searches the document corpus for chunks relevant to a query."""

    name = "retrieve"
    description = (
        "Search the industrial maintenance corpus for chunks relevant "
        "to a natural-language query. Use this when the user asks a "
        "factual question about equipment specifications (pumps, motors, "
        "valves), maintenance procedures (lubrication, inspection, "
        "overhaul), or safety protocols (lockout-tagout, PPE). "
        "Returns up to top_k chunks with source, page, and similarity score."
    )
    args_model = RetrieveArgs

    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 min_similarity: float = 0.35):
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_similarity = min_similarity

    async def execute(self, args: RetrieveArgs) -> ToolResult:
        """Execute the retrieval.

        Steps:
        1. Embed the query (uses cache — same query = no API call)
        2. Search vector store with cosine similarity
        3. Filter by min_similarity (returns empty if nothing relevant)
        4. Return as ToolResult

        The tool never raises — it returns ToolResult(success=False)
        on error. This lets the agent loop handle failures gracefully
        instead of crashing.
        """
        start = time.monotonic()

        try:
            # Step 1: Embed the query
            query_vectors = await self.embedder.embed([args.query])
            if not query_vectors:
                return ToolResult(
                    success=False,
                    data={},
                    error="Embedding returned empty result",
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            query_vector = query_vectors[0]

            # Step 2: Search the vector store
            # The vector store validates dimensions internally
            chunks: list[RetrievedChunk] = self.vector_store.search(
                query_vector=query_vector,
                top_k=args.top_k,
                min_similarity=self.min_similarity,
            )

            latency_ms = (time.monotonic() - start) * 1000

            if not chunks:
                logger.info(
                    f"retrieve: no chunks found for query "
                    f"(min_similarity={self.min_similarity})"
                )
                return ToolResult(
                    success=True,
                    data={"chunks": [], "count": 0},
                    error=None,
                    latency_ms=latency_ms,
                )

            chunk_dicts = [c.model_dump() for c in chunks]
            logger.info(
                f"retrieve: found {len(chunks)} chunks "
                f"(top_score={chunks[0].score:.3f}, latency={latency_ms:.0f}ms)"
            )

            return ToolResult(
                success=True,
                data={"chunks": chunk_dicts, "count": len(chunks)},
                error=None,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"retrieve failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data={},
                error=f"retrieval_error: {type(e).__name__}: {str(e)}",
                latency_ms=(time.monotonic() - start) * 1000,
            )
