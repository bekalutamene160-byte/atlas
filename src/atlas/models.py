"""
ATLAS Pydantic schemas — the contract layer.

Every boundary in ATLAS has a Pydantic model:
- LLM input/output
- Tool input/output
- API request/response
- Internal state

Design principles:
1. Shape validation (types, required fields) is automatic via Pydantic.
2. Semantic validation (e.g., "retrieve action requires a query") is explicit
   via @model_validator.
3. Every schema is versioned via a schema_version field so old traces can
   be migrated when schemas change.

If you add a new boundary (e.g., a new tool), add a schema here. Do not
pass dicts across boundaries — they are untyped and will break at runtime
in ways that are hard to debug.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Token Usage (provider-neutral)
# =============================================================================

class TokenUsage(BaseModel):
    """Token usage for a single LLM call. Provider-neutral.

    Each provider adapter maps its SDK-specific fields to this model.
    The agent layer never sees Groq-specific names like `completion_tokens`.
    This is the fix for the "leaky abstraction" failure mode.
    """
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    model_name: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# =============================================================================
# Corpus & Retrieval
# =============================================================================

class Chunk(BaseModel):
    """A chunk of a document. The atomic unit of the corpus."""
    chunk_id: str
    source: str = Field(description="Document source path or URL")
    page: Optional[int] = None
    section: Optional[str] = None
    text: str
    chunk_index: int = Field(ge=0, description="Position in source document")

    @field_validator("text")
    @classmethod
    def text_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chunk text cannot be empty")
        return v.strip()


class RetrievedChunk(Chunk):
    """A chunk returned by the retriever, with a similarity score."""
    score: float = Field(ge=0.0, le=1.0, description="Cosine similarity 0..1")


class Evidence(BaseModel):
    """A piece of evidence gathered during a run.

    Evidence wraps a RetrievedChunk with the query that found it and a
    unique evidence_id for citation (e.g., "e1", "e2").
    """
    evidence_id: str = Field(description="Citation ID, e.g., 'e1'")
    chunk: RetrievedChunk
    query_used: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Tool Protocol
# =============================================================================

class ToolCall(BaseModel):
    """A request from the planner to execute a tool."""
    name: str
    arguments: dict
    reasoning: str = Field(description="Why this tool was chosen, 1 sentence")

    @field_validator("name")
    @classmethod
    def name_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tool name cannot be empty")
        return v.strip()


class ToolResult(BaseModel):
    """The result of executing a tool."""
    success: bool
    data: dict = Field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = Field(ge=0.0)


# =============================================================================
# Agent Decisions (structured output from LLM)
# =============================================================================

class PlanStep(BaseModel):
    """The planner's decision for the next action.

    Semantic validators enforce:
    - retrieve action requires a non-empty query
    - synthesize action must NOT carry a query
    - refuse action requires a meaningful reasoning (min 20 chars)
    """
    action: Literal["retrieve", "synthesize", "refuse"]
    query: Optional[str] = None
    reasoning: str

    @model_validator(mode="after")
    def check_action_query_consistency(self) -> PlanStep:
        if self.action == "retrieve":
            if not self.query or not self.query.strip():
                raise ValueError(
                    f"action='{self.action}' requires a non-empty query"
                )
        elif self.action == "synthesize":
            if self.query is not None:
                raise ValueError(
                    "action='synthesize' should not carry a query"
                )
        elif self.action == "refuse":
            if len(self.reasoning) < 20:
                raise ValueError(
                    "refusal reasoning must explain why (min 20 chars), "
                    f"got: '{self.reasoning}'"
                )
        return self


class CriticVerdict(BaseModel):
    """The critic's verdict on a single claim."""
    claim_text: str
    supported: bool
    supporting_evidence_id: Optional[str] = None
    issue: Optional[str] = None


class CriticReport(BaseModel):
    """The critic's full report on a synthesized answer."""
    verdicts: list[CriticVerdict]
    all_supported: bool
    issues: list[CriticVerdict] = Field(default_factory=list)


# =============================================================================
# Agent State
# =============================================================================

class AgentState(BaseModel):
    """The mutable state of a single agent run.

    Passed by reference to every component. Components read state but do
    not mutate it directly — they return values, and the loop integrates
    those values into state. This keeps the loop as the single mutation
    point.
    """
    query: str
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    retrieved_evidence: list[Evidence] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    steps_taken: int = 0
    status: Literal[
        "planning", "retrieving", "synthesizing",
        "criticizing", "done", "refused"
    ] = "planning"
    failures: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_evidence(self, ev: Evidence) -> None:
        """Add evidence, deduplicating by chunk_id."""
        existing_ids = {e.chunk.chunk_id for e in self.retrieved_evidence}
        if ev.chunk.chunk_id not in existing_ids:
            self.retrieved_evidence.append(ev)

    def add_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)

    def add_failure(self, failure: str) -> None:
        self.failures.append(failure)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


# =============================================================================
# API Models
# =============================================================================

class AskRequest(BaseModel):
    """POST /ask request body."""
    query: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    """A citation in an answer."""
    evidence_id: str
    source: str
    page: Optional[int] = None
    section: Optional[str] = None


class AgentResponse(BaseModel):
    """Base class for all responses."""
    run_id: str
    query: str
    status: str
    steps_taken: int
    evidence_count: int
    cost_tokens: int


class SuccessResponse(AgentResponse):
    """The agent successfully produced a cited answer."""
    status: Literal["done"] = "done"
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class RefusalResponse(AgentResponse):
    """The agent refused to answer (evidence insufficient)."""
    status: Literal["refused"] = "refused"
    reason: str


class PartialResponse(AgentResponse):
    """The agent hit a bound (max_steps, timeout, budget) before finishing."""
    status: Literal["partial"] = "partial"
    reason: str
    partial_answer: Optional[str] = None


class ErrorResponse(BaseModel):
    """The agent encountered a system error."""
    run_id: Optional[str] = None
    error_code: str
    error_message: str
