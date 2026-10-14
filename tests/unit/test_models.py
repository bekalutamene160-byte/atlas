"""Unit tests for Pydantic models (the contract layer)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas.models import (
    AgentState,
    Chunk,
    CriticVerdict,
    Evidence,
    PlanStep,
    RetrievedChunk,
    TokenUsage,
    ToolCall,
)


class TestChunk:
    """Tests for the Chunk model."""

    def test_valid_chunk(self):
        c = Chunk(chunk_id="c1", source="spec.pdf", page=1, text="hello", chunk_index=0)
        assert c.chunk_id == "c1"
        assert c.page == 1

    def test_empty_text_rejected(self):
        """Empty text fails validation."""
        with pytest.raises(ValidationError):
            Chunk(chunk_id="c1", source="spec.pdf", text="   ", chunk_index=0)

    def test_optional_fields_default_none(self):
        """page and section are optional."""
        c = Chunk(chunk_id="c1", source="spec.pdf", text="hello", chunk_index=0)
        assert c.page is None
        assert c.section is None


class TestPlanStep:
    """Tests for PlanStep semantic validators."""

    def test_retrieve_requires_query(self):
        """retrieve action without query fails."""
        with pytest.raises(ValidationError):
            PlanStep(action="retrieve", query=None, reasoning="need more")

    def test_retrieve_with_empty_query_fails(self):
        """retrieve action with empty query fails."""
        with pytest.raises(ValidationError):
            PlanStep(action="retrieve", query="   ", reasoning="need more")

    def test_retrieve_with_query_succeeds(self):
        """retrieve action with non-empty query succeeds."""
        p = PlanStep(action="retrieve", query="pump specs",
                     reasoning="need more evidence")
        assert p.action == "retrieve"
        assert p.query == "pump specs"

    def test_synthesize_must_not_carry_query(self):
        """synthesize action with a query fails."""
        with pytest.raises(ValidationError):
            PlanStep(action="synthesize", query="leftover query",
                     reasoning="ready to answer")

    def test_synthesize_without_query_succeeds(self):
        """synthesize action without query succeeds."""
        p = PlanStep(action="synthesize", reasoning="ready to answer")
        assert p.action == "synthesize"
        assert p.query is None

    def test_refuse_requires_meaningful_reasoning(self):
        """refuse action with short reasoning fails."""
        with pytest.raises(ValidationError):
            PlanStep(action="refuse", reasoning="no")

    def test_refuse_with_long_reasoning_succeeds(self):
        """refuse action with meaningful reasoning succeeds."""
        p = PlanStep(action="refuse",
                     reasoning="corpus does not contain this information")
        assert p.action == "refuse"


class TestAgentState:
    """Tests for AgentState."""

    def test_default_run_id_is_uuid(self):
        """run_id is auto-generated as a UUID."""
        s = AgentState(query="test")
        assert s.run_id is not None
        assert len(s.run_id) == 36  # UUID4 string length

    def test_add_evidence_deduplicates(self):
        """Adding the same chunk twice only keeps one."""
        s = AgentState(query="test")
        chunk = RetrievedChunk(chunk_id="c1", source="spec.pdf", text="hello",
                               chunk_index=0, score=0.9)
        ev = Evidence(evidence_id="e1", chunk=chunk, query_used="q")
        s.add_evidence(ev)
        s.add_evidence(ev)
        assert len(s.retrieved_evidence) == 1

    def test_add_evidence_different_chunks_both_kept(self):
        """Different chunks are both kept."""
        s = AgentState(query="test")
        c1 = RetrievedChunk(chunk_id="c1", source="a.pdf", text="hello",
                            chunk_index=0, score=0.9)
        c2 = RetrievedChunk(chunk_id="c2", source="b.pdf", text="world",
                            chunk_index=0, score=0.8)
        s.add_evidence(Evidence(evidence_id="e1", chunk=c1, query_used="q"))
        s.add_evidence(Evidence(evidence_id="e2", chunk=c2, query_used="q"))
        assert len(s.retrieved_evidence) == 2

    def test_tool_call_count_property(self):
        """tool_call_count reflects the number of tool calls."""
        s = AgentState(query="test")
        assert s.tool_call_count == 0
        s.add_tool_call(ToolCall(name="retrieve", arguments={"query": "x"},
                                  reasoning="testing"))
        assert s.tool_call_count == 1


class TestTokenUsage:
    """Tests for the provider-neutral TokenUsage model."""

    def test_total_tokens(self):
        u = TokenUsage(input_tokens=100, output_tokens=50, model_name="test")
        assert u.total_tokens == 150

    def test_zero_tokens_allowed(self):
        """Zero tokens is valid (edge case for empty responses)."""
        u = TokenUsage(input_tokens=0, output_tokens=0, model_name="test")
        assert u.total_tokens == 0

    def test_negative_tokens_rejected(self):
        """Negative token counts are invalid."""
        with pytest.raises(ValidationError):
            TokenUsage(input_tokens=-1, output_tokens=0, model_name="test")


class TestCriticVerdict:
    """Tests for CriticVerdict."""

    def test_supported_verdict(self):
        v = CriticVerdict(claim_text="the pump is 5 MPa", supported=True,
                          supporting_evidence_id="e1")
        assert v.supported is True
        assert v.issue is None

    def test_unsupported_verdict_with_issue(self):
        v = CriticVerdict(claim_text="the pump is 10 MPa", supported=False,
                          issue="number_mismatch: evidence says 5 MPa")
        assert v.supported is False
        assert v.supporting_evidence_id is None
