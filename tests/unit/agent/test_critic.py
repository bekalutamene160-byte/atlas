"""Unit tests for the Critic."""

from __future__ import annotations

import pytest

from atlas.agent.critic import Critic, CriticResult
from atlas.models import AgentState, Evidence, RetrievedChunk, TokenUsage
from atlas.providers.mock import MockProvider


@pytest.fixture
def state_with_evidence() -> AgentState:
    """A state with 1 piece of evidence."""
    state = AgentState(query="What is the max pressure?")
    chunk = RetrievedChunk(
        chunk_id="c1", source="atlas-3-spec.pdf", page=1,
        text="Maximum operating pressure: 5 MPa",
        chunk_index=0, score=0.92
    )
    state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="pressure"))
    return state


class TestCritic:
    """Tests for the critic."""

    async def test_verify_returns_result(self, state_with_evidence: AgentState):
        """Critic returns CriticResult with report and usage."""
        mock = MockProvider(structured_responses=[
            {"verdicts": [
                {"claim_text": "The pressure is 5 MPa.",
                 "supported": True, "supporting_evidence_id": "e1"}
            ]}
        ])
        critic = Critic(model=mock)

        answer = "The maximum operating pressure is 5 MPa [e1]."
        result = await critic.verify(answer, state_with_evidence)

        assert isinstance(result, CriticResult)
        assert result.report.all_supported is True
        assert len(result.report.verdicts) == 1
        assert result.usage.total_tokens > 0

    async def test_verify_detects_unsupported_claim(self, state_with_evidence: AgentState):
        """Critic detects number mismatch."""
        mock = MockProvider(structured_responses=[
            {"verdicts": [
                {"claim_text": "The pressure is 200 MPa.",
                 "supported": False,
                 "issue": "number_mismatch: evidence says 5 MPa, claim says 200 MPa"}
            ]}
        ])
        critic = Critic(model=mock)

        answer = "The maximum operating pressure is 200 MPa [e1]."
        result = await critic.verify(answer, state_with_evidence)

        assert result.report.all_supported is False
        assert len(result.report.issues) == 1
        assert "number_mismatch" in result.report.issues[0].issue

    async def test_verify_no_evidence(self):
        """Critic handles no evidence gracefully."""
        mock = MockProvider()
        critic = Critic(model=mock)

        state = AgentState(query="test")
        result = await critic.verify("some answer", state)

        assert result.report.all_supported is False
        assert len(result.report.issues) > 0

    def test_extract_citations(self):
        """extract_citations finds all [eN] patterns."""
        answer = "The pressure is 5 MPa [e1]. The temp is 80 C [e2]. See also [e3]."
        cites = Critic.extract_citations(answer)
        assert cites == ["e1", "e2", "e3"]

    def test_extract_citations_empty(self):
        """extract_citations returns empty list for no citations."""
        cites = Critic.extract_citations("No citations here.")
        assert cites == []

    def test_extract_citations_with_other_brackets(self):
        """extract_citations only matches [eN], not [1] or [N]."""
        answer = "See [1] and [e1] but not [N] and [e2]."
        cites = Critic.extract_citations(answer)
        assert cites == ["e1", "e2"]

    async def test_verify_multiple_claims(self, state_with_evidence: AgentState):
        """Critic can verify multiple claims in one answer."""
        mock = MockProvider(structured_responses=[
            {"verdicts": [
                {"claim_text": "The pressure is 5 MPa.",
                 "supported": True, "supporting_evidence_id": "e1"},
                {"claim_text": "The pump was manufactured in 2020.",
                 "supported": False, "issue": "not_in_evidence: no date in evidence"}
            ]}
        ])
        critic = Critic(model=mock)

        answer = "The pressure is 5 MPa [e1]. The pump was manufactured in 2020."
        result = await critic.verify(answer, state_with_evidence)

        assert result.report.all_supported is False
        assert len(result.report.issues) == 1
        assert "not_in_evidence" in result.report.issues[0].issue