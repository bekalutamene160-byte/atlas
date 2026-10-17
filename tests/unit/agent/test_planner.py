"""Unit tests for Planner."""

from __future__ import annotations

import pytest

from atlas.models import AgentState, Evidence, PlanStep, RetrievedChunk, TokenUsage
from atlas.providers.mock import MockProvider
from atlas.agent.planner import Planner, PlannerResult


class TestPlanner:
    """Tests for the planner."""

    async def test_decide_returns_planner_result(self):
        """Planner returns PlannerResult (decision + usage)."""
        mock = MockProvider(structured_responses=[
            {"action": "retrieve", "query": "pump specifications",
             "reasoning": "need to find the pump spec sheet"}
        ])
        planner = Planner(model=mock)

        state = AgentState(query="What is the max pressure of Atlas-3?")
        result = await planner.decide(state)

        # Bug fix 10: planner now returns PlannerResult, not PlanStep
        assert isinstance(result, PlannerResult)
        assert isinstance(result.decision, PlanStep)
        assert isinstance(result.usage, TokenUsage)
        assert result.decision.action == "retrieve"
        assert result.decision.query == "pump specifications"
        assert result.usage.total_tokens > 0

    async def test_decide_refuse_action(self):
        """Planner can emit refuse action."""
        mock = MockProvider(structured_responses=[
            {"action": "refuse",
             "reasoning": "corpus does not contain information about baking bread"}
        ])
        planner = Planner(model=mock)

        state = AgentState(query="How do I bake bread?")
        result = await planner.decide(state)

        assert result.decision.action == "refuse"

    async def test_decide_synthesize_action(self):
        """Planner can emit synthesize action."""
        mock = MockProvider(structured_responses=[
            {"action": "synthesize",
             "reasoning": "have enough evidence to answer the question"}
        ])
        planner = Planner(model=mock)

        state = AgentState(query="What is the max pressure?")
        chunk = RetrievedChunk(
            chunk_id="c1", source="spec.pdf", page=1,
            text="max pressure is 5 MPa", chunk_index=0, score=0.9
        )
        state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="pressure"))

        result = await planner.decide(state)
        assert result.decision.action == "synthesize"
        assert result.decision.query is None

    def test_build_user_message_includes_query(self):
        """User message includes the original query."""
        mock = MockProvider()
        planner = Planner(model=mock)

        state = AgentState(query="What is the max pressure of Atlas-3?")
        msg = planner._build_user_message(state)

        assert "What is the max pressure of Atlas-3?" in msg

    def test_build_user_message_includes_evidence_count(self):
        """User message includes evidence count."""
        mock = MockProvider()
        planner = Planner(model=mock)

        state = AgentState(query="test")
        chunk = RetrievedChunk(
            chunk_id="c1", source="spec.pdf", page=1,
            text="data", chunk_index=0, score=0.9
        )
        state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="test"))

        msg = planner._build_user_message(state)
        assert "Evidence gathered: 1 chunks" in msg

    def test_build_user_message_does_not_include_evidence_text(self):
        """User message does NOT include chunk text (security boundary)."""
        mock = MockProvider()
        planner = Planner(model=mock)

        state = AgentState(query="test")
        chunk = RetrievedChunk(
            chunk_id="c1", source="spec.pdf", page=1,
            text="SECRET: this text should not be in the planner prompt",
            chunk_index=0, score=0.9
        )
        state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="test"))

        msg = planner._build_user_message(state)
        assert "SECRET" not in msg
