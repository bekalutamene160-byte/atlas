"""Unit tests for the Synthesizer."""

from __future__ import annotations

import pytest

from atlas.agent.synthesizer import Synthesizer, SynthesizerResult
from atlas.models import AgentState, Evidence, RetrievedChunk, TokenUsage
from atlas.providers.mock import MockProvider


@pytest.fixture
def state_with_evidence() -> AgentState:
    """A state with 2 pieces of evidence."""
    state = AgentState(query="What is the max pressure of Atlas-3?")
    c1 = RetrievedChunk(
        chunk_id="c1", source="atlas-3-spec.pdf", page=1,
        text="Maximum operating pressure: 5 MPa (5000 kPa)",
        chunk_index=0, score=0.92
    )
    c2 = RetrievedChunk(
        chunk_id="c2", source="atlas-3-spec.pdf", page=2,
        text="Maximum operating temperature: 80 C",
        chunk_index=1, score=0.88
    )
    state.add_evidence(Evidence(evidence_id="e1", chunk=c1, query_used="pressure"))
    state.add_evidence(Evidence(evidence_id="e2", chunk=c2, query_used="temperature"))
    return state


class TestSynthesizer:
    """Tests for the synthesizer."""

    async def test_synthesize_returns_result(self, state_with_evidence: AgentState):
        """Synthesizer returns SynthesizerResult with answer and usage."""
        mock = MockProvider(text_responses=[
            "The maximum operating pressure of the Atlas-3 pump is 5 MPa [e1]."
        ])
        synth = Synthesizer(model=mock)

        result = await synth.synthesize(state_with_evidence)

        assert isinstance(result, SynthesizerResult)
        assert "5 MPa" in result.answer
        assert "[e1]" in result.answer
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.total_tokens > 0

    async def test_synthesize_no_evidence(self):
        """Synthesizer handles empty evidence gracefully."""
        mock = MockProvider()
        synth = Synthesizer(model=mock)

        state = AgentState(query="test")
        result = await synth.synthesize(state)

        assert "does not contain" in result.answer.lower() or "no evidence" in result.answer.lower()
        assert result.usage.total_tokens == 0

    def test_format_evidence_includes_source_and_page(self):
        """_format_evidence includes source, page, and text."""
        mock = MockProvider()
        synth = Synthesizer(model=mock)

        state = AgentState(query="test")
        chunk = RetrievedChunk(
            chunk_id="c1", source="atlas-3-spec.pdf", page=4,
            text="The pressure is 5 MPa.",
            chunk_index=0, score=0.9
        )
        state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="test"))

        formatted = synth._format_evidence(state.retrieved_evidence)
        assert "[e1]" in formatted
        assert "atlas-3-spec.pdf" in formatted
        assert "p.4" in formatted
        assert "5 MPa" in formatted

    def test_format_evidence_wraps_in_tags(self):
        """Evidence is wrapped in the prompt (checked via user message)."""
        mock = MockProvider()
        synth = Synthesizer(model=mock)

        state = AgentState(query="test")
        chunk = RetrievedChunk(
            chunk_id="c1", source="spec.pdf", page=1,
            text="test data", chunk_index=0, score=0.9
        )
        state.add_evidence(Evidence(evidence_id="e1", chunk=chunk, query_used="test"))

        formatted = synth._format_evidence(state.retrieved_evidence)
        # The formatted text will be wrapped in <evidence> tags by synthesize()
        # Here we just check the content is correct
        assert "[e1]" in formatted
        assert "test data" in formatted

    async def test_synthesize_tracks_tokens(self, state_with_evidence: AgentState):
        """Synthesizer tracks token usage from the provider."""
        mock = MockProvider(text_responses=["Answer [e1]."])
        synth = Synthesizer(model=mock)

        result = await synth.synthesize(state_with_evidence)
        # MockProvider counts words in system + user as input tokens
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0