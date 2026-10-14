"""Unit tests for MockProvider."""

from __future__ import annotations

import pytest

from atlas.exceptions import ProviderError
from atlas.models import PlanStep, TokenUsage
from atlas.providers.mock import MockProvider


class TestMockProviderText:
    """Tests for generate_text."""

    async def test_returns_canned_response(self, mock_provider: MockProvider):
        """Default mock returns a string when queue is empty."""
        text, usage = await mock_provider.generate_text("system", "user")
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(usage, TokenUsage)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    async def test_returns_queued_responses_in_order(self):
        """Queued responses are returned FIFO."""
        mock = MockProvider(text_responses=["first", "second", "third"])
        r1, _ = await mock.generate_text("s", "u")
        r2, _ = await mock.generate_text("s", "u")
        r3, _ = await mock.generate_text("s", "u")
        assert r1 == "first"
        assert r2 == "second"
        assert r3 == "third"

    async def test_logs_calls(self, mock_provider: MockProvider):
        """Call log records method, system, user, model."""
        await mock_provider.generate_text("sys prompt", "user query", model="test-model")
        assert len(mock_provider.call_log) == 1
        log = mock_provider.call_log[0]
        assert log["method"] == "generate_text"
        assert "sys prompt" in log["system"]
        assert "user query" in log["user"]
        assert log["model"] == "test-model"


class TestMockProviderStructured:
    """Tests for generate_structured."""

    async def test_returns_validated_model(self):
        """Structured response is validated against the schema."""
        mock = MockProvider(
            structured_responses=[
                {"action": "retrieve", "query": "test query",
                 "reasoning": "need more evidence about the pump specs"}
            ]
        )
        response = await mock.generate_structured(PlanStep, "sys", "user")
        assert isinstance(response.data, PlanStep)
        assert response.data.action == "retrieve"
        assert response.data.query == "test query"

    async def test_default_refusal_when_queue_empty(self, mock_provider: MockProvider):
        """Empty queue returns a default refusal PlanStep."""
        response = await mock_provider.generate_structured(PlanStep, "sys", "user")
        assert response.data.action == "refuse"
        assert "insufficient" in response.data.reasoning.lower()

    async def test_raises_on_invalid_data(self):
        """Mock raises ProviderError if canned data doesn't match schema."""
        mock = MockProvider(
            structured_responses=[
                {"action": "retrieve", "query": None,
                 "reasoning": "missing query should fail validation"}
            ]
        )
        with pytest.raises(ProviderError):
            await mock.generate_structured(PlanStep, "sys", "user")


class TestMockProviderStream:
    """Tests for stream."""

    async def test_yields_tokens(self, mock_provider: MockProvider):
        """Stream yields tokens as async iterator."""
        tokens = []
        async for token in mock_provider.stream("sys", "user"):
            tokens.append(token)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    async def test_stream_text_is_split_on_words(self):
        """Stream splits the configured text on spaces."""
        mock = MockProvider(stream_text="hello world test")
        tokens = []
        async for t in mock.stream("s", "u"):
            tokens.append(t)
        assert "hello " in tokens
        assert "world " in tokens


class TestMockProviderFailureInjection:
    """Tests for the fail_on_call feature."""

    async def test_fails_on_configured_call(self):
        """fail_on_call=2 raises ProviderError on the 2nd call."""
        mock = MockProvider(fail_on_call=2)
        await mock.generate_text("s", "u")  # call 1, OK
        with pytest.raises(ProviderError):
            await mock.generate_text("s", "u")  # call 2, fails

    async def test_does_not_fail_on_other_calls(self):
        """fail_on_call=2 does not affect calls 1, 3, 4..."""
        mock = MockProvider(fail_on_call=2, text_responses=["a", "b", "c", "d"])
        await mock.generate_text("s", "u")
        with pytest.raises(ProviderError):
            await mock.generate_text("s", "u")
        await mock.generate_text("s", "u")
        await mock.generate_text("s", "u")


class TestMockProviderDeterministicEmbedding:
    """Tests for the _deterministic_embedding static method."""

    def test_same_text_same_vector(self):
        """Same input produces same output."""
        v1 = MockProvider._deterministic_embedding("test text")
        v2 = MockProvider._deterministic_embedding("test text")
        assert v1 == v2

    def test_different_text_different_vector(self):
        """Different inputs produce different outputs."""
        v1 = MockProvider._deterministic_embedding("hello")
        v2 = MockProvider._deterministic_embedding("world")
        assert v1 != v2

    def test_vector_dimensions(self):
        """Vector has the requested dimensions."""
        v = MockProvider._deterministic_embedding("test", dims=128)
        assert len(v) == 128
