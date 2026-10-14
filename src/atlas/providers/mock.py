"""
MockProvider — deterministic canned responses for tests and dev.

This is the most important provider you will write. It enables:
1. Tests that run offline in milliseconds (no API calls, no quota burned)
2. Development when Groq is down or quota is exhausted
3. Reproducible eval runs (same inputs -> same outputs, every time)

Design principle: the mock must match real provider behavior. A mock that
only returns happy-path data is worse than no mock -- it hides bugs.

The mock supports:
- Canned text responses (deque, FIFO)
- Canned structured responses (deque of dicts, validated against the schema)
- Canned streaming responses (split into tokens)
- A call_log for test assertions ("was generate_structured called twice?")
- Failure injection (raise ProviderError on the Nth call)
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import deque
from typing import AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.exceptions import ProviderError
from atlas.models import TokenUsage
from atlas.providers.base import ModelProvider, ModelResponse

T = TypeVar("T", bound=BaseModel)


class MockProvider(ModelProvider):
    """Deterministic mock for tests. Returns canned responses."""

    def __init__(
        self,
        text_responses: Optional[list[str]] = None,
        structured_responses: Optional[list[dict]] = None,
        stream_text: Optional[str] = None,
        fail_on_call: Optional[int] = None,
    ):
        """Initialize the mock.

        Args:
            text_responses: Queue of responses for generate_text. FIFO.
            structured_responses: Queue of dicts for generate_structured.
                Each dict is validated against the requested response_model.
            stream_text: Text returned by stream(), split on spaces.
            fail_on_call: If set, raise ProviderError on the Nth call
                (1-indexed). Useful for testing retry logic.
        """
        self._text_q: deque[str] = deque(text_responses or ["mock text response"])
        self._structured_q: deque[dict] = deque(structured_responses or [])
        self._stream_text = stream_text or "mock streaming response token by token"
        self._fail_on_call = fail_on_call
        self._call_count = 0
        self.call_log: list[dict] = []  # for test assertions

    def _next_call_id(self) -> int:
        self._call_count += 1
        return self._call_count

    def _maybe_fail(self) -> None:
        """Raise ProviderError if configured to fail on this call."""
        if self._fail_on_call and self._call_count == self._fail_on_call:
            raise ProviderError(f"injected failure on call {self._call_count}")

    @staticmethod
    def _deterministic_embedding(text: str, dims: int = 384) -> list[float]:
        """Generate a deterministic embedding from text.

        Same text -> same vector, every time. This lets us test retrieval
        without calling an embedding API.
        """
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        return [((h >> (i % 256)) & 1) * 0.5 + 0.5 for i in range(dims)]

    async def generate_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        call_id = self._next_call_id()
        self._maybe_fail()
        self.call_log.append({
            "call_id": call_id,
            "method": "generate_text",
            "system": system[:200],
            "user": user[:200],
            "model": model,
        })

        if not self._text_q:
            response = "mock text response (queue exhausted)"
        else:
            response = self._text_q.popleft()

        await asyncio.sleep(0.001)  # simulate small latency

        usage = TokenUsage(
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=len(response.split()),
            model_name=model or "mock",
        )
        return response, usage

    async def generate_structured(
        self,
        response_model: Type[T],
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> ModelResponse[T]:
        call_id = self._next_call_id()
        self._maybe_fail()
        self.call_log.append({
            "call_id": call_id,
            "method": "generate_structured",
            "response_model": response_model.__name__,
            "system": system[:200],
            "user": user[:200],
            "model": model,
        })

        if self._structured_q:
            raw = self._structured_q.popleft()
        else:
            # Default: construct a PlanStep that refuses (safe fallback)
            if hasattr(response_model, "model_fields") and "action" in getattr(response_model, "model_fields", {}):
                raw = {"action": "refuse", "reasoning": "mock default refusal: insufficient evidence in corpus"}
            else:
                raw = {}

        try:
            data = response_model.model_validate(raw)
        except ValidationError as e:
            raise ProviderError(
                f"mock returned invalid data for {response_model.__name__}: {e}"
            ) from e

        usage = TokenUsage(
            input_tokens=len(system.split()) + len(user.split()),
            output_tokens=50,
            model_name=model or "mock",
        )

        return ModelResponse(data=data, usage=usage, latency_ms=1.0)

    async def stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        call_id = self._next_call_id()
        self._maybe_fail()
        self.call_log.append({
            "call_id": call_id,
            "method": "stream",
            "system": system[:200],
            "user": user[:200],
            "model": model,
        })

        for word in self._stream_text.split():
            await asyncio.sleep(0.001)
            yield word + " "
