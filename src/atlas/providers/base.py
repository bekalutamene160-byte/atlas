"""
ModelProvider — the abstract interface for LLM providers.

The agent layer depends on this interface, NOT on any specific SDK.
This is what lets us swap Groq for OpenAI for a mock without touching
the agent code.

Design principle: the interface is small (3 methods). If you find yourself
needing a 4th method, ask whether the agent really needs it or whether
the provider should handle it internally. Small interfaces are stable
interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, Optional, Type, TypeVar

from pydantic import BaseModel

from atlas.models import TokenUsage

T = TypeVar("T", bound=BaseModel)


class ModelResponse(BaseModel, Generic[T]):
    """Wraps a structured output response with usage metadata."""
    data: T
    usage: TokenUsage
    latency_ms: float


class ModelProvider(ABC):
    """The only interface the agent layer is allowed to see.

    Implementations:
    - GroqProvider (Day 3): real LLM via Groq API
    - MockProvider (Day 1): deterministic canned responses for tests
    """

    @abstractmethod
    async def generate_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        """Generate plain text. Returns (text, usage).

        Used by the synthesizer (final answer is prose with citations).
        """
        ...

    @abstractmethod
    async def generate_structured(
        self,
        response_model: Type[T],
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> ModelResponse[T]:
        """Generate structured output matching response_model.

        Used by the planner (PlanStep) and critic (CriticReport).
        The provider is responsible for:
        1. Including the JSON schema in the prompt
        2. Requesting JSON mode from the API
        3. Parsing and validating the response
        4. Retrying on validation failure
        """
        ...

    @abstractmethod
    async def stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text token-by-token. Used by the SSE endpoint.

        Note: streaming does not return TokenUsage because usage is only
        known after the stream completes. The caller can estimate usage
        from the token count if needed.
        """
        ...


def make_provider(settings) -> ModelProvider:
    """Factory: pick the right provider based on settings.

    This is the only place in the codebase that knows about concrete
    providers. The agent layer calls this once at startup and gets
    a ModelProvider — it never knows which one.
    """
    # Lazy imports — only import what we need
    if settings.llm_provider == "mock":
        from atlas.providers.mock import MockProvider
        return MockProvider()
    elif settings.llm_provider == "groq":
        from atlas.providers.groq_provider import GroqProvider
        if not settings.groq_api_key:
            raise ValueError(
                "ATLAS_LLM_PROVIDER=groq but ATLAS_GROQ_API_KEY is empty. "
                "Set it in .env or switch to mock."
            )
        return GroqProvider(
            api_key=settings.groq_api_key,
            planner_model=settings.groq_planner_model,
            synthesizer_model=settings.groq_synthesizer_model,
            critic_model=settings.groq_critic_model,
        )
    else:
        raise ValueError(f"unknown llm_provider: {settings.llm_provider}")
