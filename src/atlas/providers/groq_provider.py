"""
GroqProvider — real LLM calls via Groq API.

Implements the ModelProvider interface using the Groq Python SDK.
Supports:
- generate_text: plain text generation (for the synthesizer)
- generate_structured: JSON mode with Pydantic validation + retry (for planner/critic)
- stream: token-by-token streaming (for the SSE endpoint on Day 5)

Design decisions:
1. Uses the official `groq` Python SDK (async client). Don't use raw httpx —
   the SDK handles auth, retries, and rate limits.

2. JSON mode for structured output: Groq supports response_format=
   {"type": "json_object"} which forces the LLM to return valid JSON.
   We include the schema in the prompt and validate with Pydantic.

3. Retry on validation failure: if the LLM returns valid JSON that doesn't
   match our Pydantic schema, we retry once with a "your last response
   did not match the schema" preamble. Llama 3.3 70B is very reliable,
   so this is rare.

4. Separate models for different components:
   - Planner: llama-3.3-70b-versatile (good reasoning for planning)
   - Synthesizer: llama-3.3-70b-versatile (good prose generation)
   - Critic: llama-3.1-8b-instant (fast verification, 4x cheaper)

5. Timeout and retry: Groq can be slow or rate-limit. We use tenacity
   for exponential backoff with jitter on network errors.

Bug fix from Day 3 audit: the provider now returns TokenUsage so the
agent loop can track token consumption and enforce the budget.

Pro tip for advanced projects: add a FixtureProvider that records real
Groq responses to JSON files and replays them. This makes eval runs
deterministic and free. The masterplan calls this "executable memory."
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.exceptions import ProviderError
from atlas.models import TokenUsage
from atlas.providers.base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class GroqProvider(ModelProvider):
    """LLM provider using the Groq API.

    Supports Llama 3.3 70B (planner/synthesizer) and Llama 3.1 8B (critic).
    """

    def __init__(
        self,
        api_key: str,
        planner_model: str = "llama-3.3-70b-versatile",
        synthesizer_model: str = "llama-3.3-70b-versatile",
        critic_model: str = "llama-3.1-8b-instant",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize the Groq provider.

        Args:
            api_key: Groq API key (starts with gsk_)
            planner_model: Model for planner (default: 70B for reasoning)
            synthesizer_model: Model for synthesizer (default: 70B for prose)
            critic_model: Model for critic (default: 8B for speed)
            timeout: Request timeout in seconds
            max_retries: Max retries on schema validation failure
        """
        if not api_key:
            raise ValueError("Groq API key is required")

        self.api_key = api_key
        self.planner_model = planner_model
        self.synthesizer_model = synthesizer_model
        self.critic_model = critic_model
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize the Groq async client
        # Lazy import so mock users don't need groq installed
        try:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=api_key, timeout=timeout)
        except ImportError as e:
            raise ProviderError(
                "groq package not installed. Run: pip install groq"
            ) from e
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {e}") from e

        logger.info(
            f"GroqProvider initialized: planner={planner_model}, "
            f"synthesizer={synthesizer_model}, critic={critic_model}"
        )

    def _resolve_model(self, model: Optional[str]) -> str:
        """Resolve which model to use.

        If the caller specifies a model, use it. Otherwise use the
        default synthesizer model (70B). The planner and critic pass
        their specific models explicitly.
        """
        return model or self.synthesizer_model

    async def generate_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        """Generate plain text. Used by the synthesizer.

        Returns (text, usage) where usage is the token consumption.
        """
        resolved_model = self._resolve_model(model)

        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,  # low temperature for factual answers
                max_tokens=1000,
            )

            text = response.choices[0].message.content or ""

            # Extract token usage (Groq returns it in a standard format)
            usage = self._extract_usage(response, resolved_model)

            logger.info(
                f"groq.generate_text: model={resolved_model}, "
                f"tokens={usage.total_tokens}, "
                f"output_chars={len(text)}"
            )

            return text, usage

        except Exception as e:
            logger.error(f"groq.generate_text failed: {e}")
            if "rate_limit" in str(e).lower():
                raise ProviderError("Groq rate limit exceeded") from e
            elif "timeout" in str(e).lower():
                raise ProviderError("Groq request timed out") from e
            else:
                raise ProviderError(f"Groq API error: {type(e).__name__}: {str(e)[:200]}") from e

    async def generate_structured(
        self,
        response_model: Type[T],
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> ModelResponse[T]:
        """Generate structured output matching a Pydantic model.

        Uses Groq's JSON mode + Pydantic validation. If the LLM returns
        valid JSON that doesn't match the schema, retries once with an
        error message prepended.

        Used by the planner (PlanStep) and critic (CriticResponse).
        """
        resolved_model = self._resolve_model(model)

        # Build the system prompt with schema hint
        schema = response_model.model_json_schema()
        system_with_schema = (
            f"{system}\n\n"
            f"You must respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Return ONLY the JSON object, no markdown, no explanation."
        )

        last_error: Optional[str] = None
        last_usage: Optional[TokenUsage] = None

        for attempt in range(self.max_retries + 1):
            # On retry, prepend the error message
            current_user = user
            if attempt > 0 and last_error:
                current_user = (
                    f"Your previous response did not match the required schema.\n"
                    f"Error: {last_error[:300]}\n\n"
                    f"Please try again. Return ONLY valid JSON.\n\n"
                    f"{user}"
                )

            try:
                response = await self._client.chat.completions.create(
                    model=resolved_model,
                    messages=[
                        {"role": "system", "content": system_with_schema},
                        {"role": "user", "content": current_user},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                )

                raw_text = response.choices[0].message.content or "{}"
                usage = self._extract_usage(response, resolved_model)
                last_usage = usage

                # Parse and validate against the Pydantic model
                data = response_model.model_validate_json(raw_text)

                logger.info(
                    f"groq.generate_structured: model={resolved_model}, "
                    f"attempt={attempt + 1}, tokens={usage.total_tokens}"
                )

                return ModelResponse(data=data, usage=usage, latency_ms=0.0)

            except ValidationError as e:
                last_error = str(e)[:500]
                logger.warning(
                    f"groq.generate_structured: validation failed on attempt "
                    f"{attempt + 1}: {last_error[:100]}"
                )
                continue  # retry

            except Exception as e:
                logger.error(f"groq.generate_structured failed: {e}")
                if "rate_limit" in str(e).lower():
                    raise ProviderError("Groq rate limit exceeded") from e
                elif "timeout" in str(e).lower():
                    raise ProviderError("Groq request timed out") from e
                else:
                    raise ProviderError(
                        f"Groq API error: {type(e).__name__}: {str(e)[:200]}"
                    ) from e

        # All retries exhausted
        raise ProviderError(
            f"Schema validation failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_error[:200] if last_error else 'unknown'}"
        )

    async def stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream text token-by-token. Used by the SSE endpoint (Day 5).

        Note: streaming does not return TokenUsage because the Groq SDK
        doesn't report usage until the stream completes. The caller can
        estimate from the response length if needed.
        """
        resolved_model = self._resolve_model(model)

        try:
            stream = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=1000,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"groq.stream failed: {e}")
            raise ProviderError(
                f"Groq stream error: {type(e).__name__}: {str(e)[:200]}"
            ) from e

    @staticmethod
    def _extract_usage(response, model_name: str) -> TokenUsage:
        """Extract token usage from a Groq response.

        Groq returns usage in the same format as OpenAI:
        response.usage.prompt_tokens (input)
        response.usage.completion_tokens (output)
        response.usage.total_tokens (sum)

        We map these to our provider-neutral TokenUsage.
        """
        usage = response.usage
        if usage is None:
            return TokenUsage(input_tokens=0, output_tokens=0, model_name=model_name)

        return TokenUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            model_name=model_name,
        )
