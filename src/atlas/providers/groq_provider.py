"""
GroqProvider — real LLM calls via Groq API.

Implements the ModelProvider interface using the Groq Python SDK.
Supports:
- generate_text: plain text generation (for the synthesizer)
- generate_structured: JSON mode with Pydantic validation + retry (for planner/critic)
- stream: token-by-token streaming (for the SSE endpoint)

Design decisions:
1. Uses the official `groq` Python SDK (async client). The SDK handles auth,
   connection pooling, and HTTP-level retries. We add application-level
   retry only for schema validation failures.

2. JSON mode for structured output: we include the Pydantic schema in the
   system prompt and validate the response. Markdown code-fence removal
   is applied defensively because some models wrap JSON in ```json blocks.

3. Retry on validation failure: if the LLM returns valid JSON that doesn't
   match our Pydantic schema, we retry once with a "your last response
   did not match the schema" preamble.

4. Separate models for different components (tuned for Groq free-tier
   token economy — TPD limit is 200,000/day):
   - Planner:     openai/gpt-oss-20b  (binary retrieve/answer decision)
   - Synthesizer: openai/gpt-oss-120b (writes the final answer)
   - Critic:      openai/gpt-oss-20b  (verify grounding)

5. Rate-limit handling: Groq 429 errors are parsed into RateLimitError
   with structured wait_seconds and limit_type. These propagate up
   so the eval runner can sleep the exact required time, instead of
   blind fixed waits. This is critical because TPD errors require
   minutes of waiting, not seconds.

Bug fix from Day 3 audit: the provider returns TokenUsage so the
agent loop can track token consumption and enforce the budget.
"""

from __future__ import annotations

import json
import logging
import re
from typing import AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from atlas.exceptions import ProviderError
from atlas.models import TokenUsage
from atlas.providers.base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RateLimitError(ProviderError):
    """Raised when Groq returns a 429 rate-limit error.

    Carries structured information so callers (e.g. the eval runner)
    can implement smart retry logic instead of blind fixed sleeps.

    Attributes:
        wait_seconds: Time to wait before retrying, parsed from Groq's
            "Please try again in X" hint. Always includes a +5s safety
            buffer so we never retry too early.
        limit_type: "TPD" (tokens per day), "RPM" (requests per minute),
            "RPD" (requests per day), or "unknown".
        raw_error: The original error string for debugging.
    """

    def __init__(
        self,
        message: str,
        wait_seconds: float = 60.0,
        limit_type: str = "unknown",
        raw_error: Optional[str] = None,
    ):
        super().__init__(message)
        self.wait_seconds = wait_seconds
        self.limit_type = limit_type
        self.raw_error = raw_error or message


def _parse_429_error(error: Exception) -> RateLimitError:
    """Parse a Groq 429 error into a structured RateLimitError.

    Groq error messages come in two formats:

        Format A (minutes + seconds):
            Rate limit reached for model `openai/gpt-oss-120b` ...
            on tokens per day (TPD): Limit 200000, Used 199396, ...
            Please try again in 4m18.336s.

        Format B (seconds only):
            Please try again in 58.000000 seconds.

    We extract:
        - limit_type: TPD / RPM / RPD / unknown
        - wait_seconds: parsed value + 5s safety buffer (min 10s)
    """
    err_str = str(error)

    # Detect limit type from the message
    if "tokens per day" in err_str or "(TPD)" in err_str:
        limit_type = "TPD"
    elif "requests per minute" in err_str or "(RPM)" in err_str:
        limit_type = "RPM"
    elif "requests per day" in err_str or "(RPD)" in err_str:
        limit_type = "RPD"
    else:
        limit_type = "unknown"

    # Parse wait time — try "Xm Y.YYYs" first, then "X.YYY seconds", then "X.YYYs"
    wait_seconds: Optional[float] = None

    match_min_sec = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
    if match_min_sec:
        minutes = int(match_min_sec.group(1))
        seconds = float(match_min_sec.group(2))
        wait_seconds = minutes * 60 + seconds
    else:
        match_sec_word = re.search(r"try again in ([\d.]+) seconds", err_str)
        if match_sec_word:
            wait_seconds = float(match_sec_word.group(1))
        else:
            match_sec_only = re.search(r"try again in ([\d.]+)s\b", err_str)
            if match_sec_only:
                wait_seconds = float(match_sec_only.group(1))

    # Apply safety buffer and floor
    if wait_seconds is None:
        wait_seconds = 60.0  # conservative default if parsing fails
    wait_seconds = max(wait_seconds + 5.0, 10.0)

    return RateLimitError(
        message=(
            f"Groq {limit_type} rate limit. "
            f"Wait {wait_seconds:.0f}s before retrying. "
            f"Original: {err_str[:200]}"
        ),
        wait_seconds=wait_seconds,
        limit_type=limit_type,
        raw_error=err_str,
    )


class GroqProvider(ModelProvider):
    """LLM provider using the Groq API.

    Default model selection (optimized for free-tier token economy):
        - Planner:     openai/gpt-oss-20b  (binary retrieve/answer decision)
        - Synthesizer: openai/gpt-oss-120b (writes the final answer)
        - Critic:      openai/gpt-oss-20b  (verify grounding)

    Switching planner from 120B → 20B cuts planner token cost ~40%, which
    matters because every case calls planner 1-3 times.
    """

    def __init__(
        self,
        api_key: str,
        planner_model: str = "openai/gpt-oss-20b",
        synthesizer_model: str = "openai/gpt-oss-120b",
        critic_model: str = "openai/gpt-oss-20b",
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """Initialize the Groq provider.

        Args:
            api_key: Groq API key (starts with gsk_)
            planner_model: Model for planner (default: 20B, sufficient for
                binary retrieve/answer decisions, saves ~40% tokens)
            synthesizer_model: Model for synthesizer (default: 120B for prose
                quality — do NOT downgrade this, it writes user-facing answers)
            critic_model: Model for critic (default: 20B, fast verification)
            timeout: Request timeout in seconds
            max_retries: Max retries on schema validation failure (NOT on
                rate limits — rate limits propagate immediately)
        """
        if not api_key:
            raise ValueError("Groq API key is required")

        self.api_key = api_key
        self.planner_model = planner_model
        self.synthesizer_model = synthesizer_model
        self.critic_model = critic_model
        self.timeout = timeout
        self.max_retries = max_retries

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
        default synthesizer model. The planner and critic pass their
        specific models explicitly via the model= kwarg.
        """
        return model or self.synthesizer_model

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """Detect whether an exception is a Groq 429 rate-limit error.

        We check multiple markers because Groq raises different exception
        types depending on whether the SDK retries internally or surfaces
        the 429 directly.
        """
        err_str = str(error).lower()
        return (
            "rate_limit_exceeded" in err_str
            or "429" in err_str
            or "too many requests" in err_str
            or ("rate limit" in err_str and "try again" in err_str)
        )

    async def generate_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        """Generate plain text. Used by the synthesizer.

        Returns (text, usage) where usage is the token consumption.

        Raises:
            RateLimitError: On 429 — caller must wait wait_seconds then retry.
            ProviderError: On other API failures.
        """
        resolved_model = self._resolve_model(model)

        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            text = response.choices[0].message.content or ""
            usage = self._extract_usage(response, resolved_model)

            logger.info(
                f"groq.generate_text: model={resolved_model}, "
                f"tokens={usage.total_tokens}, "
                f"output_chars={len(text)}"
            )

            return text, usage

        except Exception as e:
            # Rate-limit errors propagate IMMEDIATELY as RateLimitError.
            # Do NOT swallow them — the caller needs the wait_seconds.
            if self._is_rate_limit_error(e):
                raise _parse_429_error(e) from e

            if "timeout" in str(e).lower():
                logger.error(f"groq.generate_text timed out: {e}")
                raise ProviderError("Groq request timed out") from e

            logger.error(f"groq.generate_text failed: {e}")
            raise ProviderError(
                f"Groq API error: {type(e).__name__}: {str(e)[:200]}"
            ) from e

    async def generate_structured(
        self,
        response_model: Type[T],
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> ModelResponse[T]:
        """Generate structured output matching a Pydantic model.

        Uses Groq's JSON mode + Pydantic validation. If the LLM returns
        valid JSON that doesn't match the schema, retries with an error
        preamble. Rate-limit errors propagate immediately.

        Args:
            response_model: Pydantic class to validate against.
            system: System prompt.
            user: User prompt.
            model: Model override (otherwise uses synthesizer default).

        Returns:
            ModelResponse with validated data and token usage.

        Raises:
            RateLimitError: On 429 — caller must wait wait_seconds then retry.
            ProviderError: On other API failures or after max_retries.
        """
        resolved_model = self._resolve_model(model)

        # Build the system prompt with schema hint
        schema = response_model.model_json_schema()
        system_with_schema = (
            f"{system}\n\n"
            f"You must respond with ONLY a valid JSON object matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"CRITICAL: Return ONLY the raw JSON object. "
            f"Do NOT wrap it in markdown code blocks. "
            f"Do NOT add any text before or after the JSON. "
            f"Start your response with {{ and end it with }}."
        )

        last_error: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            # On retry, prepend the validation error so the LLM can self-correct
            current_user = user
            if attempt > 0 and last_error:
                current_user = (
                    f"Your previous response was not valid JSON or did not match the schema.\n"
                    f"Error: {last_error[:300]}\n\n"
                    f"Please try again. Return ONLY a raw JSON object starting with {{ and ending with }}.\n"
                    f"Do NOT use markdown code blocks.\n\n"
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
                    max_tokens=2000,
                )

                raw_text = response.choices[0].message.content or "{}"
                usage = self._extract_usage(response, resolved_model)

                # Clean the response: strip markdown code fences if present
                cleaned_text = raw_text.strip()

                if cleaned_text.startswith("```"):
                    lines = cleaned_text.split("\n")
                    if len(lines) > 2:
                        # Drop the opening ```json and closing ``` lines
                        cleaned_text = "\n".join(lines[1:-1])
                    elif len(lines) == 2:
                        cleaned_text = lines[1]

                # Extract the outermost {...} block (defensive — some models
                # add stray text before/after the JSON)
                first_brace = cleaned_text.find("{")
                last_brace = cleaned_text.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    cleaned_text = cleaned_text[first_brace:last_brace + 1]

                data = response_model.model_validate_json(cleaned_text)

                logger.info(
                    f"groq.generate_structured: model={resolved_model}, "
                    f"attempt={attempt + 1}, tokens={usage.total_tokens}"
                )

                return ModelResponse(data=data, usage=usage, latency_ms=0.0)

            except ValidationError as e:
                # Schema validation failed — retry with error preamble
                last_error = str(e)[:500]
                logger.warning(
                    f"groq.generate_structured: validation failed on attempt "
                    f"{attempt + 1}: {last_error[:100]}"
                )
                continue

            except Exception as e:
                error_str = str(e)

                # CRITICAL: Rate-limit errors MUST propagate immediately.
                # Do NOT swallow them in the retry loop — the caller needs
                # wait_seconds to implement smart retry.
                if self._is_rate_limit_error(e):
                    raise _parse_429_error(e) from e

                # Groq's JSON validation failure — retry without response_format
                if "json_validate_failed" in error_str or "Failed to generate JSON" in error_str:
                    last_error = error_str[:500]
                    logger.warning(
                        f"Groq JSON validation failed, retrying (attempt {attempt + 1})"
                    )
                    continue

                if "timeout" in error_str.lower():
                    raise ProviderError("Groq request timed out") from e

                logger.error(f"groq.generate_structured failed: {error_str[:200]}")
                raise ProviderError(
                    f"Groq API error: {type(e).__name__}: {error_str[:200]}"
                ) from e

        # All schema-validation retries exhausted
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
        """Stream text token-by-token. Used by the SSE endpoint.

        Note: streaming does not return TokenUsage because the Groq SDK
        doesn't report usage until the stream completes.

        Raises:
            RateLimitError: On 429 — caller must wait wait_seconds then retry.
            ProviderError: On other API failures.
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
            if self._is_rate_limit_error(e):
                raise _parse_429_error(e) from e

            logger.error(f"groq.stream failed: {e}")
            raise ProviderError(
                f"Groq stream error: {type(e).__name__}: {str(e)[:200]}"
            ) from e

    @staticmethod
    def _extract_usage(response, model_name: str) -> TokenUsage:
        """Extract token usage from a Groq response.

        Groq returns usage in OpenAI format:
            response.usage.prompt_tokens     (input)
            response.usage.completion_tokens (output)
            response.usage.total_tokens      (sum)
        """
        usage = response.usage
        if usage is None:
            return TokenUsage(input_tokens=0, output_tokens=0, model_name=model_name)

        return TokenUsage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
            model_name=model_name,
        )