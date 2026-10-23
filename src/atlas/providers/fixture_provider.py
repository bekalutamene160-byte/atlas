"""
FixtureProvider — records and replays LLM responses for deterministic eval.

Implements the same ModelProvider interface as GroqProvider. In RECORD mode,
delegates to a real GroqProvider and saves responses to JSON files. In REPLAY
mode, loads responses from disk and never calls the real API.

Why this matters:
- Eval runs become 100% free and deterministic after the first recording
- Same query → same response every time (no flaky test failures)
- Protects your Groq TPD quota (200k/day on free tier)
- Enables offline development

Usage patterns:

    # RECORD: first run, captures real responses
    provider = FixtureProvider(real_provider=groq, mode="record")

    # REPLAY: every run after, zero API cost
    provider = FixtureProvider(real_provider=groq, mode="replay")

    # AUTO: replay if fixture exists, otherwise record (best for development)
    provider = FixtureProvider(real_provider=groq, mode="auto")

Fixture storage:
    tests/fixtures/llm_responses/{hash}.json
    Each file contains:
        - system_prompt: str
        - user_prompt: str
        - model: str
        - call_type: "text" | "structured"
        - text: str (the response text)
        - usage: dict (token usage)
        - recorded_at: ISO timestamp

Hash key:
    sha256(model + call_type + system_prompt + user_prompt)
    This means: if you change ANY of these, you get a new fixture.
    This is correct behavior — changing the prompt should re-record.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional, Type, TypeVar

from pydantic import BaseModel

from atlas.exceptions import ProviderError
from atlas.models import TokenUsage
from atlas.providers.base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# Default location for fixture files — relative to project root
DEFAULT_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "llm_responses"


class FixtureProvider(ModelProvider):
    """Provider that records and replays LLM responses.

    Drop-in replacement for GroqProvider. Same interface, same return types.
    """

    def __init__(
        self,
        real_provider: Optional[ModelProvider] = None,
        mode: str = "replay",
        fixtures_dir: Optional[Path] = None,
    ):
        """Initialize the fixture provider.

        Args:
            real_provider: A real provider (e.g. GroqProvider) used in
                "record" and "auto" modes. Not needed in pure "replay" mode.
            mode: One of:
                - "record": always call real_provider, save responses
                - "replay": always load from disk, fail if missing
                - "auto": try replay first, record if missing (recommended
                  for development)
            fixtures_dir: Directory to store fixture JSON files. Defaults
                to tests/fixtures/llm_responses under project root.
        """
        if mode not in ("record", "replay", "auto"):
            raise ValueError(f"Invalid mode: {mode}. Must be record/replay/auto.")

        if mode in ("record", "auto") and real_provider is None:
            raise ValueError(
                f"real_provider is required when mode is '{mode}'"
            )

        self.real_provider = real_provider
        self.mode = mode
        self.fixtures_dir = fixtures_dir or DEFAULT_FIXTURES_DIR
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

        # Stats for logging
        self._replay_count = 0
        self._record_count = 0

        logger.info(
            f"FixtureProvider initialized: mode={mode}, "
            f"fixtures_dir={self.fixtures_dir}"
        )

    # ------------------------------------------------------------------
    # Hashing and file paths
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        system: str,
        user: str,
        model: str,
        call_type: str,
    ) -> str:
        """Generate a stable hash key for this call.

        The key includes everything that affects the response:
        - model name
        - call type (text vs structured)
        - system prompt (includes schema for structured calls)
        - user prompt (includes the query)

        If any of these change, we get a new key, which means we record
        a new fixture. This is correct — changing the prompt invalidates
        old fixtures.
        """
        key_str = f"{model}|{call_type}|{system}|{user}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:32]

    def _fixture_path(self, key: str) -> Path:
        return self.fixtures_dir / f"{key}.json"

    def _load_fixture(self, path: Path) -> dict:
        """Load a fixture from disk. Raises ProviderError if corrupt."""
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ProviderError(
                f"Corrupt fixture file {path.name}: {e}. "
                f"Delete it and re-record."
            ) from e

    def _save_fixture(self, path: Path, data: dict) -> None:
        """Save a fixture to disk atomically (write to .tmp, then rename)."""
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)  # atomic on most filesystems

    # ------------------------------------------------------------------
    # ModelProvider interface
    # ------------------------------------------------------------------

    async def generate_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> tuple[str, TokenUsage]:
        """Generate text, replaying from fixture if available."""
        resolved_model = model or getattr(
            self.real_provider, "synthesizer_model", "unknown"
        )
        key = self._make_key(system, user, resolved_model, "text")
        path = self._fixture_path(key)

        # Try replay first (in replay or auto mode)
        if self.mode in ("replay", "auto") and path.exists():
            fixture = self._load_fixture(path)
            self._replay_count += 1
            logger.debug(f"Fixture replay (text): {path.name}")

            text = fixture["text"]
            usage = TokenUsage(
                input_tokens=fixture["usage"]["input_tokens"],
                output_tokens=fixture["usage"]["output_tokens"],
                model_name=fixture["model"],
            )
            return text, usage

        # If pure replay mode and no fixture, fail loudly
        if self.mode == "replay":
            raise ProviderError(
                f"No fixture found for text call (hash={key}). "
                f"Switch to 'record' or 'auto' mode to capture it. "
                f"Expected file: {path}"
            )

        # Record mode (or auto mode with no fixture) — call real provider
        assert self.real_provider is not None
        text, usage = await self.real_provider.generate_text(system, user, model)

        # Save the fixture
        fixture = {
            "system_prompt": system,
            "user_prompt": user,
            "model": resolved_model,
            "call_type": "text",
            "text": text,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_fixture(path, fixture)
        self._record_count += 1
        logger.info(f"Fixture recorded (text): {path.name}")

        return text, usage

    async def generate_structured(
        self,
        response_model: Type[T],
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> ModelResponse[T]:
        """Generate structured output, replaying from fixture if available."""
        resolved_model = model or getattr(
            self.real_provider, "synthesizer_model", "unknown"
        )
        key = self._make_key(system, user, resolved_model, "structured")
        path = self._fixture_path(key)

        # Try replay first
        if self.mode in ("replay", "auto") and path.exists():
            fixture = self._load_fixture(path)
            self._replay_count += 1
            logger.debug(f"Fixture replay (structured): {path.name}")

            # Re-validate the cached JSON against the Pydantic model
            data = response_model.model_validate(fixture["data"])
            usage = TokenUsage(
                input_tokens=fixture["usage"]["input_tokens"],
                output_tokens=fixture["usage"]["output_tokens"],
                model_name=fixture["model"],
            )
            return ModelResponse(data=data, usage=usage, latency_ms=0.0)

        # Pure replay mode without fixture — fail
        if self.mode == "replay":
            raise ProviderError(
                f"No fixture found for structured call (hash={key}). "
                f"Switch to 'record' or 'auto' mode to capture it. "
                f"Expected file: {path}"
            )

        # Record mode — call real provider
        assert self.real_provider is not None
        response = await self.real_provider.generate_structured(
            response_model, system, user, model
        )

        # Save the fixture — store the validated dict form so we can
        # reconstruct any Pydantic model on replay
        fixture = {
            "system_prompt": system,
            "user_prompt": user,
            "model": resolved_model,
            "call_type": "structured",
            "data": response.data.model_dump(),
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_fixture(path, fixture)
        self._record_count += 1
        logger.info(f"Fixture recorded (structured): {path.name}")

        return response

    async def stream(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens. Fixtures don't support streaming — falls through
        to the real provider.

        If you need to fixture streaming, the simplest approach is to
        record the full text and yield it in chunks. But for eval purposes,
        streaming is rarely needed.
        """
        if self.real_provider is None:
            raise ProviderError(
                "FixtureProvider.stream requires a real_provider. "
                "Streaming cannot be replayed from fixtures."
            )

        async for chunk in self.real_provider.stream(system, user, model):
            yield chunk

    def stats(self) -> dict:
        """Return replay/record counts for this session. Useful for logging."""
        return {
            "replay_count": self._replay_count,
            "record_count": self._record_count,
            "mode": self.mode,
            "fixtures_dir": str(self.fixtures_dir),
        }