"""Unit tests for Settings configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas.config import Settings


class TestSettings:
    """Tests for the Settings class."""

    def test_defaults_are_mock_mode(self):
        """Default settings use mock LLM provider (no internet needed)."""
        # Use _env_file=None so the test doesn't read the real .env
        # (which might have ATLAS_LLM_PROVIDER=groq for Day 4 testing)
        s = Settings(_env_file=None)
        assert s.llm_provider == "mock"
        assert s.max_agent_steps == 8
        assert s.run_timeout_seconds == 60.0

    def test_env_prefix_atlas(self, monkeypatch):
        """Settings load from ATLAS_-prefixed env vars."""
        monkeypatch.setenv("ATLAS_MAX_AGENT_STEPS", "5")
        s = Settings()
        assert s.max_agent_steps == 5

    def test_invalid_literal_raises(self):
        """Invalid literal value raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(llm_provider="invalid_provider")

    def test_bounds_validation_rejects_zero_steps(self):
        """max_agent_steps must be >= 1."""
        with pytest.raises(ValidationError):
            Settings(max_agent_steps=0)

    def test_bounds_validation_rejects_excessive_steps(self):
        """max_agent_steps must be <= 50."""
        with pytest.raises(ValidationError):
            Settings(max_agent_steps=100)

    def test_is_mock_mode_true_when_both_mock(self):
        """is_mock_mode() returns True when both providers are mock."""
        s = Settings(llm_provider="mock", embedding_provider="mock")
        assert s.is_mock_mode() is True

    def test_is_mock_mode_false_when_real(self):
        """is_mock_mode() returns False when llm_provider is real."""
        s = Settings(llm_provider="groq", embedding_provider="mock")
        assert s.is_mock_mode() is False

    def test_chunk_overlap_must_be_non_negative(self):
        """chunk_overlap must be >= 0."""
        with pytest.raises(ValidationError):
            Settings(chunk_overlap=-1)

    def test_min_similarity_bounds(self):
        """min_similarity must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            Settings(min_similarity=1.5)
        with pytest.raises(ValidationError):
            Settings(min_similarity=-0.1)
