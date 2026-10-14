"""
Shared pytest fixtures.

Fixtures defined here are available to all tests without import.
This is pytest's convention: conftest.py is auto-discovered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.config import Settings
from atlas.providers.mock import MockProvider


@pytest.fixture
def mock_settings() -> Settings:
    """Settings configured for mock mode (no internet needed)."""
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        max_agent_steps=3,
        run_timeout_seconds=5.0,
        max_tokens_per_run=1000,
    )


@pytest.fixture
def mock_provider() -> MockProvider:
    """A fresh MockProvider with default responses."""
    return MockProvider()


@pytest.fixture
def mock_provider_with_script() -> MockProvider:
    """A MockProvider scripted for a 3-step retrieve-synthesize-refuse run."""
    return MockProvider(
        structured_responses=[
            {"action": "retrieve", "query": "Atlas-3 pump specifications",
             "reasoning": "Need to find the spec sheet for the Atlas-3 pump."},
            {"action": "synthesize", "reasoning": "Have enough evidence to answer."},
            {"verdicts": [
                {"claim_text": "The pump operates at 5 MPa.",
                 "supported": True, "supporting_evidence_id": "e1"}
            ], "all_supported": True, "issues": []},
        ],
        text_responses=[
            "The Atlas-3 pump has a maximum operating pressure of 5 MPa [e1].",
        ],
    )


@pytest.fixture
def eval_path() -> Path:
    """Path to the eval benchmark file."""
    return Path(__file__).parent.parent.parent / "eval" / "benchmark.jsonl"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """A temporary data directory for tests that touch the filesystem."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "corpus").mkdir()
    (data_dir / "embeddings_cache").mkdir()
    return data_dir
