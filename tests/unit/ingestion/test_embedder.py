"""Unit tests for the embedder."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.ingestion.embedder import (
    CachedEmbedder,
    Embedder,
    MockEmbedder,
    make_embedder,
)
from atlas.config import Settings


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    async def test_returns_vectors(self):
        embedder = MockEmbedder(dims=384)
        result = await embedder.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 384
        assert all(isinstance(v, float) for v in result[0])

    async def test_same_text_same_vector(self):
        """Deterministic: same text → same vector."""
        embedder = MockEmbedder(dims=384)
        v1 = await embedder.embed(["test"])
        v2 = await embedder.embed(["test"])
        assert v1 == v2

    async def test_different_text_different_vector(self):
        """Different texts → different vectors."""
        embedder = MockEmbedder(dims=384)
        v1 = await embedder.embed(["hello"])
        v2 = await embedder.embed(["world"])
        assert v1 != v2

    async def test_empty_input(self):
        """Empty input returns empty list."""
        embedder = MockEmbedder(dims=384)
        result = await embedder.embed([])
        assert result == []

    def test_dimensions_property(self):
        embedder = MockEmbedder(dims=128)
        assert embedder.dimensions == 128

    def test_model_name_property(self):
        embedder = MockEmbedder()
        assert embedder.model_name == "mock-embedder"


class TestCachedEmbedder:
    """Tests for the caching wrapper."""

    async def test_cache_hit_skips_inner_call(self, tmp_path: Path):
        """Second embed of same text doesn't call the inner embedder."""
        call_count = 0

        class CountingEmbedder(Embedder):
            @property
            def dimensions(self):
                return 4

            @property
            def model_name(self):
                return "counting"

            async def embed(self, texts):
                nonlocal call_count
                call_count += len(texts)
                return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

        inner = CountingEmbedder()
        cached = CachedEmbedder(inner, tmp_path / "cache")

        # First call — cache miss
        await cached.embed(["hello"])
        assert call_count == 1

        # Second call — cache hit
        await cached.embed(["hello"])
        assert call_count == 1  # didn't increase

    async def test_different_texts_both_cached(self, tmp_path: Path):
        """Different texts are both cached separately."""
        inner = MockEmbedder(dims=4)
        cached = CachedEmbedder(inner, tmp_path / "cache")

        v1 = await cached.embed(["hello"])
        v2 = await cached.embed(["world"])

        assert v1 != v2

        # Both should be in cache now
        v1_again = await cached.embed(["hello"])
        assert v1_again == v1

    async def test_persistence_across_instances(self, tmp_path: Path):
        """Cache persists across CachedEmbedder instances."""
        inner = MockEmbedder(dims=4)
        cached1 = CachedEmbedder(inner, tmp_path / "cache")

        await cached1.embed(["persisted text"])

        # New instance, same cache dir
        cached2 = CachedEmbedder(inner, tmp_path / "cache")
        v2 = await cached2.embed(["persisted text"])

        assert len(v2) == 1
        assert len(v2[0]) == 4

    async def test_model_name_in_cache_key(self, tmp_path: Path):
        """Different model names produce different cache entries."""
        class NamedEmbedder(Embedder):
            def __init__(self, name):
                self._name = name

            @property
            def dimensions(self):
                return 4

            @property
            def model_name(self):
                return self._name

            async def embed(self, texts):
                return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

        cached1 = CachedEmbedder(NamedEmbedder("model-a"), tmp_path / "cache")
        cached2 = CachedEmbedder(NamedEmbedder("model-b"), tmp_path / "cache")

        await cached1.embed(["same text"])
        await cached2.embed(["same text"])

        # Both should have stored separate cache files
        cache_files = list((tmp_path / "cache").glob("*.json"))
        assert len(cache_files) >= 2


class TestMakeEmbedder:
    """Tests for the factory function."""

    def test_mock_provider(self):
        settings = Settings(embedding_provider="mock")
        embedder = make_embedder(settings)
        assert isinstance(embedder, MockEmbedder)

    def test_local_provider(self):
        settings = Settings(embedding_provider="local")
        # Should return a CachedEmbedder wrapping a local embedder
        embedder = make_embedder(settings)
        assert isinstance(embedder, CachedEmbedder)
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_hf_without_token_falls_back_to_local(self):
        """HF provider without token falls back to local."""
        settings = Settings(embedding_provider="hf", hf_api_token="")
        embedder = make_embedder(settings)
        assert isinstance(embedder, CachedEmbedder)
        # Should be using local model
        assert "MiniLM" in embedder.model_name or "all-MiniLM" in embedder.model_name
