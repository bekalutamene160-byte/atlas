"""Unit tests for RetrieverTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.ingestion.embedder import Embedder, MockEmbedder
from atlas.models import Chunk
from atlas.retrieval.vector_store import VectorStore
from atlas.tools.retriever_tool import RetrieveArgs, RetrieverTool


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """A vector store with 3 chunks for testing."""
    store = VectorStore(db_path=str(tmp_path / "test.db"), embedding_dims=4)
    chunks = [
        Chunk(chunk_id="c1", source="a.pdf", page=1, text="hello world", chunk_index=0),
        Chunk(chunk_id="c2", source="a.pdf", page=2, text="foo bar", chunk_index=1),
        Chunk(chunk_id="c3", source="b.pdf", page=1, text="test data", chunk_index=0),
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    store.add(chunks, vectors, "test-model")
    return store


@pytest.fixture
def embedder() -> MockEmbedder:
    """A mock embedder that returns 4-dim vectors."""
    return MockEmbedder(dims=4)


@pytest.fixture
def retriever(embedder: MockEmbedder, vector_store: VectorStore) -> RetrieverTool:
    """A retriever tool with mock embedder and test vector store."""
    return RetrieverTool(
        embedder=embedder,
        vector_store=vector_store,
        min_similarity=0.0,
    )


class TestRetrieverTool:
    """Tests for the retriever tool."""

    def test_name_and_description(self, retriever: RetrieverTool):
        """Tool has correct name and description."""
        assert retriever.name == "retrieve"
        assert len(retriever.description) > 10

    def test_args_model_is_retrieve_args(self, retriever: RetrieverTool):
        """args_model is RetrieveArgs."""
        assert retriever.args_model == RetrieveArgs

    def test_get_schema(self, retriever: RetrieverTool):
        """get_schema returns name, description, and args schema."""
        schema = retriever.get_schema()
        assert schema["name"] == "retrieve"
        assert "description" in schema
        assert "args" in schema
        assert "query" in schema["args"]["properties"]

    async def test_execute_returns_chunks(self, retriever: RetrieverTool):
        """Execute returns chunks from the vector store."""
        args = RetrieveArgs(query="hello")
        result = await retriever.execute(args)

        assert result.success is True
        assert result.data["count"] > 0
        assert len(result.data["chunks"]) > 0
        assert result.latency_ms > 0

    async def test_execute_no_results(self, retriever: RetrieverTool):
        """Execute returns empty list when no chunks match."""
        # Use a high min_similarity that nothing will match
        retriever.min_similarity = 0.99
        args = RetrieveArgs(query="nothing matches this")
        result = await retriever.execute(args)

        assert result.success is True  # success=True, but empty
        assert result.data["count"] == 0
        assert result.data["chunks"] == []

    async def test_execute_handles_embedding_error(self, vector_store: VectorStore):
        """Execute returns error if embedder fails."""
        # Create a broken embedder
        class BrokenEmbedder(Embedder):
            @property
            def dimensions(self) -> int:
                return 4

            @property
            def model_name(self) -> str:
                return "broken"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                return []  # returns empty

        tool = RetrieverTool(
            embedder=BrokenEmbedder(),
            vector_store=vector_store,
            min_similarity=0.0,
        )
        args = RetrieveArgs(query="test")
        result = await tool.execute(args)

        assert result.success is False
        assert result.error is not None
        assert "empty" in result.error.lower()

    async def test_execute_never_raises(self, vector_store: VectorStore):
        """Execute catches all exceptions, never raises."""
        class ExplodingEmbedder(Embedder):
            @property
            def dimensions(self) -> int:
                return 4

            @property
            def model_name(self) -> str:
                return "exploding"

            async def embed(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("kaboom!")

        tool = RetrieverTool(
            embedder=ExplodingEmbedder(),
            vector_store=vector_store,
            min_similarity=0.0,
        )
        args = RetrieveArgs(query="test")
        result = await tool.execute(args)

        # Should return error, not raise
        assert result.success is False
        assert result.error is not None
        assert "retrieval_error" in result.error


class TestRetrieveArgs:
    """Tests for the RetrieveArgs schema."""

    def test_valid_args(self):
        """Valid args with query and default top_k."""
        args = RetrieveArgs(query="pump pressure")
        assert args.query == "pump pressure"
        assert args.top_k == 5  # default

    def test_custom_top_k(self):
        """top_k can be overridden."""
        args = RetrieveArgs(query="test", top_k=10)
        assert args.top_k == 10

    @pytest.mark.parametrize("query", ["", "   "])
    def test_rejects_blank_query(self, query: str):
        with pytest.raises(ValueError):
            RetrieveArgs(query=query)

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_rejects_nonpositive_top_k(self, top_k: int):
        with pytest.raises(ValueError):
            RetrieveArgs(query="test", top_k=top_k)
