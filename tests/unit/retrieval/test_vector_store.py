"""Unit tests for VectorStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.models import Chunk, RetrievedChunk
from atlas.retrieval.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """A fresh VectorStore using a temp database."""
    return VectorStore(
        db_path=str(tmp_path / "test.db"),
        embedding_dims=4,  # small for tests
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three sample chunks for testing."""
    return [
        Chunk(chunk_id="c1", source="a.pdf", page=1, text="hello world", chunk_index=0),
        Chunk(chunk_id="c2", source="a.pdf", page=2, text="foo bar baz", chunk_index=1),
        Chunk(chunk_id="c3", source="b.pdf", page=1, text="machine learning", chunk_index=0),
    ]


@pytest.fixture
def sample_vectors() -> list[list[float]]:
    """Vectors matching sample_chunks. c1 and c2 are similar, c3 is different."""
    return [
        [1.0, 0.0, 0.0, 0.0],  # c1
        [0.9, 0.1, 0.0, 0.0],  # c2 — similar to c1
        [0.0, 0.0, 0.0, 1.0],  # c3 — orthogonal to c1
    ]


class TestVectorStoreAdd:
    """Tests for adding chunks."""

    def test_add_chunks(self, vector_store, sample_chunks, sample_vectors):
        """Chunks are added to the store."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        assert vector_store.size == 3

    def test_add_deduplicates(self, vector_store, sample_chunks, sample_vectors):
        """Adding the same chunk_id twice doesn't duplicate."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        assert vector_store.size == 3  # still 3, not 6

    def test_add_empty_lists(self, vector_store):
        """Adding empty lists is a no-op."""
        vector_store.add([], [], "test-model")
        assert vector_store.size == 0

    def test_add_mismatched_lengths_raises(self, vector_store, sample_chunks):
        """Mismatched chunks and vectors lengths raise ValueError."""
        with pytest.raises(ValueError):
            vector_store.add(sample_chunks, [[1.0, 0.0, 0.0, 0.0]], "test-model")

    def test_add_duplicate_ids_within_batch(self, vector_store):
        chunk = Chunk(chunk_id="same", source="a.pdf", text="first", chunk_index=0)

        vector_store.add(
            [chunk, chunk],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "test-model",
        )

        assert vector_store.size == 1
        assert vector_store.vectors.shape == (1, 4)
        assert vector_store._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    def test_add_wrong_dimension_is_atomic(self, vector_store, sample_chunks):
        """Adding a vector with wrong dimensions fails and nothing is written."""
        # First add a valid chunk
        vector_store.add([sample_chunks[0]], [[1.0, 0.0, 0.0, 0.0]], "test-model")
        assert vector_store.size == 1
        
        # Now try to add a chunk with wrong dimensions (3 instead of 4)
        with pytest.raises(ValueError, match="dimensions"):
            vector_store.add([sample_chunks[1]], [[1.0, 0.0, 0.0]], "test-model")
        
        # The valid chunk should still be there (atomic - bad add didn't corrupt)
        assert vector_store.size == 1
        assert vector_store.vectors is not None
        assert vector_store.vectors.shape == (1, 4)


class TestVectorStoreSearch:
    """Tests for similarity search."""

    def test_search_empty_store(self, vector_store):
        """Searching an empty store returns no results."""
        results = vector_store.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=5,
        )
        assert results == []

    def test_search_returns_top_k(self, vector_store, sample_chunks, sample_vectors):
        """Search returns at most top_k results."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        results = vector_store.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=2,
        )
        assert len(results) <= 2

    def test_search_returns_relevant_first(self, vector_store, sample_chunks, sample_vectors):
        """Most similar chunk is returned first."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        results = vector_store.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        # c1 ([1,0,0,0]) should be most similar to query [1,0,0,0]
        assert results[0].chunk_id == "c1"
        assert results[0].score > 0.99

    def test_search_results_are_retrieved_chunks(self, vector_store, sample_chunks, sample_vectors):
        """Results are RetrievedChunk instances with scores."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        results = vector_store.search(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=3,
        )
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_min_similarity_filter(self, vector_store, sample_chunks, sample_vectors):
        """min_similarity filters out irrelevant chunks."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        # Query similar to c3, with high threshold
        results = vector_store.search(
            query_vector=[0.0, 0.0, 0.0, 1.0],
            top_k=5,
            min_similarity=0.99,  # very high
        )
        # Only c3 should pass (exact match)
        assert len(results) == 1
        assert results[0].chunk_id == "c3"

    def test_zero_query_vector_returns_empty(self, vector_store, sample_chunks, sample_vectors):
        """A zero query vector returns no results (avoid division by zero)."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        results = vector_store.search(
            query_vector=[0.0, 0.0, 0.0, 0.0],
            top_k=5,
        )
        assert results == []

    def test_search_rejects_wrong_query_dimension(self, vector_store, sample_chunks, sample_vectors):
        """Search with wrong query dimensions raises ValueError."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        with pytest.raises(ValueError, match="dimensions"):
            vector_store.search([1.0, 0.0], top_k=1)
            
    def test_search_rejects_nonpositive_top_k(self, vector_store):
        with pytest.raises(ValueError, match="top_k"):
            vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=-1)


class TestVectorStorePersistence:
    """Tests for SQLite persistence."""

    def test_persistence_across_instances(self, tmp_path: Path, sample_chunks, sample_vectors):
        """Chunks persist across VectorStore instances."""
        db_path = str(tmp_path / "persist.db")

        # First instance: add chunks
        store1 = VectorStore(db_path=db_path, embedding_dims=4)
        store1.add(sample_chunks, sample_vectors, "test-model")
        assert store1.size == 3
        store1.close()

        # Second instance: should load existing chunks
        store2 = VectorStore(db_path=db_path, embedding_dims=4)
        assert store2.size == 3

        # Search should work
        results = store2.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) > 0
        store2.close()

    def test_clear_removes_all(self, vector_store, sample_chunks, sample_vectors):
        """clear() removes all chunks."""
        vector_store.add(sample_chunks, sample_vectors, "test-model")
        assert vector_store.size == 3

        vector_store.clear()
        assert vector_store.size == 0

        # Search should return empty
        results = vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_load_skips_rows_without_embeddings(self, tmp_path: Path, sample_chunks, sample_vectors):
        db_path = str(tmp_path / "missing_embedding.db")
        store = VectorStore(db_path=db_path, embedding_dims=4)
        store._conn.execute(
            "INSERT INTO chunks (chunk_id, source, text, chunk_index, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("missing", "b.pdf", "no embedding", 0, "2000-01-01 00:00:00"),
        )
        store._conn.commit()
        store.add([sample_chunks[0]], [sample_vectors[0]], "test-model")
        store.close()

        reopened = VectorStore(db_path=db_path, embedding_dims=4)
        try:
            results = reopened.search([1.0, 0.0, 0.0, 0.0], top_k=5)
            assert [result.chunk_id for result in results] == ["c1"]
            assert reopened.size == 1
        finally:
            reopened.close()
