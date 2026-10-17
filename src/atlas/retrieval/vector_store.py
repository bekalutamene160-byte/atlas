"""
Vector store — in-memory numpy + SQLite persistence.

Stores chunk embeddings in a numpy matrix for fast cosine similarity
search. Chunks and embeddings are persisted to SQLite so they survive
restarts.

Design:
- On startup: load chunks + embeddings from SQLite → numpy matrix
- On add(): insert chunks + embeddings into SQLite + numpy matrix
- On search(): cosine similarity against numpy matrix → top-K chunks

This is brute-force search (O(N) per query). For <10K chunks, this is
sub-millisecond. For 100K+ chunks, switch to pgvector with HNSW index.

Bug fixes (Day 3 audit):
- Vector dimensions validated before write (was: corrupted on bad input)
- Duplicate chunk IDs in same batch now deduplicated (was: left duplicates in memory)
- Rows without embeddings skipped on reload (was: misaligned chunks vs vectors)
- Query vector dimensions validated in search (was: low-level NumPy error)
- Writes are now transactional (was: partial state on failure)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from atlas.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store with SQLite persistence.

    The numpy matrix is the hot path (fast similarity search).
    SQLite is the cold path (persistence across restarts).
    """

    def __init__(self, db_path: str, embedding_dims: int = 384):
        self.db_path = db_path
        self.dims = embedding_dims

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.chunks: list[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._normalized: Optional[np.ndarray] = None

        # SQLite connection — initialized directly (non-Optional)
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._init_tables()
        self._load_from_db()

    def _init_tables(self) -> None:
        """Create SQLite tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                page INTEGER,
                section TEXT,
                text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding_model TEXT,
                embedding_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)
        """)
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load all chunks + embeddings from SQLite into memory.

        Bug fix: rows without embeddings (or with malformed embeddings)
        are skipped with a warning. Previously, a row with NULL embedding_json
        would add a chunk to self.chunks but no vector to the vectors list,
        causing misalignment between chunks and vectors after reload.
        """
        cursor = self._conn.execute(
            "SELECT chunk_id, source, page, section, text, chunk_index, embedding_json "
            "FROM chunks ORDER BY created_at"
        )

        vectors: list[list[float]] = []
        skipped = 0

        for row in cursor:
            chunk = Chunk(
                chunk_id=row[0],
                source=row[1],
                page=row[2],
                section=row[3],
                text=row[4],
                chunk_index=row[5],
            )

            embedding_json = row[6]
            if not embedding_json:
                # Row has no embedding — skip it (can't search without a vector)
                skipped += 1
                continue

            try:
                vec = json.loads(embedding_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Skipping chunk {chunk.chunk_id}: malformed embedding JSON"
                )
                skipped += 1
                continue

            # Validate dimensions match what the store expects
            if len(vec) != self.dims:
                logger.warning(
                    f"Skipping chunk {chunk.chunk_id}: dimension mismatch "
                    f"(expected {self.dims}, got {len(vec)})"
                )
                skipped += 1
                continue

            # Only add chunk + vector together (keeps them aligned)
            self.chunks.append(chunk)
            vectors.append(vec)

        if vectors:
            self.vectors = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._normalized = self.vectors / norms

        if skipped > 0:
            logger.warning(
                f"Skipped {skipped} chunks with missing/malformed embeddings"
            )

        logger.info(f"Loaded {len(self.chunks)} chunks from SQLite")

    def add(self, chunks: list[Chunk], vectors: list[list[float]],
            embedding_model: str) -> None:
        """Add chunks + embeddings to the store.

        Bug fixes:
        - Validates ALL vector dimensions before writing anything (transactional)
        - Deduplicates within the same batch (was: could leave duplicates in memory)
        - If validation fails, nothing is written (no partial state)

        Args:
            chunks: List of Chunk objects to add
            vectors: List of embedding vectors (must match chunks length)
            embedding_model: Name of the model that produced the embeddings

        Raises:
            ValueError: if dimensions mismatch or lengths don't match
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have same length "
                f"(got {len(chunks)} chunks, {len(vectors)} vectors)"
            )

        # Bug fix 5: Validate ALL vector dimensions BEFORE writing anything
        # Previously, a bad vector would corrupt the write midway through
        for i, vec in enumerate(vectors):
            if len(vec) != self.dims:
                raise ValueError(
                    f"Vector {i} has {len(vec)} dimensions, expected {self.dims}. "
                    f"Chunk ID: {chunks[i].chunk_id}. "
                    f"This usually means the embedding model changed without re-indexing."
                )

        existing_ids = {c.chunk_id for c in self.chunks}
        seen_in_batch: set[str] = set()  # Bug fix 6: track within-batch duplicates
        new_chunks: list[Chunk] = []
        new_vectors: list[list[float]] = []

        for chunk, vec in zip(chunks, vectors):
            # Skip if already in store OR already seen in this batch
            if chunk.chunk_id in existing_ids or chunk.chunk_id in seen_in_batch:
                continue
            seen_in_batch.add(chunk.chunk_id)
            new_chunks.append(chunk)
            new_vectors.append(vec)

        if not new_chunks:
            logger.info("No new chunks to add (all duplicates)")
            return

        # Transactional write: all or nothing
        # Bug fix 5: if any insert fails, rollback the entire batch
        try:
            self._conn.execute("BEGIN")
            for chunk, vec in zip(new_chunks, new_vectors):
                self._conn.execute(
                    "INSERT OR IGNORE INTO chunks "
                    "(chunk_id, source, page, section, text, chunk_index, "
                    "embedding_model, embedding_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.source,
                        chunk.page,
                        chunk.section,
                        chunk.text,
                        chunk.chunk_index,
                        embedding_model,
                        json.dumps(vec),
                    ),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            logger.error("Failed to write chunks to SQLite, rolled back")
            raise

        # Update in-memory state only after successful DB write
        self.chunks.extend(new_chunks)
        new_vec_array = np.array(new_vectors, dtype=np.float32)

        if self.vectors is None:
            self.vectors = new_vec_array
        else:
            self.vectors = np.vstack([self.vectors, new_vec_array])

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._normalized = self.vectors / norms

        logger.info(f"Added {len(new_chunks)} chunks (total: {len(self.chunks)})")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Search for chunks similar to the query vector.

        Bug fixes:
        - Validates query vector dimensions (was: low-level NumPy error)
        - Validates query vector has no NaN/Inf values
        - Validates top_k and min_similarity bounds

        Args:
            query_vector: The query embedding.
            top_k: Maximum number of chunks to return.
            min_similarity: Minimum cosine similarity (0.0 to 1.0).

        Returns:
            List of RetrievedChunk objects, sorted by similarity (desc).

        Raises:
            ValueError: if query dimensions don't match or contain NaN/Inf
        """
        # Bug fix 8: validate query vector dimensions before NumPy operations
        if len(query_vector) != self.dims:
            raise ValueError(
                f"Query vector has {len(query_vector)} dimensions, "
                f"expected {self.dims}. This usually means the query was "
                f"embedded with a different model than the corpus."
            )

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError(
                f"min_similarity must be between 0.0 and 1.0, got {min_similarity}"
            )

        if self._normalized is None or len(self.chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)

        # Bug fix 8: check for NaN/Inf that would corrupt similarity scores
        if not np.all(np.isfinite(query)):
            raise ValueError("Query vector contains NaN or infinity values")

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query_normalized = query / query_norm

        similarities = self._normalized @ query_normalized

        mask = similarities >= min_similarity
        if not mask.any():
            return []

        indices = np.where(mask)[0]
        sims_filtered = similarities[mask]

        top_indices_local = np.argsort(-sims_filtered)[:top_k]

        results = []
        for local_idx in top_indices_local:
            global_idx = indices[local_idx]
            chunk = self.chunks[global_idx]
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                page=chunk.page,
                section=chunk.section,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                score=float(similarities[global_idx]),
            ))

        return results

    def clear(self) -> None:
        """Remove all chunks and embeddings."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()
        self.chunks = []
        self.vectors = None
        self._normalized = None
        logger.info("Cleared all chunks from store")

    @property
    def size(self) -> int:
        """Number of chunks in the store."""
        return len(self.chunks)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()
