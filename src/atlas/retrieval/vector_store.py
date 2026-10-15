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

        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self.chunks: list[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self._normalized: Optional[np.ndarray] = None

        # SQLite connection — initialized directly (not via separate method)
        # This is non-Optional because we always assign it here.
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path, check_same_thread=False
        )
        self._init_tables()

        # Load existing data from DB
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
        """Load all chunks + embeddings from SQLite into memory."""
        cursor = self._conn.execute(
            "SELECT chunk_id, source, page, section, text, chunk_index, embedding_json "
            "FROM chunks ORDER BY created_at"
        )

        vectors = []
        for row in cursor:
            chunk = Chunk(
                chunk_id=row[0],
                source=row[1],
                page=row[2],
                section=row[3],
                text=row[4],
                chunk_index=row[5],
            )
            self.chunks.append(chunk)

            if row[6]:
                vec = json.loads(row[6])
                vectors.append(vec)

        if vectors:
            self.vectors = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._normalized = self.vectors / norms

        logger.info(f"Loaded {len(self.chunks)} chunks from SQLite")

    def add(self, chunks: list[Chunk], vectors: list[list[float]],
            embedding_model: str) -> None:
        """Add chunks + embeddings to the store.

        Deduplicates by chunk_id — if a chunk with the same ID exists,
        it's skipped (not updated).
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have same length")

        existing_ids = {c.chunk_id for c in self.chunks}
        new_chunks = []
        new_vectors = []

        for chunk, vec in zip(chunks, vectors):
            if chunk.chunk_id in existing_ids:
                continue
            new_chunks.append(chunk)
            new_vectors.append(vec)

        if not new_chunks:
            logger.info("No new chunks to add (all duplicates)")
            return

        # Add to SQLite
        for chunk, vec in zip(new_chunks, new_vectors):
            self._conn.execute(
                "INSERT OR IGNORE INTO chunks "
                "(chunk_id, source, page, section, text, chunk_index, embedding_model, embedding_json) "
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
        self._conn.commit()

        # Add to in-memory state
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

        Uses cosine similarity (dot product of normalized vectors).
        Returns top-K chunks with similarity >= min_similarity.
        """
        if self._normalized is None or len(self.chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
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
