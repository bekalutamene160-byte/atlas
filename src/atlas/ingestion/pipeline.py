"""
Ingestion pipeline — orchestrates parser → chunker → embedder → vector store.

This is the entry point for corpus ingestion. Run it with:

    python -m atlas.ingestion.pipeline

Or programmatically:

    from atlas.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline(settings)
    await pipeline.run()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from atlas.config import Settings, settings as default_settings
from atlas.ingestion.chunker import RecursiveCharacterSplitter
from atlas.ingestion.embedder import Embedder, make_embedder
from atlas.ingestion.parser import DocumentParser
from atlas.models import Chunk
from atlas.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline.

    Steps:
    1. Parse all files in corpus_path
    2. Split each document into chunks
    3. Embed all chunks (with caching)
    4. Add chunks + embeddings to the vector store
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[RecursiveCharacterSplitter] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.settings = settings or default_settings
        self.parser = parser or DocumentParser()
        self.chunker = chunker or RecursiveCharacterSplitter(
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        self.embedder = embedder or make_embedder(self.settings)
        self.vector_store = vector_store or VectorStore(
            db_path=self.settings.db_path,
            embedding_dims=self.embedder.dimensions,
        )

    async def run(self, corpus_path: Optional[Path] = None) -> dict:
        """Run the full ingestion pipeline.

        Args:
            corpus_path: Override the corpus path from settings.

        Returns:
            Dict with stats: documents_parsed, chunks_created,
            embeddings_computed, total_time_seconds
        """
        import time
        start = time.time()

        corpus_path = corpus_path or Path(self.settings.corpus_path)

        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

        logger.info(f"Starting ingestion from: {corpus_path}")

        # Step 1: Parse all documents
        documents = await self.parser.parse_directory(corpus_path)
        logger.info(f"Parsed {len(documents)} documents")

        # Step 2: Split into chunks
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self.chunker.split_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"  {doc.source}: {len(chunks)} chunks")

        logger.info(f"Total chunks: {len(all_chunks)}")

        # Step 3: Embed all chunks (with caching)
        texts = [c.text for c in all_chunks]
        vectors = await self.embedder.embed(texts)

        logger.info(f"Computed {len(vectors)} embeddings")

        # Step 4: Add to vector store
        self.vector_store.add(
            chunks=all_chunks,
            vectors=vectors,
            embedding_model=self.embedder.model_name,
        )

        elapsed = time.time() - start
        stats = {
            "documents_parsed": len(documents),
            "chunks_created": len(all_chunks),
            "embeddings_computed": len(vectors),
            "total_time_seconds": round(elapsed, 2),
        }

        logger.info(f"Ingestion complete in {elapsed:.1f}s: {stats}")
        return stats

    def close(self) -> None:
        """Clean up resources."""
        self.vector_store.close()


async def main():
    """CLI entry point: python -m atlas.ingestion.pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = IngestionPipeline()
    try:
        stats = await pipeline.run()
        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"  vector_store_size: {pipeline.vector_store.size}")
        print("=" * 60)
    finally:
        pipeline.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
