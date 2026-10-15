"""
Embedder — converts text into vector embeddings.

Three implementations behind one interface:
1. HuggingFaceEmbedder (primary): uses HuggingFace Inference API
   - Free with HF token
   - BAAI/bge-small-en-v1.5 (384 dims)
   - Requires internet

2. LocalSentenceTransformerEmbedder (fallback): runs on CPU
   - sentence-transformers/all-MiniLM-L6-v2 (384 dims)
   - No internet needed after first download
   - Slower but works offline

3. MockEmbedder (for tests): deterministic fake embeddings
   - Same text → same vector
   - No semantic meaning, just for testing

The CachedEmbedder wraps any of these with disk + memory caching.
Same text → same vector → cache hit → no API call → fast + free.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

from atlas.exceptions import ProviderError

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @property
    def dimensions(self) -> int:
        """Dimensionality of the vectors. Subclasses must override."""
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        """Name of the embedding model. Subclasses must override."""
        raise NotImplementedError


class MockEmbedder(Embedder):
    """Deterministic mock embedder for tests.

    Same text → same vector, every time. Uses SHA256 to seed the vector.
    No semantic meaning, but consistent — which is what tests need.
    """

    def __init__(self, dims: int = 384):
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return "mock-embedder"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            vec = [((h >> (i % 256)) & 1) * 0.5 + 0.5 for i in range(self._dims)]
            results.append(vec)
        return results


class HuggingFaceEmbedder(Embedder):
    """HuggingFace Inference API embedder.

    Calls the feature-extraction pipeline endpoint.
    Requires a HuggingFace API token (free at huggingface.co/settings/tokens).
    """

    def __init__(self, api_token: str, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.api_token = api_token
        self._model_name = model_name
        self._dims = 384  # BGE-small and MiniLM are both 384
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"Authorization": f"Bearer {api_token}"},
        )

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via HuggingFace Inference API.

        Batches up to 16 texts per call (HF limit). Retries on
        network errors with exponential backoff.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 16

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = await self._client.post(
                    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self._model_name}",
                    json={
                        "inputs": batch,
                        "options": {"wait_for_model": True},
                    },
                )
                response.raise_for_status()
                data = response.json()

                # HF returns a list of lists (one per input)
                # Each input's embedding is the mean of its token embeddings
                for item in data:
                    if isinstance(item[0], list):
                        # Token-level embeddings — average them
                        vec = [sum(t) / len(t) for t in zip(*item)]
                    else:
                        # Already a single vector
                        vec = item
                    all_embeddings.append(vec)

            except httpx.HTTPStatusError as e:
                logger.error(f"HuggingFace API error: {e.response.status_code}")
                if e.response.status_code == 401:
                    raise ProviderError("Invalid HuggingFace token") from e
                elif e.response.status_code == 429:
                    raise ProviderError("HuggingFace rate limit exceeded") from e
                else:
                    raise ProviderError(f"HuggingFace API error: {e.response.status_code}") from e

        return all_embeddings


class LocalSentenceTransformerEmbedder(Embedder):
    """Local sentence-transformers embedder (fallback).

    Runs on CPU. First load downloads the model (~80MB), cached for
    future runs. No internet needed after first download.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._dims: int = 384
        # Use a sentinel value instead of None to make the type clear.
        # The model is loaded lazily on first embed() call.
        self._model: Any = None  # type: ignore[assignment]
        self._loaded: bool = False

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> None:
        """Lazy-load the model. First call downloads it.

        Raises ProviderError if sentence-transformers is not installed
        or if the model fails to load.
        """
        if self._loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self._model_name}")
            loaded_model = SentenceTransformer(self._model_name)
            # get_sentence_embedding_dimension() may return None for some models.
            # Fall back to 384 (the known dimension for MiniLM) if it does.
            dim = loaded_model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dims = dim
            # else: keep the default 384

            self._model = loaded_model
            self._loaded = True

        except ImportError as e:
            raise ProviderError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            ) from e

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model.

        Runs in a thread to avoid blocking the event loop.
        """
        if not texts:
            return []

        import asyncio

        def _embed_sync() -> list[list[float]]:
            # Load model if not yet loaded (raises if it fails)
            self._load_model()

            # At this point, _model is guaranteed non-None because
            # _load_model() either succeeds (sets _model) or raises.
            # We assert to make this explicit for the type checker.
            assert self._model is not None, "model should be loaded after _load_model()"

            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [emb.tolist() for emb in embeddings]

        return await asyncio.to_thread(_embed_sync)


class CachedEmbedder(Embedder):
    """Wraps any embedder with disk + memory caching.

    Same text → same hash → cache hit → no API call.
    Saves to ./data/embeddings_cache/{hash}.json
    """

    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, list[float]] = {}

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def _hash(self, text: str) -> str:
        """Hash text + model name → cache key.

        Includes model name so switching models invalidates the cache.
        """
        key = f"{self.inner.model_name}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[str] = []
        miss_indices: list[int] = []

        # Check cache (memory first, then disk)
        for i, text in enumerate(texts):
            h = self._hash(text)

            # Memory cache
            if h in self._mem_cache:
                results[i] = self._mem_cache[h]
                continue

            # Disk cache
            cache_file = self.cache_dir / f"{h}.json"
            if cache_file.exists():
                try:
                    vec = json.loads(cache_file.read_text())
                    self._mem_cache[h] = vec
                    results[i] = vec
                    continue
                except Exception:
                    pass  # corrupted cache file, re-embed

            misses.append(text)
            miss_indices.append(i)

        # Embed only the misses
        if misses:
            logger.info(f"Embedding {len(misses)} new texts (cache hit: {len(texts) - len(misses)})")
            new_vectors = await self.inner.embed(misses)

            for idx, text, vec in zip(miss_indices, misses, new_vectors):
                h = self._hash(text)
                self._mem_cache[h] = vec
                results[idx] = vec

                # Save to disk
                cache_file = self.cache_dir / f"{h}.json"
                try:
                    cache_file.write_text(json.dumps(vec))
                except Exception as e:
                    logger.warning(f"Failed to save embedding to cache: {e}")

        return [r for r in results if r is not None]


def make_embedder(settings) -> Embedder:
    """Factory: pick the right embedder based on settings.

    Priority:
    1. "mock" → MockEmbedder (for tests)
    2. "hf" with token → CachedEmbedder(HuggingFaceEmbedder)
    3. "local" → CachedEmbedder(LocalSentenceTransformerEmbedder)
    4. Fallback: if "hf" but no token, fall back to local
    """
    if settings.embedding_provider == "mock":
        return MockEmbedder(dims=settings.embedding_dimensions)

    elif settings.embedding_provider == "hf":
        if not settings.hf_api_token:
            logger.warning(
                "ATLAS_EMBEDDING_PROVIDER=hf but ATLAS_HF_API_TOKEN is empty. "
                "Falling back to local sentence-transformers."
            )
            inner = LocalSentenceTransformerEmbedder(settings.local_embedding_model)
        else:
            inner = HuggingFaceEmbedder(
                api_token=settings.hf_api_token,
                model_name=settings.hf_embedding_model,
            )

    else:  # "local"
        inner = LocalSentenceTransformerEmbedder(settings.local_embedding_model)

    # Wrap in cache (always — caching is non-negotiable)
    cache_dir = Path(settings.db_path).parent / "embeddings_cache"
    return CachedEmbedder(inner, cache_dir)
