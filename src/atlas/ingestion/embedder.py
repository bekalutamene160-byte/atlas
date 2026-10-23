"""
Embedder — converts text into vector embeddings.

This version uses the `requests` library to call Jina AI API.
Jina is highly stable and not blocked by standard firewalls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from atlas.exceptions import ProviderError

logger = logging.getLogger(__name__)


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError


class MockEmbedder(Embedder):
    def __init__(self, dims: int = 768):
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


class JinaEmbedder(Embedder):
    """Jina AI API embedder using the bulletproof `requests` library."""

    def __init__(self, api_token: str, dims: int = 768):
        self.api_token = api_token
        self._model_name = "jina-embeddings-v2-base-en"
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using Jina API with retry on connection drops."""
        if not texts:
            return []

        def _embed_sync() -> list[list[float]]:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Create a session with retry strategy
            session = requests.Session()
            retry_strategy = Retry(
                total=3,  # max 3 retries
                backoff_factor=2,  # wait 2s, 4s, 8s between retries
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
            all_embeddings: list[list[float]] = []
            batch_size = 16
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                url = "https://api.jina.ai/v1/embeddings"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_token}"
                }
                data = {
                    "model": self._model_name,
                    "input": batch
                }
                
                try:
                    response = session.post(url, headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    
                    for item in result["data"]:
                        all_embeddings.append(item["embedding"])
                        
                except requests.exceptions.ConnectionError as e:
                    logger.error(f"Jina connection error (retrying): {e}")
                    # Wait and retry once more
                    import time
                    time.sleep(5)
                    response = session.post(url, headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    for item in result["data"]:
                        all_embeddings.append(item["embedding"])
                        
                except requests.exceptions.RequestException as e:
                    logger.error(f"Jina API error: {e}")
                    raise ProviderError(f"Jina API request failed: {e}") from e
                    
            return all_embeddings

        return await asyncio.to_thread(_embed_sync)


class CachedEmbedder(Embedder):
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
        key = f"{self.inner.model_name}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[Optional[list[float]]] = [None] * len(texts)
        misses: list[str] = []
        miss_indices: list[int] = []

        for i, text in enumerate(texts):
            h = self._hash(text)

            if h in self._mem_cache:
                results[i] = self._mem_cache[h]
                continue

            cache_file = self.cache_dir / f"{h}.json"
            if cache_file.exists():
                try:
                    vec = json.loads(cache_file.read_text())
                    self._mem_cache[h] = vec
                    results[i] = vec
                    continue
                except Exception:
                    pass

            misses.append(text)
            miss_indices.append(i)

        if misses:
            logger.info(f"Embedding {len(misses)} new texts (cache hit: {len(texts) - len(misses)})")
            new_vectors = await self.inner.embed(misses)

            if new_vectors is None:
                raise ProviderError("Inner embedder returned None.")
            
            if len(new_vectors) != len(misses):
                raise ProviderError(f"Embedder returned {len(new_vectors)} vectors, expected {len(misses)}.")

            for idx, text, vec in zip(miss_indices, misses, new_vectors):
                h = self._hash(text)
                self._mem_cache[h] = vec
                results[idx] = vec

                cache_file = self.cache_dir / f"{h}.json"
                try:
                    cache_file.write_text(json.dumps(vec))
                except Exception as e:
                    logger.warning(f"Failed to save embedding to cache: {e}")

        return [r for r in results if r is not None]


def make_embedder(settings) -> Embedder:
    if settings.embedding_provider == "mock":
        return MockEmbedder(dims=settings.embedding_dimensions)

    elif settings.embedding_provider == "jina":
        if not settings.jina_api_key:
            raise ValueError("ATLAS_EMBEDDING_PROVIDER=jina but ATLAS_JINA_API_KEY is empty.")
        inner = JinaEmbedder(
            api_token=settings.jina_api_key,
            dims=settings.embedding_dimensions
        )

    else:
        raise ValueError("Invalid embedding provider. Use 'jina' or 'mock'.")

    cache_dir = Path(settings.db_path).parent / "embeddings_cache"
    return CachedEmbedder(inner, cache_dir)