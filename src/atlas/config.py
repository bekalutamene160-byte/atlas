"""
ATLAS configuration.

Single source of truth for all settings. Type-safe, environment-driven,
with safe defaults that allow the application to start in mock mode
without any external dependencies.

Architecture decision: every other module reads from `settings`. No module
reads environment variables directly. This centralizes configuration and
makes the application testable — tests can construct a custom Settings
instance and pass it in.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All ATLAS configuration. Loaded from environment variables or .env file.

    Every field has a default that allows the app to start in mock mode.
    Set ATLAS_LLM_PROVIDER=groq and ATLAS_GROQ_API_KEY=... to use real models.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ATLAS_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars (useful for docker-compose)
    )

    # --- Provider Selection ---
    llm_provider: Literal["groq", "mock"] = "mock"
    embedding_provider: Literal["hf", "local", "mock"] = "local"

    # --- Groq (LLM) ---
    groq_api_key: str = ""
    groq_planner_model: str = "llama-3.3-70b-versatile"
    groq_synthesizer_model: str = "llama-3.3-70b-versatile"
    groq_critic_model: str = "llama-3.1-8b-instant"

    # --- HuggingFace (Embeddings) ---
    hf_api_token: str = ""
    hf_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    # --- Agent Bounds ---
    max_agent_steps: int = Field(default=8, ge=1, le=50)
    run_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    max_tokens_per_run: int = Field(default=30_000, ge=1000, le=1_000_000)
    max_tool_calls: int = Field(default=12, ge=1, le=50)
    stuck_window: int = Field(default=3, ge=2, le=10)

    # --- Retrieval ---
    chunk_size: int = Field(default=800, ge=100, le=10_000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)

    # --- Storage ---
    db_path: str = "./data/atlas.db"
    corpus_path: str = "./data/corpus"
    logs_path: str = "./logs"

    # --- API ---
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    def is_mock_mode(self) -> bool:
        """True if both LLM and embeddings are in mock mode (no internet needed)."""
        return self.llm_provider == "mock" and self.embedding_provider == "mock"


# Single module-level instance. This is the ONLY global in the project.
# Tests construct their own Settings instances; production uses this one.
settings = Settings()
