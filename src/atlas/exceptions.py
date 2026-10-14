"""
ATLAS exception hierarchy.

Design principle: catch specific exceptions at boundaries, never catch
Exception silently. A bare `except: pass` turns a recoverable failure
into silent corruption that surfaces three days later in a user report.

The hierarchy is shallow (one base + 5 specific) on purpose. Deep
hierarchies encourage over-catching. Five well-named exceptions cover
every failure mode ATLAS can produce.
"""

from __future__ import annotations


class AtlasError(Exception):
    """Base exception for all ATLAS errors.

    Every ATLAS-specific exception inherits from this. Catch `AtlasError`
    at API boundaries to convert any internal failure into a 500 response.
    """


class ProviderError(AtlasError):
    """An LLM or embedding provider failed.

    Examples: Groq rate limit, HuggingFace timeout, malformed API response.
    The agent loop catches this and returns an ErrorResponse instead of crashing.
    """


class ToolError(AtlasError):
    """A tool failed during execution.

    Examples: retriever found no chunks, web search returned 500, file
    tool permission denied. The dispatcher catches this and logs the
    failure without crashing the run.
    """


class RetrievalError(AtlasError):
    """The retrieval pipeline failed to produce results.

    Examples: vector store is empty, embedding mismatch, similarity
    threshold too high. Distinct from ToolError because retrieval failures
    are often recoverable (try a different query) while tool failures
    usually are not.
    """


class BudgetExhausted(AtlasError):
    """The run exceeded a configured bound.

    Examples: max_steps reached, timeout, token budget exhausted, stuck.
    The agent loop raises this to terminate gracefully. The caller
    catches it and returns a PartialResponse.
    """


class SafetyViolation(AtlasError):
    """A safety policy was violated.

    Examples: prompt injection detected, dangerous tool call attempted,
    secret leakage prevention triggered. Logged at ERROR level and
    surfaced to the user as a refusal, not an error.
    """