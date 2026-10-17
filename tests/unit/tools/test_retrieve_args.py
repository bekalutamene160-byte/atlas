"""Unit tests for RetrieveArgs validation (bugs 1-4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas.tools.retriever_tool import RetrieveArgs


class TestRetrieveArgsValidation:
    """Tests for RetrieveArgs input validation."""

    def test_valid_args(self):
        """Valid args with query and default top_k."""
        args = RetrieveArgs(query="pump pressure")
        assert args.query == "pump pressure"
        assert args.top_k == 5

    def test_custom_top_k(self):
        """top_k can be overridden."""
        args = RetrieveArgs(query="test", top_k=10)
        assert args.top_k == 10

    def test_empty_query_rejected(self):
        """Bug 1: empty query is rejected."""
        with pytest.raises(ValidationError):
            RetrieveArgs(query="")

    def test_whitespace_query_rejected(self):
        """Bug 2: whitespace-only query is rejected."""
        with pytest.raises(ValidationError):
            RetrieveArgs(query="   ")

    def test_query_is_trimmed(self):
        """Query is stripped of leading/trailing whitespace."""
        args = RetrieveArgs(query="  pump pressure  ")
        assert args.query == "pump pressure"

    def test_top_k_zero_rejected(self):
        """Bug 3: top_k=0 is rejected."""
        with pytest.raises(ValidationError):
            RetrieveArgs(query="test", top_k=0)

    def test_top_k_negative_rejected(self):
        """Bug 4: negative top_k is rejected."""
        with pytest.raises(ValidationError):
            RetrieveArgs(query="test", top_k=-1)

    def test_top_k_one_accepted(self):
        """top_k=1 is the minimum valid value."""
        args = RetrieveArgs(query="test", top_k=1)
        assert args.top_k == 1
