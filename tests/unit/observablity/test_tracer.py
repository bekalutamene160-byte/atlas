"""Unit tests for the Tracer."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.observability.tracer import Tracer


@pytest.fixture
def tracer(tmp_path: Path) -> Tracer:
    """A tracer with a temporary logs directory."""
    return Tracer(logs_path=str(tmp_path / "logs"))


class TestTracer:
    """Tests for the file-based tracer."""

    def test_start_run_creates_file(self, tracer: Tracer):
        """start_run creates a JSONL file."""
        tracer.start_run("test-123", "What is the pressure?")
        trace_file = Path(tracer.logs_path) / "test-123.jsonl"
        assert trace_file.exists()

    def test_start_run_writes_first_event(self, tracer: Tracer):
        """The first event is 'run_started' with the query."""
        tracer.start_run("test-123", "What is the pressure?")
        events = tracer.get_trace("test-123")
        assert len(events) == 1
        assert events[0]["event"] == "run_started"
        assert events[0]["query"] == "What is the pressure?"
        assert events[0]["run_id"] == "test-123"

    def test_log_event_appends(self, tracer: Tracer):
        """log_event appends to the existing file."""
        tracer.start_run("test-456", "query")
        tracer.log_event("test-456", "step", step=1, status="planning")
        tracer.log_event("test-456", "evidence", step=1, count=5)

        events = tracer.get_trace("test-456")
        assert len(events) == 3
        assert events[1]["event"] == "step"
        assert events[1]["status"] == "planning"
        assert events[2]["event"] == "evidence"
        assert events[2]["count"] == 5

    def test_log_completion(self, tracer: Tracer):
        """log_completion writes the final event."""
        tracer.start_run("test-789", "query")
        tracer.log_completion("test-789", "done", steps=3, tokens=500)

        events = tracer.get_trace("test-789")
        assert len(events) == 2
        assert events[1]["event"] == "run_completed"
        assert events[1]["status"] == "done"
        assert events[1]["total_tokens"] == 500

    def test_get_trace_nonexistent_returns_empty(self, tracer: Tracer):
        """get_trace returns empty list for non-existent run."""
        events = tracer.get_trace("nonexistent-id")
        assert events == []

    def test_logs_directory_created(self, tmp_path: Path):
        """Tracer creates the logs directory if it doesn't exist."""
        logs_path = str(tmp_path / "deep" / "nested" / "logs")
        t = Tracer(logs_path=logs_path)
        assert Path(logs_path).exists()

    def test_long_query_truncated(self, tracer: Tracer):
        """Long queries are truncated to 200 chars."""
        long_query = "A" * 300
        tracer.start_run("test-long", long_query)
        events = tracer.get_trace("test-long")
        assert len(events[0]["query"]) <= 200
