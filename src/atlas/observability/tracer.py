"""
Tracer — structured JSONL logging for agent runs.

Writes one JSONL file per run to logs/{run_id}.jsonl. Each line is a
JSON event with timestamp, run_id, step, and event-specific data.

The /runs/{run_id} API endpoint reads these files to return the full
trace of a past run.

Design: the tracer is a simple file-based logger. No database, no
external service. For production at scale, replace with OpenTelemetry
or a proper log aggregation service (ELK, Datadog).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class Tracer:
    """File-based tracer for agent runs.

    Each run produces a file: logs/{run_id}.jsonl
    Each line is a JSON event.
    """

    def __init__(self, logs_path: str = "./logs"):
        self.logs_path = Path(logs_path)
        self.logs_path.mkdir(parents=True, exist_ok=True)

    def start_run(self, run_id: str, query: str) -> None:
        """Log the start of a run."""
        self._append(run_id, {
            "timestamp": time.time(),
            "run_id": run_id,
            "step_id": 0,
            "event": "run_started",
            "query": query[:200],  # truncate long queries
        })

    def log_event(self, run_id: str, event_type: str,
                  step: int = 0, **kwargs) -> None:
        """Log a generic event during a run."""
        self._append(run_id, {
            "timestamp": time.time(),
            "run_id": run_id,
            "step_id": step,
            "event": event_type,
            **kwargs,
        })

    def log_step(self, run_id: str, step: int, status: str,
                 tokens: int = 0, **kwargs) -> None:
        """Log a step transition."""
        self._append(run_id, {
            "timestamp": time.time(),
            "run_id": run_id,
            "step_id": step,
            "event": "step",
            "status": status,
            "tokens_used": tokens,
            **kwargs,
        })

    def log_completion(self, run_id: str, status: str,
                       steps: int, tokens: int, **kwargs) -> None:
        """Log the completion of a run."""
        self._append(run_id, {
            "timestamp": time.time(),
            "run_id": run_id,
            "step_id": steps,
            "event": "run_completed",
            "status": status,
            "total_tokens": tokens,
            **kwargs,
        })

    def get_trace(self, run_id: str) -> list[dict]:
        """Read all events for a run.

        Returns empty list if the run doesn't exist.
        """
        path = self.logs_path / f"{run_id}.jsonl"
        if not path.exists():
            return []

        events: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Malformed trace line in {run_id}: {line[:100]}")
        return events

    def _append(self, run_id: str, event: dict) -> None:
        """Append an event to the run's trace file."""
        path = self.logs_path / f"{run_id}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.error(f"Failed to write trace for {run_id}: {e}")
