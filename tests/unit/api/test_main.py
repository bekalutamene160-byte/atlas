"""Unit tests for the FastAPI application."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from atlas.api.main import app, get_agent


class MockAgent:
    """A mock agent for API testing — returns canned responses."""

    async def run(self, query: str) -> dict:
        return {
            "run_id": "mock-run-123",
            "query": query,
            "status": "done",
            "answer": f"The answer to '{query}' is 5 MPa [e1].",
            "citations": [
                {"evidence_id": "e1", "source": "atlas-3-spec.txt", "page": 1, "section": None}
            ],
            "steps_taken": 2,
            "evidence_count": 3,
            "cost_tokens": 150,
        }

    async def run_streaming(self, query: str):
        """Yield canned events for the SSE test."""
        yield {"type": "start", "run_id": "mock-stream-123", "query": query}
        yield {"type": "step", "step": 1, "status": "planning"}
        yield {"type": "step", "step": 1, "status": "retrieving"}
        yield {"type": "evidence", "count": 3}
        yield {"type": "step", "step": 2, "status": "synthesizing"}
        yield {"type": "step", "step": 2, "status": "criticizing"}
        yield {
            "type": "done",
            "result": {
                "run_id": "mock-stream-123",
                "query": query,
                "status": "done",
                "answer": f"The answer is 5 MPa [e1].",
                "citations": [{"evidence_id": "e1", "source": "spec.txt", "page": 1}],
                "steps_taken": 2,
                "evidence_count": 3,
                "cost_tokens": 150,
            },
        }

    def close(self):
        pass


@pytest.fixture
def client():
    """A test client with the mock agent override."""
    app.dependency_overrides[get_agent] = lambda: MockAgent()
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "llm_provider" in data
        assert "embedding_provider" in data


class TestAskEndpoint:
    """Tests for POST /ask."""

    def test_ask_returns_200(self, client: TestClient):
        response = client.post("/ask", json={"query": "What is the pressure?"})
        assert response.status_code == 200

    def test_ask_returns_answer(self, client: TestClient):
        response = client.post("/ask", json={"query": "What is the pressure?"})
        data = response.json()
        assert data["status"] == "done"
        assert "5 MPa" in data["answer"]
        assert len(data["citations"]) > 0

    def test_ask_returns_metadata(self, client: TestClient):
        response = client.post("/ask", json={"query": "test"})
        data = response.json()
        assert "run_id" in data
        assert "steps_taken" in data
        assert "evidence_count" in data
        assert "cost_tokens" in data

    def test_ask_empty_query_returns_422(self, client: TestClient):
        """Empty query is rejected by Pydantic validation."""
        response = client.post("/ask", json={"query": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_ask_missing_query_returns_422(self, client: TestClient):
        """Missing query field returns 422."""
        response = client.post("/ask", json={})
        assert response.status_code == 422


class TestStreamEndpoint:
    """Tests for POST /stream (SSE)."""

    def test_stream_returns_200(self, client: TestClient):
        response = client.post("/stream", json={"query": "test"})
        assert response.status_code == 200

    def test_stream_returns_event_stream(self, client: TestClient):
        response = client.post("/stream", json={"query": "test"})
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_emits_events(self, client: TestClient):
        """The SSE stream contains expected event types."""
        response = client.post("/stream", json={"query": "test"})
        body = response.text

        # Should contain SSE data lines
        assert "data: " in body
        # Should contain a start event
        assert '"type": "start"' in body or '"type":"start"' in body
        # Should contain a done event
        assert '"type": "done"' in body or '"type":"done"' in body


class TestRunsEndpoint:
    """Tests for GET /runs/{run_id}."""

    def test_unknown_run_returns_404(self, client: TestClient):
        response = client.get("/runs/nonexistent-id")
        assert response.status_code == 404


class TestIndexEndpoint:
    """Tests for GET / (the web UI)."""

    def test_index_returns_200(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
