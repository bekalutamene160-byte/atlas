"""Unit tests for the EvalRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas.eval.runner import CaseResult, EvalReport, EvalRunner


@pytest.fixture
def eval_file(tmp_path: Path) -> Path:
    """Create a temporary eval file with 4 test cases."""
    cases = [
        {
            "case_id": "fact_01",
            "category": "factual",
            "query": "What is the pressure?",
            "expected": {"min_citations": 1, "must_contain_any": ["5 MPa"]},
        },
        {
            "case_id": "fact_02",
            "category": "factual",
            "query": "What is the temperature?",
            "expected": {"min_citations": 1, "must_contain_any": ["80 C"]},
        },
        {
            "case_id": "off_01",
            "category": "off_topic",
            "query": "How to bake bread?",
            "expected": {"should_refuse": True},
        },
        {
            "case_id": "multi_01",
            "category": "multi_hop",
            "query": "Compare pressures.",
            "expected": {
                "min_citations": 2,
                "must_contain_all": ["Atlas-3", "Atlas-5"],
                "must_compare": True,
            },
        },
    ]

    eval_path = tmp_path / "benchmark.jsonl"
    eval_path.write_text("\n".join(json.dumps(c) for c in cases))
    return eval_path


@pytest.fixture
def mock_agent():
    """A mock agent that returns canned responses."""
    agent = MagicMock()
    agent.run = AsyncMock()

    # Script responses for each case
    responses = {
        "What is the pressure?": {
            "status": "done",
            "answer": "The pressure is 5 MPa [e1].",
            "citations": [{"evidence_id": "e1", "source": "spec.pdf", "page": 1}],
            "evidence_count": 1,
            "steps_taken": 2,
            "cost_tokens": 100,
        },
        "What is the temperature?": {
            "status": "done",
            "answer": "The temperature is 80 C [e1].",
            "citations": [{"evidence_id": "e1", "source": "spec.pdf", "page": 2}],
            "evidence_count": 1,
            "steps_taken": 2,
            "cost_tokens": 100,
        },
        "How to bake bread?": {
            "status": "refused",
            "reason": "corpus does not contain information about baking",
            "evidence_count": 0,
            "steps_taken": 1,
            "cost_tokens": 50,
        },
        "Compare pressures.": {
            "status": "done",
            "answer": "The Atlas-3 has 5 MPa while the Atlas-5 has 8 MPa. "
                      "The Atlas-5 is higher [e1] [e2].",
            "citations": [
                {"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "atlas-5-spec.pdf", "page": 1},
            ],
            "evidence_count": 2,
            "steps_taken": 3,
            "cost_tokens": 200,
        },
    }

    async def mock_run(query):
        return responses.get(query, {"status": "error", "answer": "", "citations": []})

    agent.run = mock_run
    return agent


class TestEvalRunner:
    """Tests for the EvalRunner."""

    async def test_run_all_returns_report(self, mock_agent, eval_file):
        """run_all returns an EvalReport."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()
        assert isinstance(report, EvalReport)
        assert report.total_cases == 4

    async def test_all_cases_pass(self, mock_agent, eval_file):
        """All 4 cases should pass with the mock agent."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()
        assert report.total_passed == 4
        assert report.success_rate == 100.0

    async def test_category_breakdown(self, mock_agent, eval_file):
        """Category breakdown shows per-category stats."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()

        cats = report.category_breakdown()
        assert "factual" in cats
        assert "off_topic" in cats
        assert "multi_hop" in cats
        assert cats["factual"]["total"] == 2
        assert cats["factual"]["passed"] == 2
        assert cats["off_topic"]["passed"] == 1

    async def test_summary_string(self, mock_agent, eval_file):
        """Summary produces a readable string."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()

        summary = report.summary()
        assert "ATLAS EVALUATION REPORT" in summary
        assert "Success Rate: 100.0%" in summary

    async def test_save_baseline(self, mock_agent, eval_file, tmp_path):
        """save_baseline writes a JSON file."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()

        baseline_path = tmp_path / "baselines" / "test_baseline.json"
        runner.save_baseline(report, baseline_path)

        assert baseline_path.exists()
        data = json.loads(baseline_path.read_text())
        assert data["total_cases"] == 4
        assert data["total_passed"] == 4

    async def test_failed_case_doesnt_crash(self, eval_file, tmp_path):
        """A case that raises an exception is marked as error, not crashed."""
        agent = MagicMock()

        async def mock_run(query):
            if "bread" in query:
                raise RuntimeError("API exploded")
            if "Compare" in query:
                return {"status": "done",
                        "answer": "Atlas-3 has 5 MPa while Atlas-5 has 8 MPa [e1] [e2]",
                        "citations": [
                            {"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1},
                            {"evidence_id": "e2", "source": "atlas-5-spec.pdf", "page": 1},
                        ],
                        "evidence_count": 2, "steps_taken": 2, "cost_tokens": 100}
            if "pressure" in query:
                return {"status": "done", "answer": "5 MPa [e1]",
                        "citations": [{"evidence_id": "e1", "source": "spec.pdf", "page": 1}],
                        "evidence_count": 1, "steps_taken": 1, "cost_tokens": 50}
            if "temperature" in query:
                return {"status": "done", "answer": "80 C [e1]",
                        "citations": [{"evidence_id": "e1", "source": "spec.pdf", "page": 1}],
                        "evidence_count": 1, "steps_taken": 1, "cost_tokens": 50}
            return {"status": "error", "answer": "", "citations": []}

        agent.run = mock_run

        runner = EvalRunner(agent, eval_file)
        report = await runner.run_all()

        # 3 passed, 1 failed (the bread one)
        assert report.total_passed == 3
        assert report.total_cases == 4

        # The failed case has an error
        failed = [r for r in report.results if not r.passed]
        assert len(failed) == 1
        assert failed[0].error is not None
        assert "API exploded" in failed[0].error

    async def test_to_dict_serializable(self, mock_agent, eval_file):
        """Report to_dict is JSON serializable."""
        runner = EvalRunner(mock_agent, eval_file)
        report = await runner.run_all()

        data = report.to_dict()
        # Should not raise
        json.dumps(data)
