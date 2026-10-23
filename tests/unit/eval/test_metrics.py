"""Unit tests for evaluation metrics."""

from __future__ import annotations

import pytest

from atlas.eval.metrics import (
    compute_all_metrics,
    compute_citation_quality,
    compute_grounding_score,
    compute_task_success,
    compute_tool_correctness,
)


class TestTaskSuccess:
    """Tests for the task_success metric."""

    def test_factual_success(self):
        """Factual case with correct answer and citations passes."""
        response = {
            "status": "done",
            "answer": "The maximum operating pressure is 5 MPa [e1].",
            "citations": [{"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1}],
        }
        expected = {
            "min_citations": 1,
            "must_contain_any": ["5 MPa"],
            "must_cite_source": "atlas-3-spec.pdf",
        }
        assert compute_task_success(response, expected) is True

    def test_factual_wrong_answer(self):
        """Factual case with wrong answer fails."""
        response = {
            "status": "done",
            "answer": "The pressure is 200 MPa [e1].",
            "citations": [{"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1}],
        }
        expected = {
            "min_citations": 1,
            "must_contain_any": ["5 MPa"],
            "must_cite_source": "atlas-3-spec.pdf",
        }
        assert compute_task_success(response, expected) is False

    def test_factual_no_citations(self):
        """Factual case without enough citations fails."""
        response = {
            "status": "done",
            "answer": "The pressure is 5 MPa.",
            "citations": [],
        }
        expected = {"min_citations": 1, "must_contain_any": ["5 MPa"]}
        assert compute_task_success(response, expected) is False

    def test_factual_wrong_source(self):
        """Factual case citing wrong source fails."""
        response = {
            "status": "done",
            "answer": "The pressure is 5 MPa [e1].",
            "citations": [{"evidence_id": "e1", "source": "wrong-file.pdf", "page": 1}],
        }
        expected = {
            "min_citations": 1,
            "must_contain_any": ["5 MPa"],
            "must_cite_source": "atlas-3-spec.pdf",
        }
        assert compute_task_success(response, expected) is False

    def test_off_topic_refuse_success(self):
        """Off-topic case that refuses passes."""
        response = {"status": "refused", "reason": "not in corpus"}
        expected = {"should_refuse": True}
        assert compute_task_success(response, expected) is True

    def test_off_topic_no_refuse_fails(self):
        """Off-topic case that doesn't refuse fails."""
        response = {
            "status": "done",
            "answer": "Sourdough bread requires flour and water.",
            "citations": [],
        }
        expected = {"should_refuse": True}
        assert compute_task_success(response, expected) is False

    def test_multi_hop_success(self):
        """Multi-hop case with comparison and all entities passes."""
        response = {
            "status": "done",
            "answer": "The Atlas-3 has 5 MPa while the Atlas-5 has 8 MPa. "
                      "The Atlas-5 has higher pressure [e1] [e2].",
            "citations": [
                {"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "atlas-5-spec.pdf", "page": 1},
            ],
        }
        expected = {
            "min_citations": 2,
            "must_contain_all": ["Atlas-3", "Atlas-5"],
            "must_compare": True,
        }
        assert compute_task_success(response, expected) is True

    def test_multi_hop_no_comparison_fails(self):
        """Multi-hop case without comparison words fails."""
        response = {
            "status": "done",
            "answer": "The Atlas-3 has 5 MPa. The Atlas-5 has 8 MPa. [e1] [e2]",
            "citations": [
                {"evidence_id": "e1", "source": "atlas-3-spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "atlas-5-spec.pdf", "page": 1},
            ],
        }
        expected = {
            "min_citations": 2,
            "must_contain_all": ["Atlas-3", "Atlas-5"],
            "must_compare": True,
        }
        assert compute_task_success(response, expected) is False


class TestToolCorrectness:
    """Tests for the tool_correctness metric."""

    def test_factual_with_evidence(self):
        """Factual case with evidence passes."""
        response = {"status": "done", "evidence_count": 3, "steps_taken": 2}
        expected = {}
        assert compute_tool_correctness(response, expected) is True

    def test_factual_no_evidence(self):
        """Factual case without evidence fails."""
        response = {"status": "done", "evidence_count": 0, "steps_taken": 1}
        expected = {}
        assert compute_tool_correctness(response, expected) is False

    def test_off_topic_quick_refuse(self):
        """Off-topic case that refuses quickly passes."""
        response = {"status": "refused", "steps_taken": 1, "evidence_count": 0}
        expected = {"should_refuse": True}
        assert compute_tool_correctness(response, expected) is True

    def test_off_topic_slow_refuse(self):
        """Off-topic case that takes too long to refuse fails."""
        response = {"status": "refused", "steps_taken": 5, "evidence_count": 3}
        expected = {"should_refuse": True}
        assert compute_tool_correctness(response, expected) is False


class TestGroundingScore:
    """Tests for the grounding_score metric."""

    def test_grounded_answer(self):
        """Answer with citations gets high grounding score."""
        response = {
            "status": "done",
            "answer": "The pressure is 5 MPa [e1]. The temp is 80 C [e2].",
            "citations": [
                {"evidence_id": "e1", "source": "spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "spec.pdf", "page": 2},
            ],
            "evidence_count": 2,
        }
        score = compute_grounding_score(response)
        assert score > 0.5

    def test_no_citations(self):
        """Answer without citations gets 0 grounding."""
        response = {
            "status": "done",
            "answer": "The pressure is 5 MPa.",
            "citations": [],
            "evidence_count": 1,
        }
        assert compute_grounding_score(response) == 0.0

    def test_refused_answer(self):
        """Refused answer gets 0 grounding."""
        response = {"status": "refused", "answer": "", "citations": []}
        assert compute_grounding_score(response) == 0.0


class TestCitationQuality:
    """Tests for the citation_quality metric."""

    def test_valid_citations(self):
        """Citations with evidence_id and source get quality 1.0."""
        response = {
            "citations": [
                {"evidence_id": "e1", "source": "spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "manual.pdf", "page": 4},
            ]
        }
        assert compute_citation_quality(response) == 1.0

    def test_missing_source(self):
        """Citation without source lowers quality."""
        response = {
            "citations": [
                {"evidence_id": "e1", "source": "spec.pdf", "page": 1},
                {"evidence_id": "e2", "source": "", "page": 4},
            ]
        }
        assert compute_citation_quality(response) == 0.5

    def test_no_citations(self):
        """No citations gets 0 quality."""
        response = {"citations": []}
        assert compute_citation_quality(response) == 0.0


class TestComputeAllMetrics:
    """Tests for the combined metrics function."""

    def test_all_metrics_returned(self):
        """compute_all_metrics returns all 4 metrics."""
        response = {
            "status": "done",
            "answer": "The pressure is 5 MPa [e1].",
            "citations": [{"evidence_id": "e1", "source": "spec.pdf", "page": 1}],
            "evidence_count": 1,
            "steps_taken": 2,
        }
        expected = {"min_citations": 1, "must_contain_any": ["5 MPa"]}

        metrics = compute_all_metrics(response, expected)

        assert "task_success" in metrics
        assert "tool_correctness" in metrics
        assert "grounding_score" in metrics
        assert "citation_quality" in metrics
        assert metrics["task_success"] is True
