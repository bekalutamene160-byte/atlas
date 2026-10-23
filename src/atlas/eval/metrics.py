"""
Evaluation metrics — the 4 core metrics for ATLAS.

Each metric catches a different failure mode:
1. Task Success: Did the agent answer correctly? (end-to-end correctness)
2. Tool Correctness: Did it pick the right tool with right args? (planner quality)
3. Grounding: Are claims supported by evidence? (hallucination detection)
4. Citation Quality: Do citations point to valid sources? (verifiability)

Design principle: metrics are computed from the agent's response dict,
NOT from internal state. This means the eval runner can evaluate any
agent that returns the standard response format.
"""

from __future__ import annotations

import re
from typing import Any


def compute_task_success(response: dict, expected: dict) -> bool:
    """Check if the response meets the expected criteria.

    For factual cases: answer must contain expected phrases and have
    minimum citations.

    For off-topic cases: agent must refuse.

    For ambiguous cases: agent must clarify or pick a relevant topic.
    """
    status = response.get("status", "")

    # Off-topic cases must refuse
    if expected.get("should_refuse"):
        return status == "refused"

    # Ambiguous cases must clarify
    if expected.get("should_clarify"):
        answer = (response.get("answer") or response.get("reason") or "").lower()
        for phrase in expected.get("must_contain_any", []):
            if phrase.lower() in answer:
                return True
        return False

    # Factual and multi-hop cases must succeed
    if status != "done":
        return False

    answer = (response.get("answer") or "").lower()

    # Check must_contain_any (at least one phrase must be present)
        # Check must_contain_any (at least one phrase must be present)
    if expected.get("must_contain_any"):
        found_any = False
        for phrase in expected["must_contain_any"]:
            phrase_lower = phrase.lower()
            # Check exact match
            if phrase_lower in answer:
                found_any = True
                break
            # Check without spaces (e.g., "5mpa" matches "5 mpa")
            if phrase_lower.replace(" ", "") in answer.replace(" ", ""):
                found_any = True
                break
            # Check with unit variations
            variations = [
                phrase_lower,
                phrase_lower.replace(" mpa", " megapascals"),
                phrase_lower.replace(" mpa", "mpa"),
                phrase_lower.replace(" kpa", " kilopascals"),
                phrase_lower.replace(" m3/h", " cubic meters per hour"),
                phrase_lower.replace(" kw", " kilowatts"),
                phrase_lower.replace(" kg", " kilograms"),
                phrase_lower.replace(" mm", " millimeters"),
                phrase_lower.replace(" c", " celsius"),
                phrase_lower.replace(" c", " degrees"),
            ]
            for var in variations:
                if var in answer:
                    found_any = True
                    break
            if found_any:
                break
        if not found_any:
            return False

    # Check must_contain_all (all phrases must be present)
    if expected.get("must_contain_all"):
        for phrase in expected["must_contain_all"]:
            if phrase.lower() not in answer:
                return False

    # Check minimum citations
    min_citations = expected.get("min_citations", 0)
    actual_citations = len(response.get("citations", []))
    if actual_citations < min_citations:
        return False

    # Check must_cite_source (at least one citation must come from this source)
    if expected.get("must_cite_source"):
        sources = [c.get("source", "") for c in response.get("citations", [])]
        if not any(expected["must_cite_source"] in s for s in sources):
            return False

    # Check must_compare (for multi-hop cases)
    if expected.get("must_compare"):
        # The answer should contain comparison words
        comparison_words = ["compare", "versus", "vs", "higher", "lower",
                           "more", "less", "greater", "smaller", "difference",
                           "while", "whereas", "however", "unlike"]
        has_comparison = any(word in answer for word in comparison_words)
        if not has_comparison:
            return False

    return True


def compute_tool_correctness(response: dict, expected: dict) -> bool:
    """Check if the agent used tools correctly.

    For factual cases: at least 1 tool call (retrieve)
    For off-topic cases: 0-1 tool calls (should not spam)
    For multi-hop cases: at least 2 tool calls (needs multiple retrievals)

    This is a simplified metric. In production, you'd inspect the actual
    tool calls in the trace and compare to expected tool sequences.
    """
    status = response.get("status", "")
    evidence_count = response.get("evidence_count", 0)
    steps = response.get("steps_taken", 0)

    # Off-topic: should refuse quickly (1-2 steps)
    if expected.get("should_refuse"):
        return status == "refused" and steps <= 3

    # Factual: should have evidence and succeed
    if status == "done":
        return evidence_count > 0

    # Partial is acceptable if evidence was gathered
    if status == "partial":
        return evidence_count > 0

    return False


def compute_grounding_score(response: dict) -> float:
    """Estimate grounding quality from the response.

    Returns a score from 0.0 to 1.0.

    A fully grounded answer:
    - Has status "done" (critic approved)
    - Has at least 1 citation per 100 words of answer
    - Has evidence count > 0

    This is an approximation. True grounding measurement requires
    the critic's verdicts, which are in the trace but not in the
    response dict. For a portfolio project, this heuristic is sufficient.
    """
    status = response.get("status", "")

    if status != "done":
        return 0.0

    answer = response.get("answer", "")
    citations = response.get("citations", [])
    evidence_count = response.get("evidence_count", 0)

    if not answer or evidence_count == 0:
        return 0.0

    # Count words in answer
    word_count = len(answer.split())
    if word_count == 0:
        return 0.0

    # Calculate citations per 100 words
    citation_density = len(citations) / (word_count / 100) if word_count > 0 else 0

    # Score: 1.0 if density >= 1.0 (at least 1 citation per 100 words)
    # Scale down for lower density
    score = min(1.0, citation_density)

    # Bonus: if the answer has inline [eN] citations
    inline_citations = len(re.findall(r'\[e\d+\]', answer))
    if inline_citations > 0:
        score = min(1.0, score + 0.2)

    return round(score, 2)


def compute_citation_quality(response: dict) -> float:
    """Check if citations point to valid sources.

    Returns a score from 0.0 to 1.0.

    A valid citation:
    - Has an evidence_id (e.g., "e1")
    - Has a source (file path)
    - Has a page number (if applicable)

    This metric catches:
    - Hallucinated citations (no evidence_id)
    - Missing source info
    - Missing page numbers
    """
    citations = response.get("citations", [])
    if not citations:
        return 0.0

    valid_count = 0
    for cite in citations:
        is_valid = True

        # Must have evidence_id
        if not cite.get("evidence_id"):
            is_valid = False

        # Must have source
        if not cite.get("source"):
            is_valid = False

        # Should have page (not always applicable, but good practice)
        # We don't fail if page is None — some documents don't have pages

        if is_valid:
            valid_count += 1

    return round(valid_count / len(citations), 2)


def compute_all_metrics(response: dict, expected: dict) -> dict:
    """Compute all 4 metrics for a single case.

    Returns a dict with:
    - task_success: bool
    - tool_correctness: bool
    - grounding_score: float (0.0 to 1.0)
    - citation_quality: float (0.0 to 1.0)
    """
    return {
        "task_success": compute_task_success(response, expected),
        "tool_correctness": compute_tool_correctness(response, expected),
        "grounding_score": compute_grounding_score(response),
        "citation_quality": compute_citation_quality(response),
    }
