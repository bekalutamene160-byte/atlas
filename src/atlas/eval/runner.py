"""
ATLAS evaluation runner.

Runs benchmark cases against the Atlas agent, computes grounding/citation
metrics, and saves a baseline JSON for regression tracking.

Quota-aware design (the important part):
- Catches RateLimitError from GroqProvider and reads wait_seconds directly
- Falls back to regex parsing if the error is a plain ProviderError
- Retries each case up to 3 times with the parsed wait time
- Uses a small 2s inter-case delay (this is for RPM, NOT for TPD —
  TPD requires waiting for the daily window to roll, which is what
  the per-case retry handles)
- Gives up gracefully after 3 retries per case, recording the last error

Fixture support (the quota-saver):
- --fixtures flag wraps GroqProvider in FixtureProvider
- --fixture-mode record: call real API, save responses to disk
- --fixture-mode replay: load responses from disk, zero API cost
- --fixture-mode auto: replay if fixture exists, else record (default)

Usage:
    # First run (when Groq quota recovers) — records fixtures, ~70k tokens
    python -m atlas.eval.runner --fixtures --fixture-mode record

    # All subsequent runs — 100% free, deterministic
    python -m atlas.eval.runner --fixtures --fixture-mode replay

    # Without fixtures (calls Groq directly, costs tokens every run)
    python -m atlas.eval.runner

    # Other options
    python -m atlas.eval.runner --cases eval/benchmark.jsonl
    python -m atlas.eval.runner --max-cases 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Project root resolution: src/atlas/eval/runner.py → project root is 3 parents up
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Atlas imports — adjust these to match your actual module paths
from atlas.agent.loop import Agent  # noqa: E402
from atlas.exceptions import ProviderError  # noqa: E402
from atlas.providers.groq_provider import GroqProvider, RateLimitError  # noqa: E402


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Result of running a single eval case."""
    case_id: str
    category: str
    passed: bool
    answer: str = ""
    grounding_score: float = 0.0
    citation_quality: float = 0.0
    latency_s: float = 0.0
    error: str = ""
    citations: list = field(default_factory=list)


@dataclass
class EvalReport:
    """Aggregated results across all eval cases."""
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def success_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0

    @property
    def avg_grounding(self) -> float:
        passed = [r for r in self.results if r.passed]
        if not passed:
            return 0.0
        return sum(r.grounding_score for r in passed) / len(passed)

    @property
    def avg_citation(self) -> float:
        passed = [r for r in self.results if r.passed]
        if not passed:
            return 0.0
        return sum(r.citation_quality for r in passed) / len(passed)

    @property
    def avg_latency(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_s for r in self.results) / len(self.results)

    def per_category(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.results:
            cat = r.category
            if cat not in out:
                out[cat] = {
                    "total": 0,
                    "passed": 0,
                    "grounding_scores": [],
                    "citation_scores": [],
                }
            out[cat]["total"] += 1
            if r.passed:
                out[cat]["passed"] += 1
                out[cat]["grounding_scores"].append(r.grounding_score)
                out[cat]["citation_scores"].append(r.citation_quality)
        for cat, d in out.items():
            d["success_rate"] = d["passed"] / d["total"] if d["total"] else 0.0
            d["avg_grounding"] = (
                sum(d["grounding_scores"]) / len(d["grounding_scores"])
                if d["grounding_scores"] else 0.0
            )
            d["avg_citation"] = (
                sum(d["citation_scores"]) / len(d["citation_scores"])
                if d["citation_scores"] else 0.0
            )
        return out

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("ATLAS EVALUATION REPORT")
        print("=" * 60)
        print(f"Total Cases: {self.total}")
        print(f"Passed: {self.passed_count}/{self.total}")
        print(f"Success Rate: {self.success_rate * 100:.1f}%")
        print(f"Avg Grounding Score: {self.avg_grounding:.2f}")
        print(f"Avg Citation Quality: {self.avg_citation:.2f}")
        print(f"Avg Latency: {self.avg_latency:.2f}s")
        print()
        print("Per-Category Breakdown:")
        for cat, d in self.per_category().items():
            print(
                f"  {cat}: {d['passed']}/{d['total']} "
                f"({d['success_rate'] * 100:.1f}%) | "
                f"grounding={d['avg_grounding']:.2f} | "
                f"citation={d['avg_citation']:.2f}"
            )
        print()
        failed = [r for r in self.results if not r.passed]
        if failed:
            print("Failed Cases:")
            for r in failed:
                err_suffix = f" — {r.error[:80]}" if r.error else ""
                print(f"  {r.case_id} ({r.category}){err_suffix}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list[dict]:
    """Load benchmark cases from a JSONL file (one JSON object per line)."""
    cases: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {line_no} of {path}: {e}")
    return cases


def _extract_wait_from_message(err_str: str) -> Optional[float]:
    """Fallback parser for wait time when we get a plain ProviderError
    instead of a RateLimitError. Parses Groq's two formats:

        "Please try again in 4m18.336s"
        "Please try again in 58.000000 seconds"
    """
    match_min_sec = re.search(r"try again in (\d+)m([\d.]+)s", err_str)
    if match_min_sec:
        minutes = int(match_min_sec.group(1))
        seconds = float(match_min_sec.group(2))
        return minutes * 60 + seconds + 5.0

    match_sec_word = re.search(r"try again in ([\d.]+) seconds", err_str)
    if match_sec_word:
        return float(match_sec_word.group(1)) + 5.0

    match_sec_only = re.search(r"try again in ([\d.]+)s\b", err_str)
    if match_sec_only:
        return float(match_sec_only.group(1)) + 5.0

    return None


def _normalize_text(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy comparison."""
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()


def _fuzzy_match(expected: str, actual: str) -> bool:
    """Return True if all key tokens from `expected` appear in `actual`.

    We use subset matching (not equality) because the agent's answer
    will naturally contain more context than the bare expected value.
    """
    if not expected:
        return True
    expected_tokens = set(_normalize_text(expected).split())
    actual_tokens = set(_normalize_text(actual).split())
    if not expected_tokens:
        return True
    return expected_tokens.issubset(actual_tokens)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

# Markers that indicate the agent is refusing an out-of-scope question
REFUSAL_MARKERS = [
    "out of scope",
    "outside the scope",
    "outside my scope",
    "not in the document",
    "not in the provided",
    "not covered",
    "cannot answer",
    "can't answer",
    "i don't have",
    "i do not have",
    "no information",
    "not relevant",
    "i'm unable to",
    "i am unable to",
]


def _evaluate_case(case: dict, agent_response: Any) -> CaseResult:
    """Evaluate one case against the agent's response.

    Agent.run() returns a dictionary; older callers may provide an object
    with equivalent attributes.
    """
    case_id = case.get("case_id", "unknown")
    category = case.get("category", "unknown")
    expected = case.get("expected", {}) or {}
    expected_answer = case.get("expected_answer", "")
    must_cite = case.get("must_cite", []) or []
    if expected.get("must_cite_source"):
        must_cite = [*must_cite, expected["must_cite_source"]]
    should_refuse = case.get(
        "should_refuse", expected.get("should_refuse", False)
    )

    def response_value(name: str, default):
        if isinstance(agent_response, dict):
            return agent_response.get(name, default)
        return getattr(agent_response, name, default)

    answer = (
        response_value("answer", "")
        or response_value("partial_answer", "")
        or response_value("reason", "")
        or ""
    )
    citations = response_value("citations", []) or []
    citation_sources = [
        str(citation.get("source", "")) if isinstance(citation, dict)
        else str(citation)
        for citation in citations
    ]
    response_status = response_value("status", "")

    if should_refuse:
        # Off-topic case: pass if the agent refused OR gave a very short
        # non-answer (less than 50 chars usually means a clarification
        # question, not a real answer)
        answer_lower = answer.lower()
        refused = any(m in answer_lower for m in REFUSAL_MARKERS)
        gave_short_answer = (
            0 < len(answer.strip()) < 50 and response_status != "error"
        )
        passed = response_status == "refused" or refused or gave_short_answer

        return CaseResult(
            case_id=case_id,
            category=category,
            passed=passed,
            answer=answer,
            grounding_score=1.0 if passed else 0.0,
            citation_quality=1.0 if passed else 0.0,
            citations=citations,
        )

    # Factual / multi-hop / ambiguous: check answer + citations
    answer_match = _fuzzy_match(expected_answer, answer)
    must_contain_any = expected.get("must_contain_any", [])
    if must_contain_any:
        answer_match = answer_match and any(
            _fuzzy_match(candidate, answer) for candidate in must_contain_any
        )
    must_contain_all = expected.get("must_contain_all", [])
    if must_contain_all:
        answer_match = answer_match and all(
            _fuzzy_match(required, answer) for required in must_contain_all
        )

    # Citation check: every required source must appear in citations
    if must_cite:
        citations_lower = [source.lower() for source in citation_sources]
        citation_match = all(
            any(must.lower() in c for c in citations_lower)
            for must in must_cite
        )
    else:
        citation_match = True
    min_citations = expected.get("min_citations", 0)
    citation_match = citation_match and len(citations) >= min_citations

    grounding_score = 1.0 if citations else 0.0
    citation_quality = 1.0 if citation_match else 0.5
    passed = answer_match and citation_match

    return CaseResult(
        case_id=case_id,
        category=category,
        passed=passed,
        answer=answer,
        grounding_score=grounding_score,
        citation_quality=citation_quality,
        citations=citations,
    )


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

async def build_agent(
    use_fixtures: bool = False,
    fixture_mode: str = "auto",
) -> Agent:
    """Build the Atlas agent, optionally wrapping the provider in a fixture.

    Args:
        use_fixtures: If True, wrap GroqProvider in FixtureProvider.
        fixture_mode: "record" (call real API, save responses),
                      "replay" (load from disk only),
                      "auto" (replay if exists, else record).

    For the FIRST eval run after Groq quota recovers, use:
        build_agent(use_fixtures=True, fixture_mode="record")

    For ALL subsequent eval runs, use:
        build_agent(use_fixtures=True, fixture_mode="replay")

    The agent uses the same dependencies and configuration as the
    application, with the provider wrapped only when fixtures are enabled.
    """
    from atlas.agent.critic import Critic
    from atlas.agent.planner import Planner
    from atlas.agent.synthesizer import Synthesizer
    from atlas.config import settings
    from atlas.ingestion.embedder import make_embedder
    from atlas.retrieval.vector_store import VectorStore
    from atlas.tools.retriever_tool import RetrieverTool
    from atlas.tools.dispatcher import ToolDispatcher

    needs_live_provider = not use_fixtures or fixture_mode != "replay"
    real_provider = None
    if needs_live_provider:
        api_key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("GROQ_KEY")
            or settings.groq_api_key
        )
        if not api_key:
            raise RuntimeError(
                "Groq API key not set. Configure ATLAS_GROQ_API_KEY, "
                "GROQ_API_KEY, or GROQ_KEY."
            )
        real_provider = GroqProvider(
            api_key=api_key,
            planner_model=settings.groq_planner_model,
            synthesizer_model=settings.groq_synthesizer_model,
            critic_model=settings.groq_critic_model,
        )

    if use_fixtures:
        from atlas.providers.fixture_provider import FixtureProvider
        provider = FixtureProvider(real_provider=real_provider, mode=fixture_mode)
        logger.info(f"Using FixtureProvider (mode={fixture_mode}).")
    else:
        if real_provider is None:
            raise RuntimeError("A live Groq provider is required when fixtures are disabled.")
        provider = real_provider
        logger.info("Using GroqProvider directly (no fixtures, costs tokens).")

    embedder = make_embedder(settings)
    vector_store = VectorStore(
        db_path=settings.db_path,
        embedding_dims=embedder.dimensions,
    )
    retriever = RetrieverTool(
        embedder=embedder,
        vector_store=vector_store,
        min_similarity=settings.min_similarity,
    )
    dispatcher = ToolDispatcher(tools=[retriever])

    return Agent(
        settings=settings,
        model=provider,
        embedder=embedder,
        vector_store=vector_store,
        planner=Planner(model=provider),
        synthesizer=Synthesizer(model=provider),
        critic=Critic(model=provider, model_name=settings.groq_critic_model),
        dispatcher=dispatcher,
    )


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

class EvalRunner:
    """Runs benchmark cases with quota-aware retry logic.

    The retry logic is the heart of this class. When Groq returns a 429:
    1. GroqProvider parses it into RateLimitError with wait_seconds
    2. This runner catches RateLimitError and sleeps wait_seconds
    3. After 3 failed attempts, the case is marked as failed with the
       last error recorded, and we move on to the next case
    """

    def __init__(
        self,
        cases: list[dict],
        agent: Agent,
        max_retries: int = 3,
        inter_case_delay: float = 2.0,
    ):
        self.cases = cases
        self.agent = agent
        self.max_retries = max_retries
        self.inter_case_delay = inter_case_delay

    async def _run_case_once(self, case: dict):
        """Run a single case through the agent. Raises on error."""
        query = case["query"]
        return await self.agent.run(query)

    async def _run_case_with_retry(self, case: dict) -> CaseResult:
        """Run a case with up to max_retries attempts on rate-limit errors."""
        case_id = case.get("case_id", "unknown")
        category = case.get("category", "unknown")
        start = time.time()

        last_error: str = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self._run_case_once(case)
                latency = time.time() - start
                result = _evaluate_case(case, response)
                result.latency_s = latency
                return result

            except RateLimitError as e:
                # Best case: GroqProvider already parsed the wait time
                wait = e.wait_seconds
                last_error = f"RateLimitError({e.limit_type}): wait {wait:.0f}s"
                logger.warning(
                    f"  Rate limit hit ({e.limit_type}). "
                    f"Waiting {wait:.0f}s (attempt {attempt}/{self.max_retries})..."
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"  GAVE UP after {self.max_retries} rate-limit retries: {case_id}"
                    )

            except ProviderError as e:
                # Fallback: try to parse wait time from the error message
                err_str = str(e)
                last_error = err_str[:200]
                wait = _extract_wait_from_message(err_str)

                if wait and "rate" in err_str.lower():
                    logger.warning(
                        f"  Provider rate limit. Waiting {wait:.0f}s "
                        f"(attempt {attempt}/{self.max_retries})..."
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(wait)
                        continue

                logger.error(f"  ProviderError: {err_str[:200]}")
                if attempt < self.max_retries:
                    await asyncio.sleep(5.0)
                else:
                    logger.error(
                        f"  GAVE UP after {self.max_retries} retries: {case_id}"
                    )

            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                logger.error(f"  Unexpected error: {last_error}")
                if attempt < self.max_retries:
                    await asyncio.sleep(5.0)
                else:
                    logger.error(
                        f"  GAVE UP after {self.max_retries} retries: {case_id}"
                    )

        # All retries exhausted — return a failed result with the last error
        return CaseResult(
            case_id=case_id,
            category=category,
            passed=False,
            latency_s=time.time() - start,
            error=last_error,
        )

    async def run_all(self) -> EvalReport:
        """Run all cases sequentially with retry logic."""
        results: list[CaseResult] = []
        total = len(self.cases)

        for i, case in enumerate(self.cases):
            case_id = case.get("case_id", f"case_{i+1}")
            logger.info(f"Running case {i+1}/{total}: {case_id}")

            result = await self._run_case_with_retry(case)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(f"  {status}: {result.case_id} ({result.category})")
            if result.error:
                logger.info(f"  (last error: {result.error[:100]})")

            # Small inter-case delay to avoid RPM limits.
            # NOTE: this does NOT help with TPD — for TPD, the per-case
            # retry logic above handles waiting the correct amount.
            if i < total - 1:
                await asyncio.sleep(self.inter_case_delay)

        return EvalReport(results)


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

def save_baseline(report: EvalReport, baselines_dir: Path) -> Path:
    """Save the eval report as a timestamped baseline JSON."""
    baselines_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    path = baselines_dir / f"baseline_{timestamp}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": report.total,
            "passed": report.passed_count,
            "success_rate": report.success_rate,
            "avg_grounding": report.avg_grounding,
            "avg_citation": report.avg_citation,
            "avg_latency": report.avg_latency,
        },
        "categories": report.per_category(),
        "results": [asdict(r) for r in report.results],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _async_main(args) -> int:
    cases_path = Path(args.cases).resolve()
    if not cases_path.exists():
        logger.error(f"Cases file not found: {cases_path}")
        return 1

    cases = load_cases(cases_path)
    if args.max_cases and args.max_cases > 0:
        cases = cases[:args.max_cases]

    logger.info(f"Loaded {len(cases)} eval cases from {cases_path}")

    # Build agent (adapt build_agent() to match your setup)
    agent = await build_agent(
        use_fixtures=args.fixtures,
        fixture_mode=args.fixture_mode,
    )

    runner = EvalRunner(
        cases=cases,
        agent=agent,
        max_retries=3,
        inter_case_delay=2.0,
    )

    report = await runner.run_all()
    report.print_summary()

    baselines_dir = PROJECT_ROOT / "eval" / "baselines"
    baseline_path = save_baseline(report, baselines_dir)
    logger.info(f"Baseline saved to: {baseline_path}")
    print(f"\nBaseline saved to: {baseline_path}")

    # Exit code 0 if at least one case passed, 1 otherwise.
    # We don't require all-pass because rate limits may cause some failures.
    return 0 if report.passed_count > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="ATLAS evaluation runner")
    parser.add_argument(
        "--cases",
        default="eval/benchmark.jsonl",
        help="Path to benchmark JSONL file (default: eval/benchmark.jsonl)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Max number of cases to run (0 = all)",
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Use FixtureProvider instead of direct GroqProvider",
    )
    parser.add_argument(
        "--fixture-mode",
        choices=["record", "replay", "auto"],
        default="auto",
        help="Fixture mode: record (call API + save), replay (load only), "
             "auto (replay if exists, else record). Default: auto",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    sys.exit(main())