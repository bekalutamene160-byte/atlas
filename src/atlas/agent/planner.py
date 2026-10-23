"""
Planner — the agent's decision-maker.

The planner looks at the current state and decides the next action:
- "retrieve": search the corpus for more evidence (requires a query)
- "synthesize": produce the final answer (Day 4)
- "refuse": decline to answer (evidence is insufficient)

The planner uses the LLM's structured output (JSON mode) to emit a
PlanStep. The PlanStep is validated by Pydantic — if the LLM emits
invalid JSON or a semantically inconsistent decision, the provider retries.

Bug fix (Day 3 audit): the planner now returns TokenUsage alongside
the PlanStep, so the agent loop can track token consumption and enforce
the token budget. Previously, tokens were never counted, so the budget
bound never triggered.
"""

from __future__ import annotations

import logging
from typing import Optional

from atlas.models import AgentState, PlanStep, TokenUsage
from atlas.providers.base import ModelProvider, ModelResponse

logger = logging.getLogger(__name__)


class PlannerResult:
    """Wraps a planner decision with its token usage.

    This lets the agent loop track tokens without the planner needing
    to know about budgets. The planner just returns what it got from
    the provider; the loop decides what to do with it.
    """
    def __init__(self, decision: PlanStep, usage: TokenUsage):
        self.decision = decision
        self.usage = usage


class Planner:
    """Decides the next action for the agent.

    Uses the LLM to emit a PlanStep (retrieve, synthesize, or refuse).
    """

    SYSTEM_PROMPT = """You are ATLAS, a research planning agent for an industrial maintenance documentation corpus.

Your corpus contains:
- Equipment specification sheets (pumps, motors, valves, compressors)
- Maintenance manuals with procedures
- Safety protocols (lockout-tagout, PPE)
- Inspection standards

Your job is to decide the next action based on the current state.

Available actions:
- "retrieve": Search the corpus for more evidence. Requires a "query" field.
- "synthesize": Produce the final cited answer. Use this when you have ANY relevant evidence.
- "refuse": Decline to answer.

CRITICAL RULES:
1. After 1-2 successful retrievals, you MUST synthesize. Do not retrieve more than 2 times.
2. If you have 5+ evidence chunks, you have enough. SYNTHESIZE immediately.
3. If you have retrieved the same query twice, SYNTHESIZE with what you have.
4. If the evidence sources include spec sheets or manuals, they likely contain the answer. SYNTHESIZE.
5. Do NOT say "need more specific evidence" if you already have spec sheets. SYNTHESIZE.
6. Only refuse if the query is clearly off-topic (baking, sports, programming, geography).
7. For "refuse", explain what the user asked and why the corpus doesn't have it.

Return JSON: {"action": "retrieve" | "synthesize" | "refuse", "query": "...", "reasoning": "..."}
For "synthesize" and "refuse", "query" must be null."""

    def __init__(self, model: ModelProvider, model_name: Optional[str] = None):
        """Initialize the planner.

        Args:
            model: The LLM provider (GroqProvider or MockProvider)
            model_name: Override the model (e.g., "llama-3.3-70b-versatile").
                        If None, uses the provider's default.
        """
        self.model = model
        self.model_name = model_name

    async def decide(self, state: AgentState) -> PlannerResult:
        """Decide the next action based on current state.

        Bug fix: now returns PlannerResult (decision + usage) instead of
        just PlanStep. This lets the agent loop track token consumption.

        Args:
            state: The current agent state (query, evidence, steps, failures)

        Returns:
            PlannerResult containing the PlanStep and TokenUsage

        Raises:
            ProviderError: if the LLM call fails after retries
        """
        user_msg = self._build_user_message(state)

        response: ModelResponse = await self.model.generate_structured(
            response_model=PlanStep,
            system=self.SYSTEM_PROMPT,
            user=user_msg,
            model=self.model_name,
        )

        logger.info(
            f"planner: decision={response.data.action}, "
            f"reasoning='{response.data.reasoning[:80]}...', "
            f"tokens={response.usage.total_tokens}"
        )

        return PlannerResult(decision=response.data, usage=response.usage)

    def _build_user_message(self, state: AgentState) -> str:
        """Build the user message from state.

        Includes:
        - The original query
        - How many evidence chunks have been gathered
        - How many steps have been taken
        - Recent failures (so the planner can learn from them)

        Does NOT include the actual evidence text — the planner only
        needs to know HOW MUCH evidence exists, not what it says.
        This is a security boundary: it prevents prompt injection from
        retrieved chunks from affecting the planner's decisions.
        """
        lines = [
            f"Query: {state.query}",
            f"Evidence gathered: {len(state.retrieved_evidence)} chunks",
            f"Steps taken: {state.steps_taken}",
        ]

        if state.failures:
            recent = state.failures[-3:]
            lines.append(f"Recent failures: {recent}")

        if state.retrieved_evidence:
            evidence_summary = [
                f"  - {e.evidence_id}: {e.chunk.source} "
                f"(p.{e.chunk.page}, score={e.chunk.score:.2f})"
                for e in state.retrieved_evidence[:5]
            ]
            lines.append("Evidence available:")
            lines.extend(evidence_summary)

        return "\n".join(lines)
