"""
Agent loop — the heart of ATLAS.

The canonical observe-decide-act cycle with 5 bounds:
1. max_steps: maximum loop iterations
2. timeout: wall-clock time limit (enforced via asyncio.wait_for)
3. token_budget: maximum tokens consumed (tracked from planner + synthesizer + critic)
4. tool_budget: maximum tool calls
5. stuck_detection: abort if no state change for N steps

Day 4 update: the synthesize action is now fully implemented.
When the planner emits "synthesize":
1. Call the synthesizer → produce cited draft answer
2. Call the critic → verify every claim against evidence
3. If all supported → return SuccessResponse
4. If some unsupported → add failures to state, loop back to planning

The loop NEVER crashes. Every failure mode produces a structured response.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import AsyncIterator, Optional

from atlas.config import Settings, settings as default_settings
from atlas.exceptions import ProviderError
from atlas.models import (
    AgentState,
    Citation,
    Evidence,
    PartialResponse,
    PlanStep,
    RefusalResponse,
    RetrievedChunk,
    SuccessResponse,
    ToolCall,
    TokenUsage,
)
from atlas.providers.base import ModelProvider, make_provider
from atlas.agent.planner import Planner, PlannerResult
from atlas.agent.synthesizer import Synthesizer, SynthesizerResult
from atlas.agent.critic import Critic, CriticResult
from atlas.ingestion.embedder import make_embedder
from atlas.retrieval.vector_store import VectorStore
from atlas.tools.dispatcher import ToolDispatcher
from atlas.tools.retriever_tool import RetrieverTool

logger = logging.getLogger(__name__)


class Agent:
    """The ATLAS agent — a bounded loop that retrieves, synthesizes, and verifies."""

    def __init__(
        self,
        settings: Settings,
        model: ModelProvider,
        embedder,
        vector_store: VectorStore,
        planner: Planner,
        synthesizer: Synthesizer,
        critic: Critic,
        dispatcher: ToolDispatcher,
    ):
        self.settings = settings
        self.model = model
        self.embedder = embedder
        self.vector_store = vector_store
        self.planner = planner
        self.synthesizer = synthesizer
        self.critic = critic
        self.dispatcher = dispatcher
        self._total_tokens: int = 0

    @classmethod
    def from_default(cls, settings: Optional[Settings] = None) -> "Agent":
        """Create an agent from settings, with all dependencies wired."""
        settings = settings or default_settings
        model = make_provider(settings)
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
        planner = Planner(model=model)

        # Day 4: synthesizer uses the synthesizer model (70B)
        synthesizer = Synthesizer(model=model)

        # Day 4: critic uses the critic model (8B for speed)
        # We pass the model name explicitly — the GroqProvider uses it
        critic = Critic(model=model, model_name=settings.groq_critic_model)

        return cls(
            settings=settings,
            model=model,
            embedder=embedder,
            vector_store=vector_store,
            planner=planner,
            synthesizer=synthesizer,
            critic=critic,
            dispatcher=dispatcher,
        )

    async def run(self, query: str) -> dict:
        """Run the agent on a query.

        Returns a dict with: run_id, query, status, steps_taken,
        evidence_count, cost_tokens.

        Status is one of: "done", "refused", "partial", "error".
        """
        state = AgentState(query=query)
        start_time = time.monotonic()
        self._total_tokens = 0
        last_state_hash: Optional[str] = None
        stuck_count = 0

        logger.info(f"agent.run: query='{query[:80]}...', run_id={state.run_id}")

        try:
            for step in range(self.settings.max_agent_steps):
                # ===== BOUND CHECKS =====
                elapsed = time.monotonic() - start_time
                if elapsed > self.settings.run_timeout_seconds:
                    return self._partial_response(state, "timeout")

                if self._total_tokens > self.settings.max_tokens_per_run:
                    return self._partial_response(state, "budget_exhausted")

                if state.tool_call_count >= self.settings.max_tool_calls:
                    return self._partial_response(state, "tool_budget_exhausted")

                current_hash = self._state_hash(state)
                if current_hash == last_state_hash:
                    stuck_count += 1
                    if stuck_count >= self.settings.stuck_window:
                        return self._partial_response(state, "stuck")
                else:
                    stuck_count = 0
                last_state_hash = current_hash

                # ===== OBSERVE + DECIDE =====
                state.steps_taken = step + 1
                state.status = "planning"

                remaining_time = self.settings.run_timeout_seconds - elapsed
                if remaining_time <= 0:
                    return self._partial_response(state, "timeout")

                try:
                    planner_result: PlannerResult = await asyncio.wait_for(
                        self.planner.decide(state),
                        timeout=remaining_time,
                    )
                except asyncio.TimeoutError:
                    return self._partial_response(state, "timeout")
                except ProviderError as e:
                    return self._error_response(state, "provider_unavailable", str(e))

                self._total_tokens += planner_result.usage.total_tokens
                decision: PlanStep = planner_result.decision

                if self._total_tokens > self.settings.max_tokens_per_run:
                    return self._partial_response(state, "budget_exhausted")

                # ===== EXECUTE BASED ON DECISION =====

                if decision.action == "refuse":
                    state.status = "refused"
                    return self._refusal_response(state, decision.reasoning)

                elif decision.action == "retrieve":
                    state.status = "retrieving"
                    await self._execute_retrieve(state, decision, start_time)

                elif decision.action == "synthesize":
                    state.status = "synthesizing"
                    result = await self._execute_synthesize(state, start_time)
                    if result is not None:
                        return result  # either SuccessResponse or PartialResponse

                else:
                    return self._error_response(
                        state, "internal_error", f"unknown action: {decision.action}"
                    )

            return self._partial_response(state, "max_steps")

        except Exception as e:
            logger.error(f"agent: unhandled exception: {e}", exc_info=True)
            return self._error_response(state, "internal_error", str(e))

    async def _execute_retrieve(
        self, state: AgentState, decision: PlanStep, start_time: float
    ) -> None:
        """Execute a retrieve action."""
        tool_call = ToolCall(
            name="retrieve",
            arguments={
                "query": decision.query,
                "top_k": self.settings.retrieval_top_k,
            },
            reasoning=decision.reasoning,
        )
        state.add_tool_call(tool_call)

        remaining_time = self.settings.run_timeout_seconds - (time.monotonic() - start_time)
        if remaining_time <= 0:
            state.add_failure("retrieve timed out before execution")
            return

        try:
            result = await asyncio.wait_for(
                self.dispatcher.dispatch(tool_call),
                timeout=remaining_time,
            )
        except asyncio.TimeoutError:
            state.add_failure("retrieve timed out")
            return

        if not result.success:
            state.add_failure(f"retrieve failed: {result.error}")
            return

        chunks_data = result.data.get("chunks", [])
        for chunk_dict in chunks_data:
            chunk = RetrievedChunk(**chunk_dict)
            evidence_id = f"e{len(state.retrieved_evidence) + 1}"
            evidence = Evidence(
                evidence_id=evidence_id,
                chunk=chunk,
                query_used=decision.query or "",
            )
            state.add_evidence(evidence)

        logger.info(
            f"agent: retrieved {len(chunks_data)} chunks "
            f"(total evidence: {len(state.retrieved_evidence)})"
        )

    async def _execute_synthesize(
        self, state: AgentState, start_time: float
    ) -> Optional[dict]:
        """Execute synthesize + critic. Returns a response dict or None.

        If the critic approves → return SuccessResponse.
        If the critic rejects → add failures to state, return None (loop continues).
        If an error occurs → return PartialResponse or ErrorResponse.
        """
        remaining_time = self.settings.run_timeout_seconds - (time.monotonic() - start_time)
        if remaining_time <= 0:
            return self._partial_response(state, "timeout")

        # Step 1: Synthesize
        try:
            synth_result: SynthesizerResult = await asyncio.wait_for(
                self.synthesizer.synthesize(state),
                timeout=remaining_time,
            )
        except asyncio.TimeoutError:
            return self._partial_response(state, "timeout")
        except ProviderError as e:
            return self._error_response(state, "provider_unavailable", str(e))

        self._total_tokens += synth_result.usage.total_tokens

        if self._total_tokens > self.settings.max_tokens_per_run:
            return self._partial_response(state, "budget_exhausted")

        # Step 2: Critique
        remaining_time = self.settings.run_timeout_seconds - (time.monotonic() - start_time)
        if remaining_time <= 0:
            return self._partial_response(state, "timeout")

        try:
            critic_result: CriticResult = await asyncio.wait_for(
                self.critic.verify(synth_result.answer, state),
                timeout=remaining_time,
            )
        except asyncio.TimeoutError:
            return self._partial_response(state, "timeout")
        except ProviderError as e:
            return self._error_response(state, "provider_unavailable", str(e))

        self._total_tokens += critic_result.usage.total_tokens

        # Step 3: Check verdict
        state.status = "criticizing"

        if critic_result.report.all_supported:
            # All claims verified — return success!
            state.status = "done"
            citations = self._build_citations(synth_result.answer, state)
            logger.info("agent: synthesis approved by critic")
            return self._success_response(state, synth_result.answer, citations)
        else:
            # Some claims unsupported — add failures and loop back
            failure_msgs = [
                f"claim '{v.claim_text[:50]}...' {v.issue}"
                for v in critic_result.report.issues
            ]
            for msg in failure_msgs:
                state.add_failure(msg)
            logger.info(
                f"agent: synthesis rejected ({len(critic_result.report.issues)} issues), "
                f"looping back to planning"
            )
            return None  # loop continues — planner will see the failures

    def _build_citations(self, answer: str, state: AgentState) -> list[Citation]:
        """Build the citations list for the success response.

        Parses [eN] citations from the answer and maps them to evidence.
        """
        from atlas.agent.critic import Critic as CriticClass
        cited_ids = CriticClass.extract_citations(answer)
        evidence_by_id = {e.evidence_id: e for e in state.retrieved_evidence}

        citations: list[Citation] = []
        seen: set[str] = set()
        for cid in cited_ids:
            if cid in evidence_by_id and cid not in seen:
                ev = evidence_by_id[cid]
                citations.append(Citation(
                    evidence_id=cid,
                    source=ev.chunk.source,
                    page=ev.chunk.page,
                    section=ev.chunk.section,
                ))
                seen.add(cid)
        return citations

    def _state_hash(self, state: AgentState) -> str:
        evidence_ids = tuple(e.evidence_id for e in state.retrieved_evidence)
        return hashlib.md5(
            f"{state.status}:{evidence_ids}:{len(state.failures)}".encode()
        ).hexdigest()

    def _success_response(self, state: AgentState, answer: str,
                          citations: list[Citation]) -> dict:
        return SuccessResponse(
            run_id=state.run_id,
            query=state.query,
            status="done",
            steps_taken=state.steps_taken,
            evidence_count=len(state.retrieved_evidence),
            cost_tokens=self._total_tokens,
            answer=answer,
            citations=citations,
        ).model_dump()

    def _refusal_response(self, state: AgentState, reason: str) -> dict:
        return RefusalResponse(
            run_id=state.run_id,
            query=state.query,
            status="refused",
            steps_taken=state.steps_taken,
            evidence_count=len(state.retrieved_evidence),
            cost_tokens=self._total_tokens,
            reason=reason,
        ).model_dump()

    def _partial_response(self, state: AgentState, reason: str) -> dict:
        return PartialResponse(
            run_id=state.run_id,
            query=state.query,
            status="partial",
            steps_taken=state.steps_taken,
            evidence_count=len(state.retrieved_evidence),
            cost_tokens=self._total_tokens,
            reason=reason,
            partial_answer=None,
        ).model_dump()

    def _error_response(
        self, state: AgentState, error_code: str, error_message: str
    ) -> dict:
        return {
            "run_id": state.run_id,
            "query": state.query,
            "status": "error",
            "error_code": error_code,
            "error_message": error_message,
            "steps_taken": state.steps_taken,
            "evidence_count": len(state.retrieved_evidence),
            "cost_tokens": self._total_tokens,
        }

    async def run_streaming(self, query: str) -> AsyncIterator[dict]:
        """Run the agent, yielding events as they happen.

        This is the streaming version of run(). It yields events that
        the SSE endpoint forwards to the browser.

        Events:
        - {"type": "start", "run_id": "...", "query": "..."}
        - {"type": "step", "step": 1, "status": "planning"}
        - {"type": "step", "step": 1, "status": "retrieving"}
        - {"type": "evidence", "count": 5}
        - {"type": "step", "step": 2, "status": "synthesizing"}
        - {"type": "step", "step": 2, "status": "criticizing"}
        - {"type": "done", "result": {...}}
        - {"type": "refused", "result": {...}}
        - {"type": "partial", "result": {...}}
        - {"type": "error", "result": {...}}

        The existing run() method is UNCHANGED. This is a new method
        that duplicates the loop logic but yields events. This is
        intentional — keeping run() unchanged means existing tests
        don't break.
        """
        state = AgentState(query=query)
        start_time = time.monotonic()
        self._total_tokens = 0
        last_state_hash: Optional[str] = None
        stuck_count = 0

        yield {"type": "start", "run_id": state.run_id, "query": query}

        try:
            for step in range(self.settings.max_agent_steps):
                # ===== BOUND CHECKS =====
                elapsed = time.monotonic() - start_time
                if elapsed > self.settings.run_timeout_seconds:
                    yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                    return

                if self._total_tokens > self.settings.max_tokens_per_run:
                    yield {"type": "partial", "result": self._partial_response(state, "budget_exhausted")}
                    return

                if state.tool_call_count >= self.settings.max_tool_calls:
                    yield {"type": "partial", "result": self._partial_response(state, "tool_budget_exhausted")}
                    return

                current_hash = self._state_hash(state)
                if current_hash == last_state_hash:
                    stuck_count += 1
                    if stuck_count >= self.settings.stuck_window:
                        yield {"type": "partial", "result": self._partial_response(state, "stuck")}
                        return
                else:
                    stuck_count = 0
                last_state_hash = current_hash

                # ===== OBSERVE + DECIDE =====
                state.steps_taken = step + 1
                state.status = "planning"
                yield {"type": "step", "step": state.steps_taken, "status": "planning"}

                remaining_time = self.settings.run_timeout_seconds - elapsed
                if remaining_time <= 0:
                    yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                    return

                try:
                    planner_result = await asyncio.wait_for(
                        self.planner.decide(state),
                        timeout=remaining_time,
                    )
                except asyncio.TimeoutError:
                    yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                    return
                except ProviderError as e:
                    yield {"type": "error", "result": self._error_response(state, "provider_unavailable", str(e))}
                    return

                self._total_tokens += planner_result.usage.total_tokens
                decision = planner_result.decision

                if self._total_tokens > self.settings.max_tokens_per_run:
                    yield {"type": "partial", "result": self._partial_response(state, "budget_exhausted")}
                    return

                # ===== EXECUTE =====
                if decision.action == "refuse":
                    state.status = "refused"
                    yield {"type": "refused", "result": self._refusal_response(state, decision.reasoning)}
                    return

                elif decision.action == "retrieve":
                    state.status = "retrieving"
                    yield {"type": "step", "step": state.steps_taken, "status": "retrieving"}
                    await self._execute_retrieve(state, decision, start_time)
                    yield {"type": "evidence", "count": len(state.retrieved_evidence)}

                elif decision.action == "synthesize":
                    state.status = "synthesizing"
                    yield {"type": "step", "step": state.steps_taken, "status": "synthesizing"}

                    # Synthesize
                    remaining_time = self.settings.run_timeout_seconds - (time.monotonic() - start_time)
                    if remaining_time <= 0:
                        yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                        return

                    try:
                        synth_result = await asyncio.wait_for(
                            self.synthesizer.synthesize(state),
                            timeout=remaining_time,
                        )
                    except asyncio.TimeoutError:
                        yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                        return
                    except ProviderError as e:
                        yield {"type": "error", "result": self._error_response(state, "provider_unavailable", str(e))}
                        return

                    self._total_tokens += synth_result.usage.total_tokens

                    if self._total_tokens > self.settings.max_tokens_per_run:
                        yield {"type": "partial", "result": self._partial_response(state, "budget_exhausted")}
                        return

                    # Critique
                    state.status = "criticizing"
                    yield {"type": "step", "step": state.steps_taken, "status": "criticizing"}

                    remaining_time = self.settings.run_timeout_seconds - (time.monotonic() - start_time)
                    if remaining_time <= 0:
                        yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                        return

                    try:
                        critic_result = await asyncio.wait_for(
                            self.critic.verify(synth_result.answer, state),
                            timeout=remaining_time,
                        )
                    except asyncio.TimeoutError:
                        yield {"type": "partial", "result": self._partial_response(state, "timeout")}
                        return
                    except ProviderError as e:
                        yield {"type": "error", "result": self._error_response(state, "provider_unavailable", str(e))}
                        return

                    self._total_tokens += critic_result.usage.total_tokens

                    # Check verdict
                    if critic_result.report.all_supported:
                        state.status = "done"
                        citations = self._build_citations(synth_result.answer, state)
                        result = self._success_response(state, synth_result.answer, citations)
                        yield {"type": "done", "result": result}
                        return
                    else:
                        # Rejected — add failures and continue
                        for v in critic_result.report.issues:
                            state.add_failure(f"claim '{v.claim_text[:50]}...' {v.issue}")
                        yield {"type": "step", "step": state.steps_taken, "status": "critic_rejected"}
                        # Loop continues — planner will see failures

                else:
                    yield {"type": "error", "result": self._error_response(
                        state, "internal_error", f"unknown action: {decision.action}")}
                    return

            # Max steps
            yield {"type": "partial", "result": self._partial_response(state, "max_steps")}

        except Exception as e:
            logger.error(f"agent.run_streaming: unhandled exception: {e}", exc_info=True)
            yield {"type": "error", "result": self._error_response(state, "internal_error", str(e))}

    def close(self) -> None:
        """Clean up resources."""
        self.vector_store.close()
