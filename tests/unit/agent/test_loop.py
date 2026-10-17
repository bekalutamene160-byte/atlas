"""Unit tests for the Agent loop (Day 4: with synthesizer + critic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.agent.loop import Agent
from atlas.agent.planner import Planner
from atlas.agent.synthesizer import Synthesizer
from atlas.agent.critic import Critic
from atlas.config import Settings
from atlas.ingestion.embedder import MockEmbedder
from atlas.models import Chunk
from atlas.providers.mock import MockProvider
from atlas.retrieval.vector_store import VectorStore
from atlas.tools.dispatcher import ToolDispatcher
from atlas.tools.retriever_tool import RetrieverTool


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_provider="mock",
        embedding_provider="mock",
        max_agent_steps=5,
        run_timeout_seconds=10.0,
        max_tokens_per_run=10000,
        max_tool_calls=5,
        stuck_window=3,
    )


@pytest.fixture
def vector_store_with_data(tmp_path: Path) -> VectorStore:
    store = VectorStore(db_path=str(tmp_path / "agent_test.db"), embedding_dims=4)
    chunks = [
        Chunk(chunk_id="c1", source="atlas-3-spec.pdf", page=1,
              text="max pressure is 5 MPa", chunk_index=0),
        Chunk(chunk_id="c2", source="atlas-3-spec.pdf", page=2,
              text="max temperature is 80 C", chunk_index=1),
        Chunk(chunk_id="c3", source="atlas-5-spec.pdf", page=1,
              text="flow rate is 120 m3/h", chunk_index=0),
    ]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    store.add(chunks, vectors, "test-model")
    return store


@pytest.fixture
def agent(test_settings: Settings, vector_store_with_data: VectorStore) -> Agent:
    """Agent with mock provider and test data."""
    model = MockProvider()
    embedder = MockEmbedder(dims=4)
    planner = Planner(model=model)
    synthesizer = Synthesizer(model=model)
    critic = Critic(model=model)
    retriever = RetrieverTool(
        embedder=embedder,
        vector_store=vector_store_with_data,
        min_similarity=0.0,
    )
    dispatcher = ToolDispatcher(tools=[retriever])
    return Agent(
        settings=test_settings,
        model=model,
        embedder=embedder,
        vector_store=vector_store_with_data,
        planner=planner,
        synthesizer=synthesizer,
        critic=critic,
        dispatcher=dispatcher,
    )


class TestAgentRun:
    """Tests for the agent loop."""

    async def test_refuse_on_first_step(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "refuse",
             "reasoning": "corpus does not contain information about baking bread"}
        ])
        agent.planner = Planner(model=agent.model)
        result = await agent.run("How do I bake bread?")
        assert result["status"] == "refused"
        assert result["steps_taken"] == 1

    async def test_retrieve_then_refuse(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "retrieve", "query": "baking bread",
             "reasoning": "searching for bread info"},
            {"action": "refuse",
             "reasoning": "corpus does not contain information about baking bread"},
        ])
        agent.planner = Planner(model=agent.model)
        result = await agent.run("How do I bake bread?")
        assert result["status"] == "refused"
        assert result["steps_taken"] == 2

    async def test_max_steps_reached(self, agent: Agent):
        retrieve_response = {"action": "retrieve", "query": "test",
                              "reasoning": "need more evidence"}
        agent.model = MockProvider(structured_responses=[retrieve_response] * 10)
        agent.planner = Planner(model=agent.model)
        agent.settings = Settings(max_agent_steps=3, stuck_window=10,
                                   run_timeout_seconds=10.0,
                                   max_tokens_per_run=10000, max_tool_calls=50)
        result = await agent.run("test query")
        assert result["status"] == "partial"
        assert result["reason"] == "max_steps"

    async def test_stuck_detection(self, agent: Agent):
        retrieve_response = {"action": "retrieve", "query": "same query",
                              "reasoning": "need more evidence"}
        agent.model = MockProvider(structured_responses=[retrieve_response] * 10)
        agent.planner = Planner(model=agent.model)
        agent.settings = Settings(max_agent_steps=50, stuck_window=2,
                                   run_timeout_seconds=10.0,
                                   max_tokens_per_run=10000, max_tool_calls=50)
        result = await agent.run("test query")
        assert result["status"] == "partial"
        assert result["reason"] == "stuck"

    async def test_synthesize_success(self, agent: Agent):
        """Day 4: synthesize + critic → success."""
        # Script: retrieve → synthesize → critic approves
        agent.model = MockProvider(
            structured_responses=[
                # Step 1: planner says retrieve
                {"action": "retrieve", "query": "pressure",
                 "reasoning": "need pressure info"},
                # Step 2: planner says synthesize
                {"action": "synthesize",
                 "reasoning": "have enough evidence"},
                # Step 3: critic verdict (all supported)
                {"verdicts": [
                    {"claim_text": "The pressure is 5 MPa.",
                     "supported": True, "supporting_evidence_id": "e1"}
                ]},
            ],
            text_responses=[
                # Synthesizer output
                "The maximum operating pressure is 5 MPa [e1].",
            ],
        )
        agent.planner = Planner(model=agent.model)
        agent.synthesizer = Synthesizer(model=agent.model)
        agent.critic = Critic(model=agent.model)

        result = await agent.run("What is the max pressure?")
        assert result["status"] == "done"
        assert "5 MPa" in result["answer"]
        assert len(result["citations"]) > 0

    async def test_synthesize_rejected_then_refuse(self, agent: Agent):
        """Day 4: critic rejects → planner refuses."""
        agent.model = MockProvider(
            structured_responses=[
                {"action": "retrieve", "query": "pressure",
                 "reasoning": "need pressure info"},
                {"action": "synthesize",
                 "reasoning": "have enough evidence"},
                # Critic: one unsupported claim
                {"verdicts": [
                    {"claim_text": "The pressure is 200 MPa.",
                     "supported": False, "issue": "number_mismatch: evidence says 5 MPa"}
                ]},
                # After rejection, planner refuses
                {"action": "refuse",
                 "reasoning": "could not synthesize a verified answer"},
            ],
            text_responses=[
                "The maximum operating pressure is 200 MPa [e1].",
            ],
        )
        agent.planner = Planner(model=agent.model)
        agent.synthesizer = Synthesizer(model=agent.model)
        agent.critic = Critic(model=agent.model)

        result = await agent.run("What is the max pressure?")
        assert result["status"] == "refused"

    async def test_response_has_required_fields(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "refuse",
             "reasoning": "corpus does not contain this information"},
        ])
        agent.planner = Planner(model=agent.model)
        result = await agent.run("test query")
        for field in ["run_id", "query", "status", "steps_taken",
                       "evidence_count", "cost_tokens"]:
            assert field in result

    async def test_agent_never_crashes(self, agent: Agent):
        class ExplodingPlanner:
            async def decide(self, state):
                raise RuntimeError("planner exploded")
        agent.planner = ExplodingPlanner()
        result = await agent.run("test query")
        assert result["status"] == "error"
        assert result["error_code"] == "internal_error"

    async def test_retrieve_adds_evidence(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "retrieve", "query": "pressure",
             "reasoning": "searching for pressure info"},
            {"action": "refuse",
             "reasoning": "could not find enough information about pressure"},
        ])
        agent.planner = Planner(model=agent.model)
        result = await agent.run("What is the pressure?")
        assert result["evidence_count"] > 0

    async def test_token_tracking(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "refuse",
             "reasoning": "corpus does not contain this information"},
        ])
        agent.planner = Planner(model=agent.model)
        result = await agent.run("test query")
        assert result["cost_tokens"] > 0

    async def test_token_budget_exhausted(self, agent: Agent):
        agent.model = MockProvider(structured_responses=[
            {"action": "retrieve", "query": "test",
             "reasoning": "need more evidence"},
        ] * 10)
        agent.planner = Planner(model=agent.model)
        agent.settings = Settings(max_agent_steps=50, stuck_window=10,
                                   run_timeout_seconds=10.0,
                                   max_tokens_per_run=1000, max_tool_calls=50)
        result = await agent.run("test query")
        assert result["status"] == "partial"
        assert result["reason"] == "budget_exhausted"
