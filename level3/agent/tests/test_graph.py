import json
from collections import deque

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict

from anestrace_agent.config import AgentSettings
from anestrace_agent.graph import build_graph
from anestrace_agent.schemas import TurnObservation
from anestrace_agent.tools import RuntimeTool


VALID = {
    "state_assessment": {
        "primary_problem": "Intraoperative hypotension",
        "severity": "Moderate",
        "key_evidence": ["MAP is persistently decreasing."],
    },
    "Specific_intervention": {
        "treatment_goal": "Restore adequate perfusion.",
        "action_list": [
            {"action_type": "Diagnostic_Check", "decision": "Verify the arterial waveform."}
        ],
    },
}


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Query(BaseModel):
    query: str


class FakeModel:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.inputs = []

    def bind_tools(self, tools, tool_choice="auto"):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        if not self.responses:
            raise AssertionError("Fake model ran out of responses")
        return self.responses.popleft()


def state():
    turn = TurnObservation(
        turn_id="episode-T1",
        ordinal="T1",
        timeline_start="operation_start",
        seconds_after_timeline_start=160,
        seconds_after_previous=60,
        anesthesia_medication_state="MEDICATION_SECRET",
        vital_sign_trends="MAP decreased to 55 mmHg.",
        question="Assess the current state.",
        observed_intervention={"source_decision_point": "T0", "actions": ["ACTION_SECRET"]},
        visible_test_results="LAB_SECRET",
    )
    return {
        "run_id": "run",
        "dataset_source": "VitalDB_INSPIRE",
        "episode_id": "episode",
        "procedure_name": "Aneurysm repair",
        "patient_profile": "PROFILE_SECRET",
        "procedure_anesthesia_context": "PROCEDURE_SECRET",
        "current_turn": turn.model_dump(),
        "episode_memory": [
            {
                "decision_point": "T0",
                "diagnosis_summary": "MEMORY_SECRET",
                "severity": "Mild",
                "treatment_goal": "Observe",
                "proposed_actions": [
                    {"action_type": "Do_Nothing_Observe", "decision": "Continue monitoring."}
                ],
            }
        ],
    }


def context_tool(name="get_anesthesia_medication_state"):
    tool = StructuredTool.from_function(
        func=lambda: {}, name=name, description="current context", args_schema=NoArgs
    )
    return RuntimeTool(
        tool=tool,
        execute=lambda args, graph_state: {
            "anesthesia_medication_state": graph_state["current_turn"]["anesthesia_medication_state"]
        },
        tool_class="context",
        information_section="anesthesia_medication_state",
    )


def knowledge_tool():
    tool = StructuredTool.from_function(
        func=lambda query: {}, name="search_test", description="search", args_schema=Query
    )
    return RuntimeTool(
        tool=tool,
        execute=lambda args, graph_state: {"answer": args["query"]},
        tool_class="knowledge",
    )


@pytest.mark.asyncio
async def test_no_tool_answer_updates_memory_without_hidden_prompt_leak(tmp_path):
    model = FakeModel([AIMessage(content=json.dumps(VALID))])
    graph = build_graph(model, {}, AgentSettings(), tmp_path, checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        state(), {"configurable": {"thread_id": "thread-1"}, "recursion_limit": 24}
    )
    record = result["turn_result"]
    assert record["status"] == "success"
    assert record["memory_after"][-1]["decision_point"] == "T1"
    visible = json.dumps(record["input"])
    assert "MAP decreased" in visible
    assert "PROFILE_SECRET" not in visible
    assert "MEDICATION_SECRET" not in visible
    assert "ACTION_SECRET" not in visible
    assert "LAB_SECRET" not in visible
    initial_messages = model.inputs[0]
    initial_text = "\n".join(str(item.content) for item in initial_messages)
    assert "MAP decreased" in initial_text
    assert "PROFILE_SECRET" not in initial_text
    assert "MEMORY_SECRET" not in initial_text


@pytest.mark.asyncio
async def test_context_tool_is_autonomous_and_audited(tmp_path):
    runtime = context_tool()
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": runtime.tool.name, "args": {}, "id": "c1"}],
            ),
            AIMessage(content=json.dumps(VALID)),
        ]
    )
    graph = build_graph(model, {runtime.tool.name: runtime}, AgentSettings(), tmp_path)
    result = await graph.ainvoke(state(), {"recursion_limit": 24})
    record = result["turn_result"]
    assert record["context_tool_call_count"] == 1
    assert record["knowledge_tool_call_count"] == 0
    assert record["accessed_context_sections"] == ["anesthesia_medication_state"]
    assert record["tool_calls"][0]["tool_class"] == "context"
    assert (tmp_path / record["tool_calls"][0]["result_path"]).exists()


@pytest.mark.asyncio
async def test_duplicate_context_call_is_rejected_without_consuming_budget(tmp_path):
    runtime = context_tool()
    call = lambda identifier: AIMessage(
        content="", tool_calls=[{"name": runtime.tool.name, "args": {}, "id": identifier}]
    )
    model = FakeModel([call("c1"), call("c2"), AIMessage(content=json.dumps(VALID))])
    graph = build_graph(model, {runtime.tool.name: runtime}, AgentSettings(), tmp_path)
    result = await graph.ainvoke(state(), {"recursion_limit": 24})
    record = result["turn_result"]
    assert [item["status"] for item in record["tool_calls"]] == ["success", "duplicate_rejected"]
    assert record["tool_call_count"] == 1
    assert record["tool_rounds"] == 2


@pytest.mark.asyncio
async def test_context_and_knowledge_budgets_are_separate(tmp_path):
    context = context_tool()
    knowledge = knowledge_tool()
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": context.tool.name, "args": {}, "id": "c1"},
                    {"name": knowledge.tool.name, "args": {"query": "hypotension"}, "id": "k1"},
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": knowledge.tool.name, "args": {"query": "vasopressor"}, "id": "k2"}
                ],
            ),
            AIMessage(content=json.dumps(VALID)),
        ]
    )
    settings = AgentSettings(
        max_tool_rounds_per_turn=3,
        max_context_tool_calls_per_turn=1,
        max_knowledge_tool_calls_per_turn=1,
        max_total_tool_calls_per_turn=2,
    )
    graph = build_graph(
        model,
        {context.tool.name: context, knowledge.tool.name: knowledge},
        settings,
        tmp_path,
    )
    result = await graph.ainvoke(state(), {"recursion_limit": 24})
    record = result["turn_result"]
    assert record["context_tool_call_count"] == 1
    assert record["knowledge_tool_call_count"] == 1
    assert record["tool_calls"][-1]["status"] == "budget_rejected"


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_once(tmp_path):
    model = FakeModel([AIMessage(content="not-json"), AIMessage(content=json.dumps(VALID))])
    graph = build_graph(model, {}, AgentSettings(), tmp_path)
    result = await graph.ainvoke(state(), {"recursion_limit": 24})
    assert result["turn_result"]["status"] == "success_after_repair"
    assert [item["phase"] for item in result["turn_result"]["model_calls"]] == [
        "decision", "repair"
    ]
