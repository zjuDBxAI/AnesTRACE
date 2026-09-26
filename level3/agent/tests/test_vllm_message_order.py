import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from anestrace_agent.config import AgentSettings
from anestrace_agent.graph import build_graph
from anestrace_agent.schemas import TurnObservation
from anestrace_agent.tools import RuntimeTool


class Query(BaseModel):
    query: str


VALID = {
    "state_assessment": {
        "primary_problem": "Intraoperative hypotension",
        "severity": "Moderate",
        "key_evidence": ["MAP is decreasing."],
    },
    "Specific_intervention": {
        "treatment_goal": "Restore perfusion.",
        "action_list": [
            {"action_type": "Diagnostic_Check", "decision": "Verify the arterial waveform."}
        ],
    },
}


class OrderCheckingModel:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, tool_choice="auto"):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        system_positions = [index for index, item in enumerate(messages) if isinstance(item, SystemMessage)]
        assert system_positions in ([], [0])
        if self.calls <= 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_test", "args": {"query": "test"}, "id": f"c{self.calls}"}
                ],
            )
        assert isinstance(messages[-1], HumanMessage)
        return AIMessage(content=json.dumps(VALID))


@pytest.mark.asyncio
async def test_forced_answer_keeps_system_message_first(tmp_path):
    tool = StructuredTool.from_function(
        func=lambda query: {}, name="search_test", description="test", args_schema=Query
    )
    runtime = RuntimeTool(
        tool=tool,
        execute=lambda args, state: {"ok": True},
        tool_class="knowledge",
    )
    model = OrderCheckingModel()
    graph = build_graph(
        model,
        {"search_test": runtime},
        AgentSettings(
            max_tool_rounds_per_turn=1,
            max_context_tool_calls_per_turn=1,
            max_knowledge_tool_calls_per_turn=1,
            max_total_tool_calls_per_turn=1,
        ),
        tmp_path,
    )
    turn = TurnObservation(
        turn_id="episode-T0",
        ordinal="T0",
        timeline_start="operation_start",
        seconds_after_timeline_start=100,
        seconds_after_previous=None,
        anesthesia_medication_state="hidden",
        vital_sign_trends="MAP 55 mmHg.",
        question="Assess.",
        observed_intervention={"source_decision_point": None, "actions": []},
        visible_test_results="hidden",
    )
    result = await graph.ainvoke(
        {
            "run_id": "run",
            "dataset_source": "VitalDB_INSPIRE",
            "episode_id": "episode",
            "procedure_name": "Aneurysm repair",
            "patient_profile": "hidden",
            "procedure_anesthesia_context": "hidden",
            "current_turn": turn.model_dump(),
            "episode_memory": [],
        },
        {"recursion_limit": 24},
    )
    assert result["turn_result"]["status"] == "success"
