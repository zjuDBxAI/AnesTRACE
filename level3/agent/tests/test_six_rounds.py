import json
from collections import deque

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from anestrace_agent.config import AgentSettings
from anestrace_agent.graph import build_graph
from anestrace_agent.schemas import TurnObservation
from anestrace_agent.tools import RuntimeTool


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model:
    def __init__(self, responses):
        self.responses = deque(responses)

    def bind_tools(self, tools, tool_choice="auto"):
        return self

    async def ainvoke(self, messages):
        return self.responses.popleft()


@pytest.mark.asyncio
async def test_six_sequential_context_rounds_fit_recursion_limit_24(tmp_path):
    names = [f"get_context_{index}" for index in range(6)]
    runtime = {}
    responses = []
    for index, name in enumerate(names):
        tool = StructuredTool.from_function(
            func=lambda: {}, name=name, description="context", args_schema=NoArgs
        )
        runtime[name] = RuntimeTool(
            tool=tool,
            execute=lambda args, state: {"ok": True},
            tool_class="context",
            information_section=name,
        )
        responses.append(
            AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": f"c{index}"}])
        )
    responses.append(
        AIMessage(
            content=json.dumps(
                {
                    "state_assessment": {
                        "primary_problem": "Stable intraoperative state",
                        "severity": "Stable",
                        "key_evidence": ["Vital signs are stable."],
                    },
                    "Specific_intervention": {
                        "treatment_goal": "Maintain stability.",
                        "action_list": [
                            {
                                "action_type": "Do_Nothing_Observe",
                                "decision": "Continue routine monitoring.",
                            }
                        ],
                    },
                }
            )
        )
    )
    turn = TurnObservation(
        turn_id="episode-T0",
        ordinal="T0",
        timeline_start="operation_start",
        seconds_after_timeline_start=10,
        anesthesia_medication_state="hidden",
        vital_sign_trends="Stable.",
        observed_intervention={"source_decision_point": None, "actions": []},
        visible_test_results="hidden",
        question="Assess.",
    )
    graph = build_graph(
        Model(responses),
        runtime,
        AgentSettings(
            max_tool_rounds_per_turn=6,
            max_context_tool_calls_per_turn=6,
            max_knowledge_tool_calls_per_turn=0,
            max_total_tool_calls_per_turn=6,
        ),
        tmp_path,
    )
    result = await graph.ainvoke(
        {
            "run_id": "run",
            "dataset_source": "VitalDB_INSPIRE",
            "episode_id": "episode",
            "procedure_name": "Procedure",
            "patient_profile": "hidden",
            "procedure_anesthesia_context": "hidden",
            "current_turn": turn.model_dump(),
            "episode_memory": [],
        },
        {"recursion_limit": 24},
    )
    record = result["turn_result"]
    assert record["status"] == "success"
    assert record["tool_rounds"] == 6
    assert record["context_tool_call_count"] == 6
