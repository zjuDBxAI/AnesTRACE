import pytest
from pydantic import ValidationError

from anestrace_agent.schemas import DecisionMemory, DecisionOutput


def answer(severity="Moderate", action_type="Diagnostic_Check"):
    return {
        "state_assessment": {
            "primary_problem": "Intraoperative hypotension",
            "severity": severity,
            "key_evidence": ["MAP is persistently decreasing."],
        },
        "Specific_intervention": {
            "treatment_goal": "Restore adequate perfusion.",
            "action_list": [
                {
                    "action_type": action_type,
                    "decision": "Verify the arterial waveform and assess volume status.",
                }
            ],
        },
    }


def test_public_schema_and_deterministic_memory():
    output = DecisionOutput.model_validate(answer())
    assert output.public_dict() == answer()
    memory = DecisionMemory.from_output("T0", output)
    assert memory.decision_point == "T0"
    assert not hasattr(memory, "key_evidence")


def test_stable_state_requires_observation_action():
    with pytest.raises(ValidationError):
        DecisionOutput.model_validate(answer(severity="Stable"))
    output = DecisionOutput.model_validate(
        answer(severity="Stable", action_type="Do_Nothing_Observe")
    )
    assert output.state_assessment.severity == "Stable"


def test_invalid_action_type_and_old_chinese_schema_are_rejected():
    with pytest.raises(ValidationError):
        DecisionOutput.model_validate(answer(action_type="Invented_Action"))
    with pytest.raises(ValidationError):
        DecisionOutput.model_validate(
            {
                "状态判断": {"当前主要问题": "低血压", "严重程度": "中度", "关键依据": ["MAP低"]},
                "专家建议动作": {"当前处理目标": "改善灌注", "action_list": []},
            }
        )
