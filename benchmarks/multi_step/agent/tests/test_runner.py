import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from anestrace_agent import benchmark_runner
from anestrace_agent.config import load_settings


VALID = {
    "state_assessment": {
        "primary_problem": "Mild blood pressure variation requiring observation",
        "severity": "Mild",
        "key_evidence": ["MAP has mildly decreased."],
    },
    "Specific_intervention": {
        "treatment_goal": "Verify the signal and monitor the trend.",
        "action_list": [
            {"action_type": "Diagnostic_Check", "decision": "Verify the arterial waveform."}
        ],
    },
}


class AlwaysAnswerModel:
    def bind_tools(self, tools, tool_choice="auto"):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content=json.dumps(VALID))


def make_dataset(path: Path):
    turns = []
    for index in range(2):
        ordinal = f"T{index}"
        turns.append(
            {
                "turn_id": f"episode-1-{ordinal}",
                "decision_point": {
                    "ordinal": ordinal,
                    "timeline_start": "operation_start",
                    "seconds_after_timeline_start": 100 + index * 60,
                    "seconds_after_previous": None if index == 0 else 60,
                },
                "input": {
                    "anesthesia_medication_state": "MEDICATION_SECRET",
                    "vital_sign_trends": "MAP has mildly decreased.",
                    "previous_intervention": {
                        "source_decision_point": None if index == 0 else "T0",
                        "actions": [],
                    },
                    "visible_test_results": {"content": "LAB_SECRET"},
                },
                "answer": {"must_not_leak": True},
                "media": {"must_not_leak": True},
            }
        )
    record = {
        "schema_version": "anesbench-level-three.atomic-episode.v3-en-agent",
        "episode_id": "episode-1",
        "data_source": "VitalDB_INSPIRE",
        "procedure_name": "Aneurysm repair",
        "split": "test",
        "patient_information": {
            "patient_profile": "PROFILE_SECRET",
            "procedure_anesthesia_context": "PROCEDURE_SECRET",
        },
        "question": "Assess the current state.",
        "media": {"must_not_leak": True},
        "turns": turns,
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_runner_sqlite_outputs_resume_and_no_default_hidden_input(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    run_dir = tmp_path / "run"
    make_dataset(dataset)
    monkeypatch.setattr(benchmark_runner, "build_chat_model", lambda settings: AlwaysAnswerModel())
    monkeypatch.setattr(benchmark_runner, "build_runtime_tools", lambda settings: {})
    settings = load_settings()
    settings.runner.episode_workers = 1

    await benchmark_runner.run_benchmark(
        settings=settings,
        input_path=dataset,
        run_dir=run_dir,
        run_id="test-run",
    )
    predictions = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(predictions) == 2
    second = json.loads(predictions[1])
    assert len(second["memory_before"]) == 1
    direct = json.dumps(second["input"])
    assert "MAP has mildly decreased" in direct
    assert "PROFILE_SECRET" not in direct
    assert "MEDICATION_SECRET" not in direct
    assert "LAB_SECRET" not in direct
    assert "must_not_leak" not in json.dumps(second)
    assert (run_dir / "checkpoints.sqlite").exists()

    await benchmark_runner.run_benchmark(
        settings=settings,
        input_path=dataset,
        run_dir=run_dir,
        run_id="test-run",
        resume=True,
    )
    predictions_after = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert predictions_after == predictions
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert manifest["completed_turns"] == 2
    assert manifest["tool_usage"]["turns_without_context_tool"] == 2
