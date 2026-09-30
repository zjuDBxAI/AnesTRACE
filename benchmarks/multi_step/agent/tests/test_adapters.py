import json
from pathlib import Path

from anestrace_agent.adapters.level_three import DEFAULT_INPUT, episode_count_summary, load_episodes


def test_default_dataset_path_is_inside_public_subproject():
    assert DEFAULT_INPUT.name == "Level_three_v3_en_agent.jsonl"
    assert DEFAULT_INPUT.parent.name == "data"


def test_adapter_whitelists_hidden_context_and_discards_labels_media_future(tmp_path: Path):
    sentinel = "FUTURE_SECRET_SENTINEL"
    record = {
        "schema_version": "anesbench-level-three.atomic-episode.v3-en-agent",
        "episode_id": "episode-1",
        "data_source": "VitalDB_INSPIRE",
        "procedure_name": "Aneurysm repair",
        "split": "test",
        "patient_information": {
            "patient_profile": "PROFILE_SECRET",
            "procedure_anesthesia_context": "PROCEDURE_CONTEXT_SECRET",
        },
        "question": "Assess the current state.",
        "media": {"secret": sentinel},
        "turns": [
            {
                "turn_id": "episode-1-T0",
                "decision_point": {
                    "ordinal": "T0",
                    "timeline_start": "operation_start",
                    "seconds_after_timeline_start": 100,
                    "seconds_after_previous": None,
                },
                "input": {
                    "anesthesia_medication_state": "MEDICATION_SECRET",
                    "vital_sign_trends": "VITAL_VISIBLE",
                    "previous_intervention": {"source_decision_point": None, "actions": []},
                    "visible_test_results": {"content": "TEST_SECRET"},
                },
                "answer": {"leak": sentinel},
            },
            {
                "turn_id": "episode-1-T1",
                "decision_point": {
                    "ordinal": "T1",
                    "timeline_start": "operation_start",
                    "seconds_after_timeline_start": 160,
                    "seconds_after_previous": 60,
                },
                "input": {
                    "anesthesia_medication_state": sentinel,
                    "vital_sign_trends": sentinel,
                    "previous_intervention": {
                        "source_decision_point": "T0",
                        "actions": [sentinel],
                    },
                    "visible_test_results": {"content": sentinel},
                },
                "answer": {"leak": sentinel},
            },
        ],
    }
    source = tmp_path / "sample.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    episode = load_episodes(source)[0]
    assert episode_count_summary([episode]) == {
        "episodes": 1,
        "turns": 2,
        "data_sources": {"VitalDB_INSPIRE": 1},
        "episode_length_distribution": {"2": 1},
    }
    assert episode.procedure_name == "Aneurysm repair"
    assert episode.turns[0].vital_sign_trends == "VITAL_VISIBLE"
    assert sentinel not in episode.turns[0].model_dump_json()
    assert not hasattr(episode, "media")
    assert not hasattr(episode.turns[0], "answer")
