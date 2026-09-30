from anestrace_agent.prompts import render_turn_prompt
from anestrace_agent.schemas import TurnObservation


def test_initial_prompt_contains_only_direct_observation():
    turn = TurnObservation(
        turn_id="episode-T1",
        ordinal="T1",
        timeline_start="operation_start",
        seconds_after_timeline_start=160,
        seconds_after_previous=60,
        anesthesia_medication_state="MEDICATION_SECRET",
        vital_sign_trends="VITAL_VISIBLE",
        observed_intervention={"source_decision_point": "T0", "actions": ["ACTION_SECRET"]},
        visible_test_results="LAB_SECRET",
        question="Assess the current state.",
    )
    prompt = render_turn_prompt("Aneurysm repair", turn)
    assert "Aneurysm repair" in prompt
    assert "T1" in prompt
    assert "160 seconds since operation start" in prompt
    assert "60 seconds since the previous decision point" in prompt
    assert "VITAL_VISIBLE" in prompt
    assert "MEDICATION_SECRET" not in prompt
    assert "ACTION_SECRET" not in prompt
    assert "LAB_SECRET" not in prompt


def test_mover_time_is_not_mislabeled_as_operation_start():
    turn = TurnObservation(
        turn_id="episode-T0",
        ordinal="T0",
        timeline_start="anesthesia_start",
        seconds_after_timeline_start=120,
        seconds_after_previous=None,
        anesthesia_medication_state="hidden",
        vital_sign_trends="visible",
        observed_intervention={"source_decision_point": None, "actions": []},
        visible_test_results="hidden",
        question="Assess.",
    )
    prompt = render_turn_prompt("Exploratory laparotomy", turn)
    assert "120 seconds since anesthesia start" in prompt
    assert "first decision point" in prompt
    assert "operation start" not in prompt
