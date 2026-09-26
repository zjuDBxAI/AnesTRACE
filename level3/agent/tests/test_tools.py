from anestrace_agent.config import load_settings
from anestrace_agent.knowledge import search_local_knowledge
from anestrace_agent.tools import build_runtime_tools, compact_tool_result


CONTEXT_TOOLS = {
    "get_patient_profile",
    "get_procedure_anesthesia_context",
    "get_anesthesia_medication_state",
    "get_previous_intervention",
    "get_visible_test_results",
    "get_episode_memory",
}
KNOWLEDGE_TOOLS = {
    "get_anesthesia_drug_information",
    "search_guidelines",
    "search_miller_anesthesia",
    "search_pubmed",
}


def test_exact_ten_restricted_tool_names_and_arguments():
    runtime = build_runtime_tools(load_settings().tools)
    assert set(runtime) == CONTEXT_TOOLS | KNOWLEDGE_TOOLS
    assert all(runtime[name].tool.args == {} for name in CONTEXT_TOOLS)
    assert set(runtime["search_pubmed"].tool.args) == {"query"}
    assert set(runtime["get_anesthesia_drug_information"].tool.args) == {"drug_name"}
    assert all(runtime[name].tool_class == "context" for name in CONTEXT_TOOLS)
    assert all(runtime[name].tool_class == "knowledge" for name in KNOWLEDGE_TOOLS)


def test_context_tools_are_bound_to_current_state_only():
    runtime = build_runtime_tools(load_settings().tools)
    state = {
        "patient_profile": "CURRENT_PROFILE",
        "procedure_anesthesia_context": "CURRENT_PROCEDURE",
        "episode_memory": [{"decision_point": "T0"}],
        "current_turn": {
            "ordinal": "T1",
            "anesthesia_medication_state": "CURRENT_MEDICATION",
            "observed_intervention": {"source_decision_point": "T0", "actions": ["ACTION"]},
            "visible_test_results": "CURRENT_TEST",
        },
    }
    assert runtime["get_patient_profile"].execute({}, state)["patient_profile"] == "CURRENT_PROFILE"
    assert runtime["get_anesthesia_medication_state"].execute({}, state)["decision_point"] == "T1"
    assert runtime["get_previous_intervention"].execute({}, state)["actions"] == ["ACTION"]
    assert runtime["get_visible_test_results"].execute({}, state)["visible_test_results"] == "CURRENT_TEST"
    assert runtime["get_episode_memory"].execute({}, state)["memory"] == [{"decision_point": "T0"}]


def test_compaction_is_bounded():
    compact = compact_tool_result({"results": [{"abstract": "x" * 20000}]}, 2000)
    assert len(str(compact)) < 9000


def test_local_knowledge_requires_explicit_corpus_and_searches_jsonl(tmp_path):
    unavailable = search_local_knowledge(
        "hypotension",
        corpus_path=None,
        top_k=3,
        source_kind="guideline",
    )
    assert unavailable["status"] == "unavailable"

    corpus = tmp_path / "knowledge.jsonl"
    corpus.write_text(
        '{"chunk_id":"a","title":"Blood pressure","text":"Treat persistent intraoperative hypotension."}\n'
        '{"chunk_id":"b","title":"Airway","text":"Confirm airway position after intubation."}\n',
        encoding="utf-8",
    )
    result = search_local_knowledge(
        "persistent hypotension",
        corpus_path=corpus,
        top_k=1,
        source_kind="guideline",
    )
    assert result["status"] == "ok"
    assert result["results"][0]["chunk_id"] == "a"
