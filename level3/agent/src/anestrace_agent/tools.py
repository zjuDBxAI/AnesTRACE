"""Bounded knowledge tools and state-bound context tools for AnesTRACE-Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .config import PROJECT_ROOT, ToolSettings
from .knowledge import (
    get_anesthesia_drug_information,
    search_local_knowledge,
    search_pubmed,
)


class _ArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoArgs(_ArgsModel):
    pass


class QueryArgs(_ArgsModel):
    query: str = Field(min_length=2, max_length=500)


class DrugArgs(_ArgsModel):
    drug_name: str = Field(min_length=1, max_length=120)


@dataclass(frozen=True)
class RuntimeTool:
    tool: StructuredTool
    execute: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]
    tool_class: Literal["context", "knowledge"]
    information_section: str | None = None


def _trim(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return "[nested content omitted]"
    if isinstance(value, str):
        return value if len(value) <= 1500 else value[:1500] + "...[truncated]"
    if isinstance(value, list):
        return [_trim(item, depth=depth + 1) for item in value[:3]]
    if isinstance(value, dict):
        return {str(key): _trim(item, depth=depth + 1) for key, item in value.items()}
    return value


def compact_tool_result(value: Any, max_chars: int) -> Any:
    compact = _trim(value)
    encoded = json.dumps(compact, ensure_ascii=False, default=str)
    if len(encoded) <= max_chars:
        return compact
    return {
        "truncated": True,
        "content": encoded[:max_chars],
        "note": "The complete tool result is stored in the run trace.",
    }


def _resolve_project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _context_tools() -> list[RuntimeTool]:
    def patient_profile(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        return {"patient_profile": state["patient_profile"], "scope": "current_episode"}

    def procedure_context(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "procedure_anesthesia_context": state["procedure_anesthesia_context"],
            "scope": "current_episode",
        }

    def medication_state(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        turn = state["current_turn"]
        return {
            "decision_point": turn["ordinal"],
            "anesthesia_medication_state": turn["anesthesia_medication_state"],
            "scope": "current_turn",
        }

    def previous_intervention(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        turn = state["current_turn"]
        observed = turn["observed_intervention"]
        return {
            "decision_point": turn["ordinal"],
            "source_decision_point": observed.get("source_decision_point"),
            "actions": list(observed.get("actions") or []),
            "semantics": "observed intervention from the immediately preceding decision point",
        }

    def visible_tests(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        turn = state["current_turn"]
        return {
            "decision_point": turn["ordinal"],
            "visible_test_results": turn["visible_test_results"],
            "semantics": "newly available test results in the current turn only",
        }

    def episode_memory(_: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "memory": list(state.get("episode_memory") or []),
            "semantics": (
                "historical agent judgments; recommendations are not assumed to have been "
                "executed and earlier diagnoses are not assumed to remain active"
            ),
        }

    definitions = [
        (
            "get_patient_profile",
            "Return the current episode's age, sex, body size, ASA class, and recorded preoperative comorbidities. Takes no arguments.",
            "patient_profile",
            patient_profile,
        ),
        (
            "get_procedure_anesthesia_context",
            "Return the current episode's complete procedure and anesthesia context, including approach, position, urgency, airway, and arterial access. Takes no arguments.",
            "procedure_anesthesia_context",
            procedure_context,
        ),
        (
            "get_anesthesia_medication_state",
            "Return anesthesia, analgesia, vasoactive medication, and delivery-device state for the current decision point only. Takes no arguments.",
            "anesthesia_medication_state",
            medication_state,
        ),
        (
            "get_previous_intervention",
            "Return only the observed intervention assigned to the immediately preceding decision point. This is distinct from agent recommendations. Takes no arguments.",
            "previous_intervention",
            previous_intervention,
        ),
        (
            "get_visible_test_results",
            "Return only test results newly available at the current decision point; earlier results are not accumulated. Takes no arguments.",
            "visible_test_results",
            visible_tests,
        ),
        (
            "get_episode_memory",
            "Return prior agent diagnosis and recommendation summaries for this episode. These recommendations are not assumed to have been executed. Takes no arguments.",
            "episode_memory",
            episode_memory,
        ),
    ]
    output: list[RuntimeTool] = []
    for name, description, section, executor in definitions:
        output.append(
            RuntimeTool(
                tool=StructuredTool.from_function(
                    func=lambda: {}, name=name, description=description, args_schema=NoArgs
                ),
                execute=executor,
                tool_class="context",
                information_section=section,
            )
        )
    return output


def build_runtime_tools(settings: ToolSettings) -> dict[str, RuntimeTool]:
    cache_dir = _resolve_project_path(settings.cache_dir)
    assert cache_dir is not None
    guideline_corpus = _resolve_project_path(settings.guideline_corpus_path)
    miller_corpus = _resolve_project_path(settings.miller_corpus_path)

    def drug_execute(args: dict[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        return get_anesthesia_drug_information(
            str(args["drug_name"]),
            cache_dir=cache_dir / "drug_information",
            offline=not settings.network_enabled,
        )

    def guideline_execute(args: dict[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        return search_local_knowledge(
            str(args["query"]),
            corpus_path=guideline_corpus,
            top_k=settings.top_k,
            source_kind="guideline",
        )

    def miller_execute(args: dict[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        return search_local_knowledge(
            str(args["query"]),
            corpus_path=miller_corpus,
            top_k=settings.top_k,
            source_kind="licensed_textbook",
        )

    def pubmed_execute(args: dict[str, Any], _: Mapping[str, Any]) -> dict[str, Any]:
        return search_pubmed(
            str(args["query"]),
            top_k=settings.top_k,
            cache_dir=cache_dir / "pubmed",
            offline=not settings.network_enabled,
        )

    knowledge_definitions = [
        (
            "get_anesthesia_drug_information",
            "Retrieve RxNorm, DailyMed, and openFDA label information for an anesthesia-related drug. Supply only the drug name.",
            DrugArgs,
            drug_execute,
        ),
        (
            "search_guidelines",
            "Search anesthesia, perioperative, and resuscitation guidelines. Use a concise clinical question and spell out ambiguous abbreviations such as minimum alveolar concentration.",
            QueryArgs,
            guideline_execute,
        ),
        (
            "search_miller_anesthesia",
            "Search Miller's Anesthesia for relevant evidence. Use a concise clinical question and spell out ambiguous abbreviations.",
            QueryArgs,
            miller_execute,
        ),
        (
            "search_pubmed",
            "Search PubMed for perioperative research evidence when current literature is necessary.",
            QueryArgs,
            pubmed_execute,
        ),
    ]
    items = _context_tools()
    for name, description, schema, executor in knowledge_definitions:
        items.append(
            RuntimeTool(
                tool=StructuredTool.from_function(
                    func=lambda **_: {}, name=name, description=description, args_schema=schema
                ),
                execute=executor,
                tool_class="knowledge",
            )
        )
    return {item.tool.name: item for item in items}
