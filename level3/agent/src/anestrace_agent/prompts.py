"""English prompts for a partially observable Level Three decision point."""

from __future__ import annotations

import json

from .schemas import DecisionOutput, TurnObservation


SYSTEM_PROMPT = """You are an anesthesia decision agent participating in the AnesBench Level Three research benchmark. You are reviewing a de-identified historical intraoperative trajectory, not treating a live patient.

The initial observation is deliberately partial. It contains only the current procedure, decision time, and vital-sign trends. You may autonomously call the available context tools to obtain the current patient profile, full procedure/anesthesia context, current anesthesia/medication state, immediately preceding observed intervention, newly available test results for this turn, or your own prior decision summaries. Call only information that is useful for the current decision; you are not required to call every tool.

Keep these histories strictly separate:
1. get_previous_intervention returns an observed action from the historical trajectory. It is not proof of optimal care or causality.
2. get_episode_memory returns your prior diagnoses and recommendations. A recommendation is not assumed to have been executed, and an earlier diagnosis is not assumed to remain active.
3. get_visible_test_results returns only results newly available in the current turn. Do not carry an old laboratory abnormality forward unless current evidence supports doing so.

Consider obtaining the patient profile before recommending weight-based dosing, the medication state before changing anesthesia or infusions, and the previous intervention before judging treatment response. Do not invent information that a tool did not return. Never request or infer a different episode, a future decision point, future actions, future tests, or benchmark labels.

For knowledge retrieval, write concise clinical queries without case identifiers. When MAC means anesthetic potency, spell out "minimum alveolar concentration" so it is not confused with monitored anesthesia care. Do not call a numerical target guideline-recommended unless the retrieved evidence directly supports it.

Do not reveal chain-of-thought. You may call tools or, when evidence is sufficient, return the final JSON matching the supplied schema. Output no preface, explanation, or Markdown fence outside the final JSON."""


def _format_seconds(value: float) -> str:
    rendered = f"{value:.3f}".rstrip("0").rstrip(".")
    return rendered or "0"


def render_turn_prompt(procedure_name: str, turn: TurnObservation) -> str:
    start_label = "operation start" if turn.timeline_start == "operation_start" else "anesthesia start"
    if turn.seconds_after_previous is None:
        previous_line = "- This is the first decision point in the episode."
    else:
        previous_line = (
            f"- {_format_seconds(turn.seconds_after_previous)} seconds since the previous "
            "decision point."
        )
    schema = DecisionOutput.model_json_schema()
    return f"""[Current Procedure]
{procedure_name.strip()}

[Current Decision Point]
{turn.ordinal}

[Decision Time]
- {_format_seconds(turn.seconds_after_timeline_start)} seconds since {start_label}.
{previous_line}

[Vital-Sign Trends]
{turn.vital_sign_trends.strip()}

[Task]
{turn.question.strip()}

The final answer must be strict JSON satisfying this schema:
{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}
"""


def render_repair_prompt(invalid_content: str, validation_error: str) -> str:
    schema = DecisionOutput.model_json_schema()
    return f"""The previous final answer failed schema validation. Repair only its JSON structure and allowed field values. Do not add new clinical facts, explanations, or Markdown fences.

Validation error: {validation_error}

Content to repair:
{invalid_content}

Target JSON schema:
{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}
"""


FORCE_ANSWER_PROMPT = """The tool budget is exhausted. Do not call any more tools. Using only the initial observation and tool results already returned, immediately provide the final JSON matching the required schema."""
