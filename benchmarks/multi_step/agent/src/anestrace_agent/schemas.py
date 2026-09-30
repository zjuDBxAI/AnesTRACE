"""Validated public and internal schemas for the English Level Three agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Severity = Literal["Stable", "Mild", "Moderate", "Severe", "Critical", "Uncertain"]
ActionType = Literal[
    "Bolus", "Infusion_Start", "Infusion_Adjust", "Infusion_Stop",
    "Device_Adjust", "Airway_Procedure", "Fluid_Admin", "Blood_Product",
    "Diagnostic_Check", "Escalate", "Do_Nothing_Observe",
]
TimelineStart = Literal["operation_start", "anesthesia_start"]
DatasetSource = Literal["VitalDB_INSPIRE", "Mover_EPIC"]
ToolClass = Literal["context", "knowledge"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StateAssessment(StrictModel):
    primary_problem: str = Field(min_length=1)
    severity: Severity
    key_evidence: list[str] = Field(min_length=1, max_length=5)

    @field_validator("primary_problem")
    @classmethod
    def strip_problem(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("primary_problem must not be empty")
        return value


class DecisionAction(StrictModel):
    action_type: ActionType
    decision: str = Field(min_length=1)

    @field_validator("decision")
    @classmethod
    def strip_decision(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("decision must not be empty")
        return value


class SpecificIntervention(StrictModel):
    treatment_goal: str = Field(min_length=1)
    action_list: list[DecisionAction] = Field(min_length=1, max_length=5)


class DecisionOutput(StrictModel):
    state_assessment: StateAssessment
    Specific_intervention: SpecificIntervention

    @model_validator(mode="after")
    def stable_state_has_observation_action(self) -> "DecisionOutput":
        if self.state_assessment.severity == "Stable" and not any(
            action.action_type == "Do_Nothing_Observe"
            for action in self.Specific_intervention.action_list
        ):
            raise ValueError("Stable state must include Do_Nothing_Observe")
        return self

    def public_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DecisionMemory(StrictModel):
    decision_point: str
    diagnosis_summary: str
    severity: Severity
    treatment_goal: str
    proposed_actions: list[DecisionAction]

    @classmethod
    def from_output(cls, decision_point: str, output: DecisionOutput) -> "DecisionMemory":
        return cls(
            decision_point=decision_point,
            diagnosis_summary=output.state_assessment.primary_problem,
            severity=output.state_assessment.severity,
            treatment_goal=output.Specific_intervention.treatment_goal,
            proposed_actions=output.Specific_intervention.action_list,
        )


class ObservedIntervention(StrictModel):
    source_decision_point: str | None = None
    actions: list[str] = Field(default_factory=list)


class TurnObservation(StrictModel):
    turn_id: str
    ordinal: str = Field(pattern=r"^T[0-4]$")
    timeline_start: TimelineStart
    seconds_after_timeline_start: float = Field(ge=0)
    seconds_after_previous: float | None = Field(default=None, ge=0)
    anesthesia_medication_state: str = Field(min_length=1)
    vital_sign_trends: str = Field(min_length=1)
    observed_intervention: ObservedIntervention
    visible_test_results: str = Field(min_length=1)
    question: str = Field(min_length=1)


class EpisodeInput(StrictModel):
    dataset_source: DatasetSource
    source_path: str
    source_line: int = Field(ge=1)
    episode_id: str
    split: str | None = None
    procedure_name: str = Field(min_length=1)
    patient_profile: str = Field(min_length=1)
    procedure_anesthesia_context: str = Field(min_length=1)
    turns: list[TurnObservation] = Field(min_length=1)


class ToolTrace(StrictModel):
    call_id: str
    tool_name: str
    tool_class: ToolClass
    information_section: str | None = None
    arguments: dict[str, Any]
    status: Literal["success", "error", "budget_rejected", "duplicate_rejected"]
    started_at: str
    elapsed_seconds: float = Field(ge=0)
    result_sha256: str | None = None
    result_path: str | None = None
    compact_result: Any = None
    error: str | None = None


class ModelCallTrace(StrictModel):
    phase: Literal["decision", "forced_answer", "repair"]
    elapsed_seconds: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    response_id: str | None = None
