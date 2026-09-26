from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskType(str, Enum):
    PARAMETER_EXTRACTION = "parameter_extraction"
    WAVEFORM_UNDERSTANDING = "waveform_understanding"
    ANESTHESIA_DIAGNOSIS = "anesthesia_diagnosis"
    VISUAL_GROUNDING = "visual_grounding"
    VISUAL_DESCRIPTION = "visual_description"


class Modality(str, Enum):
    MONITOR_SCREEN = "monitor_screen"
    ANESTHESIA_EVENT_PANEL = "anesthesia_event_panel"
    TREND_PLOT = "trend_plot"
    ECG = "ecg"
    ABP = "abp"
    PLETH = "pleth"
    ETCO2 = "etco2"
    TEE = "tee"
    CHEST_XRAY = "chest_xray"


class LabelTier(str, Enum):
    GOLD = "gold"
    SILVER = "silver"
    WEAK = "weak"


class KnowledgeDocumentType(str, Enum):
    CLINICAL_GUIDELINE = "clinical_guideline"
    DEVICE_MANUAL = "device_manual"
    CONSENSUS = "consensus"
    DRUG_LABEL = "drug_label"
    OTHER = "other"


class MediaRole(str, Enum):
    SHORT_WAVEFORM = "short_waveform"
    CONTEXT_WAVEFORM = "context_waveform"
    NUMERIC_TREND = "numeric_trend"
    THERAPY_TIMELINE = "therapy_timeline"
    MONITOR_SCREEN = "monitor_screen"
    PRIMARY_IMAGE = "primary_image"


class DiagnosticFamily(str, Enum):
    INDETERMINATE = "indeterminate"
    ARRHYTHMIA = "arrhythmia"
    HEMODYNAMIC = "hemodynamic"
    OXYGENATION_PERFUSION = "oxygenation_perfusion"
    VENTILATION_AIRWAY = "ventilation_airway"
    ANESTHESIA_DEPTH = "anesthesia_depth"
    EQUIPMENT_ARTIFACT = "equipment_artifact"
    COMBINED = "combined"


class DiagnosticSeverity(str, Enum):
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    INDETERMINATE = "indeterminate"


class DiagnosticCertainty(str, Enum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    INDETERMINATE = "indeterminate"


class ClinicalUrgency(str, Enum):
    ROUTINE = "routine"
    PROMPT = "prompt"
    URGENT = "urgent"
    IMMEDIATE = "immediate"
    INDETERMINATE = "indeterminate"


class BoundingBox(BaseModel):
    """Normalized xyxy coordinates in the inclusive range [0, 1]."""

    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_order(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return self


class MediaItem(BaseModel):
    path: str
    media_type: Literal["image", "video"] = "image"
    role: MediaRole | None = None
    visible_until_sec: float | None = Field(
        default=None,
        description="Latest visible time relative to diagnostic index time T0.",
    )
    sha256: str | None = None
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)


class KnowledgeReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_id: str
    document_type: KnowledgeDocumentType
    title: str
    version: str | None = None
    organization: str | None = None
    section: str
    excerpt: str = Field(min_length=1)
    source_uri: str | None = None
    sha256: str | None = None


class ParameterTarget(BaseModel):
    name: str
    value: float | str | None = None
    unit: str | None = None
    alarm_low: float | None = None
    alarm_high: float | None = None
    status: str | None = None
    bbox: BoundingBox | None = None


class WaveformTarget(BaseModel):
    channel: str
    morphology: list[str] = Field(default_factory=list)
    rhythm: list[str] = Field(default_factory=list)
    quality: list[str] = Field(default_factory=list)
    temporal_start_sec: float | None = None
    temporal_end_sec: float | None = None


class GroundingTarget(BaseModel):
    label: str
    bbox: BoundingBox


class DescriptionTarget(BaseModel):
    findings: list[str] = Field(default_factory=list)
    impression: str | None = None
    evidence_regions: list[GroundingTarget] = Field(default_factory=list)


class DiagnosticEvidence(BaseModel):
    channel: str
    findings: list[str] = Field(default_factory=list)
    temporal_start_sec: float | None = None
    temporal_end_sec: float | None = None
    bbox: BoundingBox | None = None

    @model_validator(mode="after")
    def check_temporal_order(self) -> "DiagnosticEvidence":
        if (
            self.temporal_start_sec is not None
            and self.temporal_end_sec is not None
            and self.temporal_end_sec <= self.temporal_start_sec
        ):
            raise ValueError("temporal_end_sec must be greater than temporal_start_sec")
        return self


class AnesthesiaDiagnosisTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostic_family: DiagnosticFamily
    primary_diagnosis: str | None = None
    status: Literal["present", "absent", "indeterminate"] = "present"
    severity: DiagnosticSeverity = DiagnosticSeverity.INDETERMINATE
    certainty: DiagnosticCertainty = DiagnosticCertainty.INDETERMINATE
    secondary_diagnoses: list[str] = Field(default_factory=list)
    observable_findings: list[str] = Field(default_factory=list)
    evidence: list[DiagnosticEvidence] = Field(default_factory=list)
    differential_diagnoses: list[str] = Field(default_factory=list)
    likely_causes: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    urgency: ClinicalUrgency = ClinicalUrgency.INDETERMINATE
    recommended_actions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_primary_diagnosis(self) -> "AnesthesiaDiagnosisTarget":
        if self.status == "present" and not self.primary_diagnosis:
            raise ValueError("primary_diagnosis is required when status is present")
        if self.status == "present" and not self.observable_findings:
            raise ValueError("observable_findings are required when status is present")
        if self.status == "present" and not self.evidence:
            raise ValueError("channel evidence is required when status is present")
        for item in self.evidence:
            if item.temporal_end_sec is not None and item.temporal_end_sec > 0:
                raise ValueError("diagnostic evidence cannot extend beyond index time T0")
        return self


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: list[ParameterTarget] = Field(default_factory=list)
    waveforms: list[WaveformTarget] = Field(default_factory=list)
    groundings: list[GroundingTarget] = Field(default_factory=list)
    description: DescriptionTarget | None = None
    diagnosis: AnesthesiaDiagnosisTarget | None = None
    device_status: list[str] = Field(default_factory=list)
    abstain_reason: str | None = None


class Provenance(BaseModel):
    dataset: str
    source_record: str
    label_source: str
    label_tier: LabelTier
    field_label_tiers: dict[str, LabelTier] = Field(default_factory=dict)
    reviewed_by: list[str] = Field(default_factory=list)
    annotation_model: str | None = None
    annotation_prompt_version: str | None = None
    knowledge_reference_ids: list[str] = Field(default_factory=list)
    license: str | None = None
    notes: list[str] = Field(default_factory=list)


class ClinicalOutcomeAudit(BaseModel):
    """Post-T0 facts retained for audit and never included in the model input/target."""

    model_config = ConfigDict(extra="forbid")

    observation_start_sec: float = Field(default=0.0, ge=0)
    observation_end_sec: float | None = Field(default=None, gt=0)
    actual_interventions: list[str] = Field(default_factory=list)
    intervention_response: list[str] = Field(default_factory=list)
    clinical_outcomes: list[str] = Field(default_factory=list)
    source: str | None = None
    reviewed_by: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_observation_window(self) -> "ClinicalOutcomeAudit":
        if (
            self.observation_end_sec is not None
            and self.observation_end_sec <= self.observation_start_sec
        ):
            raise ValueError("observation_end_sec must be greater than observation_start_sec")
        return self


class BenchmarkSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    patient_id: str
    study_id: str | None = None
    task: TaskType
    modality: Modality
    media: list[MediaItem] = Field(min_length=1)
    knowledge_references: list[KnowledgeReference] = Field(default_factory=list)
    prompt: str
    answer: StructuredAnswer
    provenance: Provenance
    outcome_audit: ClinicalOutcomeAudit | None = None
    split: Literal["train", "validation", "test"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_task_contract(self) -> "BenchmarkSample":
        is_unlabeled = self.metadata.get("annotation_status") == "unlabeled"
        if (
            self.task == TaskType.ANESTHESIA_DIAGNOSIS
            and self.answer.diagnosis is None
            and not is_unlabeled
        ):
            raise ValueError("anesthesia_diagnosis samples require answer.diagnosis")
        if self.task == TaskType.ANESTHESIA_DIAGNOSIS:
            for item in self.media:
                if item.visible_until_sec is not None and item.visible_until_sec > 0:
                    raise ValueError("diagnostic input media cannot expose information after T0")
        return self


class PredictionRecord(BaseModel):
    sample_id: str
    prediction: StructuredAnswer | None = None
    raw_output: str | None = None
    error: str | None = None
    model: str | None = None
    latency_ms: float | None = None
