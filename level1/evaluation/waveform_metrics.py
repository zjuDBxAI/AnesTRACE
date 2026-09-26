from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from .waveform_schema import (
        BenchmarkSample,
        BoundingBox,
        ParameterTarget,
        PredictionRecord,
    )
except ImportError:  # Support direct execution from this directory.
    from waveform_schema import (
        BenchmarkSample,
        BoundingBox,
        ParameterTarget,
        PredictionRecord,
    )

UNIT_ALIASES = {
    "beats/min": "bpm",
    "beat/min": "bpm",
    "mm hg": "mmhg",
    "percent": "%",
    "c": "degc",
    "celsius": "degc",
    "l/min/m2": "l/min/m2",
    "l/min/m^2": "l/min/m2",
}


def _norm_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _norm_unit(value: str | None) -> str:
    unit = _norm_text(value).replace("²", "2")
    return UNIT_ALIASES.get(unit, unit)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_match(reference: Any, prediction: Any, abs_tol: float, rel_tol: float) -> bool:
    ref, pred = _numeric(reference), _numeric(prediction)
    if ref is None or pred is None:
        return reference == prediction
    return abs(ref - pred) <= max(abs_tol, abs(ref) * rel_tol)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _set_counts(reference: set[str], prediction: set[str]) -> tuple[int, int, int]:
    return len(reference & prediction), len(prediction - reference), len(reference - prediction)


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


@dataclass
class Accumulator:
    samples: int = 0
    prediction_errors: int = 0
    missing_predictions: int = 0
    param_ref: int = 0
    param_pred: int = 0
    param_detected: int = 0
    value_correct: int = 0
    unit_scored: int = 0
    unit_correct: int = 0
    status_scored: int = 0
    status_correct: int = 0
    alarm_scored: int = 0
    alarm_correct: int = 0
    waveform_tp: int = 0
    waveform_fp: int = 0
    waveform_fn: int = 0
    waveform_samples: int = 0
    grounding_tp: int = 0
    grounding_fp: int = 0
    grounding_fn: int = 0
    grounding_samples: int = 0
    finding_tp: int = 0
    finding_fp: int = 0
    finding_fn: int = 0
    description_samples: int = 0
    diagnosis_samples: int = 0
    diagnosis_primary_scored: int = 0
    diagnosis_primary_correct: int = 0
    diagnosis_status_scored: int = 0
    diagnosis_status_correct: int = 0
    diagnosis_family_scored: int = 0
    diagnosis_family_correct: int = 0
    diagnosis_severity_scored: int = 0
    diagnosis_severity_correct: int = 0
    diagnosis_certainty_scored: int = 0
    diagnosis_certainty_correct: int = 0
    diagnosis_urgency_scored: int = 0
    diagnosis_urgency_correct: int = 0
    diagnosis_finding_tp: int = 0
    diagnosis_finding_fp: int = 0
    diagnosis_finding_fn: int = 0
    diagnosis_evidence_tp: int = 0
    diagnosis_evidence_fp: int = 0
    diagnosis_evidence_fn: int = 0
    diagnosis_differential_tp: int = 0
    diagnosis_differential_fp: int = 0
    diagnosis_differential_fn: int = 0
    diagnosis_cause_tp: int = 0
    diagnosis_cause_fp: int = 0
    diagnosis_cause_fn: int = 0
    diagnosis_action_tp: int = 0
    diagnosis_action_fp: int = 0
    diagnosis_action_fn: int = 0
    by_task: Counter[str] = field(default_factory=Counter)
    parameter_reference: Counter[str] = field(default_factory=Counter)
    parameter_detected: Counter[str] = field(default_factory=Counter)
    parameter_value_correct: Counter[str] = field(default_factory=Counter)


def _canonical_parameter_name(name: str) -> str:
    value = _norm_text(name).replace("map", "mbp")
    return value.replace("spo2", "spo2").replace("etco₂", "etco2")


def _parameter_index(parameters: Iterable[ParameterTarget]) -> dict[str, ParameterTarget]:
    return {_canonical_parameter_name(item.name): item for item in parameters}


def _diagnostic_evidence_set(diagnosis: Any) -> set[str]:
    if diagnosis is None:
        return set()
    return {
        f"{_norm_text(item.channel)}|{_norm_text(finding)}"
        for item in diagnosis.evidence
        for finding in item.findings
    }


def _diagnostic_sets(diagnosis: Any) -> dict[str, set[str]]:
    if diagnosis is None:
        return {
            "findings": set(),
            "evidence": set(),
            "differential": set(),
            "causes": set(),
            "actions": set(),
        }
    return {
        "findings": {_norm_text(value) for value in diagnosis.observable_findings},
        "evidence": _diagnostic_evidence_set(diagnosis),
        "differential": {_norm_text(value) for value in diagnosis.differential_diagnoses},
        "causes": {_norm_text(value) for value in diagnosis.likely_causes},
        "actions": {_norm_text(value) for value in diagnosis.recommended_actions},
    }


def _score_diagnostic_sets(acc: Accumulator, reference: Any, prediction: Any) -> None:
    ref_sets = _diagnostic_sets(reference)
    pred_sets = _diagnostic_sets(prediction)
    for name, prefix in (
        ("findings", "diagnosis_finding"),
        ("evidence", "diagnosis_evidence"),
        ("differential", "diagnosis_differential"),
        ("causes", "diagnosis_cause"),
        ("actions", "diagnosis_action"),
    ):
        tp, fp, fn = _set_counts(ref_sets[name], pred_sets[name])
        setattr(acc, prefix + "_tp", getattr(acc, prefix + "_tp") + tp)
        setattr(acc, prefix + "_fp", getattr(acc, prefix + "_fp") + fp)
        setattr(acc, prefix + "_fn", getattr(acc, prefix + "_fn") + fn)


def evaluate_records(
    samples: Iterable[BenchmarkSample],
    predictions: Iterable[PredictionRecord],
    abs_tol: float = 1.0,
    rel_tol: float = 0.02,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    refs = {sample.sample_id: sample for sample in samples}
    preds = {prediction.sample_id: prediction for prediction in predictions}
    acc = Accumulator(samples=len(refs))

    for sample_id, sample in refs.items():
        acc.by_task[sample.task.value] += 1
        ref_diagnosis = sample.answer.diagnosis
        if sample.task.value == "anesthesia_diagnosis" and ref_diagnosis is not None:
            acc.diagnosis_samples += 1
            acc.diagnosis_status_scored += 1
            acc.diagnosis_family_scored += 1
            acc.diagnosis_severity_scored += 1
            acc.diagnosis_certainty_scored += 1
            acc.diagnosis_urgency_scored += 1
            if ref_diagnosis.primary_diagnosis is not None:
                acc.diagnosis_primary_scored += 1
        record = preds.get(sample_id)
        if record is None:
            acc.missing_predictions += 1
            if ref_diagnosis is not None:
                _score_diagnostic_sets(acc, ref_diagnosis, None)
            continue
        if record.error or record.prediction is None:
            acc.prediction_errors += 1
            if ref_diagnosis is not None:
                _score_diagnostic_sets(acc, ref_diagnosis, None)
            continue
        answer, predicted = sample.answer, record.prediction

        ref_params, pred_params = _parameter_index(answer.parameters), _parameter_index(predicted.parameters)
        acc.param_ref += len(ref_params)
        acc.param_pred += len(pred_params)
        for name, ref in ref_params.items():
            acc.parameter_reference[name] += 1
            pred = pred_params.get(name)
            if pred is None:
                continue
            acc.param_detected += 1
            acc.parameter_detected[name] += 1
            is_value_correct = int(numeric_match(ref.value, pred.value, abs_tol, rel_tol))
            acc.value_correct += is_value_correct
            acc.parameter_value_correct[name] += is_value_correct
            if ref.unit is not None:
                acc.unit_scored += 1
                acc.unit_correct += int(_norm_unit(ref.unit) == _norm_unit(pred.unit))
            if ref.status is not None:
                acc.status_scored += 1
                acc.status_correct += int(_norm_text(ref.status) == _norm_text(pred.status))
            for field_name in ("alarm_low", "alarm_high"):
                ref_value = getattr(ref, field_name)
                if ref_value is not None:
                    acc.alarm_scored += 1
                    acc.alarm_correct += int(
                        numeric_match(ref_value, getattr(pred, field_name), abs_tol, rel_tol)
                    )

        if sample.task.value == "waveform_understanding":
            acc.waveform_samples += 1
            ref_wave = {
                _norm_text(value)
                for item in answer.waveforms
                for value in [*item.morphology, *item.rhythm, *item.quality]
            }
            pred_wave = {
                _norm_text(value)
                for item in predicted.waveforms
                for value in [*item.morphology, *item.rhythm, *item.quality]
            }
            tp, fp, fn = _set_counts(ref_wave, pred_wave)
            acc.waveform_tp += tp
            acc.waveform_fp += fp
            acc.waveform_fn += fn

        if sample.task.value == "visual_grounding":
            acc.grounding_samples += 1
            unmatched_pred = set(range(len(predicted.groundings)))
            for ref_grounding in answer.groundings:
                candidates = [
                    (bbox_iou(ref_grounding.bbox, predicted.groundings[idx].bbox), idx)
                    for idx in unmatched_pred
                    if _norm_text(ref_grounding.label) == _norm_text(predicted.groundings[idx].label)
                ]
                best = max(candidates, default=(0.0, -1))
                if best[0] >= iou_threshold:
                    acc.grounding_tp += 1
                    unmatched_pred.remove(best[1])
                else:
                    acc.grounding_fn += 1
            acc.grounding_fp += len(unmatched_pred)

        if sample.task.value == "visual_description":
            acc.description_samples += 1
            ref_findings = {
                _norm_text(value) for value in (answer.description.findings if answer.description else [])
            }
            pred_findings = {
                _norm_text(value) for value in (predicted.description.findings if predicted.description else [])
            }
            tp, fp, fn = _set_counts(ref_findings, pred_findings)
            acc.finding_tp += tp
            acc.finding_fp += fp
            acc.finding_fn += fn

        if sample.task.value == "anesthesia_diagnosis" and ref_diagnosis is not None:
            pred_diagnosis = predicted.diagnosis
            _score_diagnostic_sets(acc, ref_diagnosis, pred_diagnosis)
            if pred_diagnosis is not None:
                acc.diagnosis_status_correct += int(
                    ref_diagnosis.status == pred_diagnosis.status
                )
                acc.diagnosis_primary_correct += int(
                    ref_diagnosis.primary_diagnosis is not None
                    and _norm_text(ref_diagnosis.primary_diagnosis)
                    == _norm_text(pred_diagnosis.primary_diagnosis)
                )
                acc.diagnosis_family_correct += int(
                    ref_diagnosis.diagnostic_family == pred_diagnosis.diagnostic_family
                )
                acc.diagnosis_severity_correct += int(
                    ref_diagnosis.severity == pred_diagnosis.severity
                )
                acc.diagnosis_certainty_correct += int(
                    ref_diagnosis.certainty == pred_diagnosis.certainty
                )
                acc.diagnosis_urgency_correct += int(
                    ref_diagnosis.urgency == pred_diagnosis.urgency
                )

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    parameter_breakdown = {}
    for name in sorted(acc.parameter_reference):
        reference_count = acc.parameter_reference[name]
        parameter_breakdown[name] = {
            "reference": reference_count,
            "detection_recall": ratio(acc.parameter_detected[name], reference_count),
            "value_accuracy": ratio(acc.parameter_value_correct[name], reference_count),
        }

    param_precision = ratio(acc.param_detected, acc.param_pred)
    param_recall = ratio(acc.param_detected, acc.param_ref)
    param_f1 = (
        2 * param_precision * param_recall / (param_precision + param_recall)
        if param_precision is not None and param_recall is not None and param_precision + param_recall
        else None
    )

    return {
        "samples": acc.samples,
        "by_task": dict(acc.by_task),
        "missing_predictions": acc.missing_predictions,
        "prediction_errors": acc.prediction_errors,
        "parameter_extraction": {
            "reference_parameters": acc.param_ref,
            "predicted_parameters": acc.param_pred,
            "detection_precision": param_precision,
            "detection_recall": param_recall,
            "detection_f1": param_f1,
            "value_accuracy": ratio(acc.value_correct, acc.param_ref),
            "unit_accuracy": ratio(acc.unit_correct, acc.unit_scored),
            "status_accuracy": ratio(acc.status_correct, acc.status_scored),
            "alarm_threshold_accuracy": ratio(acc.alarm_correct, acc.alarm_scored),
            "numeric_tolerance": {"absolute": abs_tol, "relative": rel_tol},
            "by_parameter": parameter_breakdown,
        },
        "waveform_label_micro_f1": (
            _f1(acc.waveform_tp, acc.waveform_fp, acc.waveform_fn) if acc.waveform_samples else None
        ),
        "grounding_f1_at_iou": {
            "threshold": iou_threshold,
            "f1": (
                _f1(acc.grounding_tp, acc.grounding_fp, acc.grounding_fn)
                if acc.grounding_samples
                else None
            ),
        },
        "structured_finding_micro_f1": (
            _f1(acc.finding_tp, acc.finding_fp, acc.finding_fn)
            if acc.description_samples
            else None
        ),
        "anesthesia_diagnosis": {
            "samples": acc.diagnosis_samples,
            "status_accuracy": ratio(
                acc.diagnosis_status_correct, acc.diagnosis_status_scored
            ),
            "primary_diagnosis_accuracy": ratio(
                acc.diagnosis_primary_correct, acc.diagnosis_primary_scored
            ),
            "diagnostic_family_accuracy": ratio(
                acc.diagnosis_family_correct, acc.diagnosis_family_scored
            ),
            "severity_accuracy": ratio(
                acc.diagnosis_severity_correct, acc.diagnosis_severity_scored
            ),
            "certainty_accuracy": ratio(
                acc.diagnosis_certainty_correct, acc.diagnosis_certainty_scored
            ),
            "urgency_accuracy": ratio(
                acc.diagnosis_urgency_correct, acc.diagnosis_urgency_scored
            ),
            "observable_finding_micro_f1": (
                _f1(
                    acc.diagnosis_finding_tp,
                    acc.diagnosis_finding_fp,
                    acc.diagnosis_finding_fn,
                )
                if acc.diagnosis_samples
                else None
            ),
            "channel_evidence_micro_f1": (
                _f1(
                    acc.diagnosis_evidence_tp,
                    acc.diagnosis_evidence_fp,
                    acc.diagnosis_evidence_fn,
                )
                if acc.diagnosis_samples
                else None
            ),
            "differential_micro_f1": (
                _f1(
                    acc.diagnosis_differential_tp,
                    acc.diagnosis_differential_fp,
                    acc.diagnosis_differential_fn,
                )
                if acc.diagnosis_samples
                else None
            ),
            "likely_cause_micro_f1": (
                _f1(
                    acc.diagnosis_cause_tp,
                    acc.diagnosis_cause_fp,
                    acc.diagnosis_cause_fn,
                )
                if acc.diagnosis_samples
                else None
            ),
            "recommended_action_micro_f1": (
                _f1(
                    acc.diagnosis_action_tp,
                    acc.diagnosis_action_fp,
                    acc.diagnosis_action_fn,
                )
                if acc.diagnosis_samples
                else None
            ),
        },
        "notes": [
            "Free-text radiology/TEE quality requires a clinical evaluator such as RadGraph/RadCliQ plus human review.",
            "The structured finding F1 is not a substitute for factuality or clinical safety assessment.",
        ],
    }
