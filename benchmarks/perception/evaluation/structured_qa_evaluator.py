"""Public reference-based L1 scoring primitives."""

from __future__ import annotations

from typing import Any


def _answer(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("prediction", row.get("answer"))
    return value if isinstance(value, dict) else {}


def _box(answer: dict[str, Any]) -> list[float] | None:
    value = (answer.get("localization") or {}).get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return box if 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 else None


def _iou(a: list[float], b: list[float]) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def _facts(answer: dict[str, Any]) -> set[tuple[str, ...]]:
    rows = answer.get("clinical_facts") or []
    return {tuple(str(part).casefold() for part in row) for row in rows if isinstance(row, list)}


def evaluate_item(prediction: dict[str, Any], gold: dict[str, Any], profile: str) -> dict[str, Any]:
    reference = _answer(gold)
    candidate = _answer(prediction)
    kind = gold.get("answer_type")
    if kind == "multiple_choice":
        expected = (reference.get("choice") or {}).get("option_id")
        actual = (candidate.get("choice") or {}).get("option_id")
        valid = isinstance(expected, str) and isinstance(actual, str)
        score = float(expected == actual) if valid else 0.0
        components = {"accuracy": score}
    elif kind == "bbox_or_point":
        expected, actual = _box(reference), _box(candidate)
        valid = expected is not None and actual is not None
        score = _iou(expected, actual) if valid else 0.0
        components = {"miou": score}
    elif kind == "structured_label_plus_natural_language":
        expected = _facts(reference)
        actual = _facts(candidate)
        valid = bool(expected) and bool(candidate)
        tp = len(expected & actual)
        fp = len(actual - expected)
        fn = len(expected - actual)
        score = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        components = {"fact_f1": score, "fact_tp": tp, "fact_fp": fp, "fact_fn": fn}
    else:
        valid, score, components = False, 0.0, {}
    return {
        "valid": valid,
        "score": score,
        "uncapped_score": score,
        "components": components,
        "active_components": list(components),
        "error": None if valid else "missing or invalid answer",
    }
