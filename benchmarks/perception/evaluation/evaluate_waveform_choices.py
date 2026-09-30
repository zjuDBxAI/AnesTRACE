#!/usr/bin/env python3
"""Score WaveQA Level 1 single-choice predictions with the Python standard library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score WaveQA Level 1 accuracy.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    questions = _read_jsonl(args.benchmark)
    predictions = _read_jsonl(args.predictions) if args.predictions.is_file() else []

    question_ids = {str(row.get("question_id")) for row in questions}
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in predictions:
        question_id = str(row.get("question_id"))
        if question_id in prediction_by_id:
            duplicate_ids.append(question_id)
        prediction_by_id[question_id] = row

    details: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    correct = 0
    scored = 0
    for question in questions:
        question_id = str(question["question_id"])
        expected = str(question["evaluation"]["answer_key"]).strip().upper()
        prediction = prediction_by_id.get(question_id)
        answer = prediction.get("answer") if prediction is not None else None
        normalized = answer.strip().upper() if isinstance(answer, str) else None
        error: str | None = None
        if prediction is None:
            missing.append(question_id)
            error = "missing prediction"
        elif normalized not in {"A", "B", "C", "D"}:
            invalid.append(question_id)
            error = "answer must be A, B, C, or D"
        else:
            scored += 1
            if normalized == expected:
                correct += 1
        details.append({
            "question_id": question_id,
            "expected": expected,
            "predicted": normalized,
            "correct": normalized == expected if error is None else False,
            "error": error,
        })

    extra = sorted(set(prediction_by_id) - question_ids)
    complete = not (missing or invalid or extra or duplicate_ids)
    metrics = {
        "complete": complete,
        "questions": len(questions),
        "scored": scored,
        "correct": correct,
        "accuracy": correct / scored if scored else 0.0,
        "coverage": scored / len(questions) if questions else 0.0,
        "missing_predictions": missing,
        "invalid_predictions": invalid,
        "extra_predictions": extra,
        "duplicate_predictions": sorted(set(duplicate_ids)),
        "inputs": {
            "benchmark": str(args.benchmark.resolve()),
            "predictions": str(args.predictions.resolve()),
        },
    }
    _atomic_json(args.metrics_output, metrics)
    _atomic_jsonl(args.details_output, details)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if complete or args.allow_partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
