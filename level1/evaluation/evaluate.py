"""Command-line evaluation for AnesBench structured QA predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.structured_qa_evaluator import evaluate_item
else:
    from eval.structured_qa_evaluator import evaluate_item


OFFICIAL_ANNOTATIONS = {"human_reviewed", "expert_adjudicated", "ground_truth"}
OFFICIAL_REVIEWS = {"accepted", "revised", "adjudicated"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(value)
    return records


def index_by_qa_id(
    records: list[dict[str, Any]], source: Path, *, latest_wins: bool = False
) -> tuple[dict[str, dict[str, Any]], int]:
    indexed: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for record in records:
        qa_id = record.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id:
            raise ValueError(f"{source}: every record requires a non-empty qa_id")
        if qa_id in indexed:
            if not latest_wins:
                raise ValueError(f"{source}: duplicate qa_id: {qa_id}")
            duplicates += 1
        indexed[qa_id] = record
    return indexed, duplicates


def is_official(record: dict[str, Any]) -> bool:
    annotation = (record.get("ground_truth") or {}).get("annotation_status")
    return annotation in OFFICIAL_ANNOTATIONS or record.get("review_status") in OFFICIAL_REVIEWS


def infer_profile(path: Path, records: list[dict[str, Any]]) -> str:
    name = path.name.casefold()
    if "waveform" in name:
        return "waveform"
    if "tee" in name:
        return "tee"
    for record in records:
        media_path = str((record.get("media") or {}).get("image_path") or "").casefold()
        if "waveform" in media_path:
            return "waveform"
        if "tee" in media_path:
            return "tee"
    raise ValueError("cannot infer profile; pass --profile tee or --profile waveform")


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def evaluate(
    gold_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    profile: str,
    official_only: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions, duplicate_predictions = index_by_qa_id(
        prediction_records, Path("predictions"), latest_wins=True
    )
    gold_index, _ = index_by_qa_id(gold_records, Path("gold"))
    eligible = [
        record
        for record in gold_records
        if (
            profile == "tee"
            or record.get("answer_type") == "structured_label_plus_natural_language"
        )
        and (not official_only or is_official(record))
    ]
    audit: list[dict[str, Any]] = []
    by_task: dict[str, list[float]] = defaultdict(list)
    by_answer_type: dict[str, list[float]] = defaultdict(list)
    by_annotation: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    components_by_task: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    missing = 0
    invalid = 0
    invalid_references = 0
    legacy_references = 0
    weak_references = 0

    for gold in eligible:
        qa_id = gold["qa_id"]
        reference_check = evaluate_item(gold, gold, profile)
        if not reference_check["valid"]:
            invalid_references += 1
            audit.append(
                {
                    "qa_id": qa_id,
                    "task_name": str(gold.get("task_name") or "unknown"),
                    "answer_type": gold.get("answer_type"),
                    "review_status": gold.get("review_status"),
                    "annotation_status": (gold.get("ground_truth") or {}).get(
                        "annotation_status"
                    ),
                    "valid": False,
                    "score": None,
                    "error": f"invalid reference: {reference_check.get('error', 'unknown')}",
                }
            )
            continue
        prediction = predictions.get(qa_id)
        if prediction is None:
            result = {
                "valid": False,
                "score": 0.0,
                "uncapped_score": 0.0,
                "components": {},
                "active_components": [],
                "error": "missing prediction",
            }
            missing += 1
        elif prediction.get("status") not in {None, "ok"}:
            result = {
                "valid": False,
                "score": 0.0,
                "uncapped_score": 0.0,
                "components": {},
                "active_components": [],
                "error": f"prediction status is {prediction.get('status')!r}",
            }
            invalid += 1
        else:
            result = evaluate_item(prediction, gold, profile)
            invalid += int(not result["valid"])
        if str(result.get("reference_adapter", "")).startswith("legacy"):
            legacy_references += 1
        if not is_official(gold):
            weak_references += 1
        task = str(gold.get("task_name") or "unknown")
        answer_type = str(gold.get("answer_type") or "unknown")
        ground_truth = gold.get("ground_truth") or {}
        annotation = str(ground_truth.get("annotation_status") or "unknown")
        source = str(ground_truth.get("source_type") or "unknown")
        score = float(result["score"])
        by_task[task].append(score)
        by_answer_type[answer_type].append(score)
        by_annotation[annotation].append(score)
        by_source[source].append(score)
        for component, value in result.get("components", {}).items():
            if isinstance(value, (int, float)):
                components_by_task[task][component].append(float(value))
        audit.append(
            {
                "qa_id": qa_id,
                "task_name": task,
                "answer_type": answer_type,
                "review_status": gold.get("review_status"),
                "annotation_status": annotation,
                "source_type": source,
                **result,
            }
        )

    scores = [float(item["score"]) for item in audit if item.get("score") is not None]
    task_scores = {task: mean(values) for task, values in sorted(by_task.items())}
    task_macro_values = [value for value in task_scores.values() if value is not None]
    task_metrics = {
        task: {
            "count": len(by_task[task]),
            "score": task_scores[task],
            "components": {
                name: mean(values)
                for name, values in sorted(components_by_task[task].items())
            },
        }
        for task in sorted(by_task)
    }
    summary = {
        "profile": profile,
        "score_name": (
            "AnesBench TEE Task-Macro Score"
            if profile == "tee"
            else "Structured Rubric Score (SRS)"
        ),
        "macro_score": mean(task_macro_values),
        "item_macro_score": mean(scores),
        "evaluated_items": len(scores),
        "eligible_items": len(eligible),
        "available_predictions": len(predictions),
        "duplicate_prediction_records": duplicate_predictions,
        "unexpected_prediction_ids": len(set(predictions) - set(gold_index)),
        "missing_predictions": missing,
        "invalid_predictions": invalid,
        "invalid_references_excluded": invalid_references,
        "weak_reference_items": weak_references,
        "legacy_reference_items": legacy_references,
        "official_only": official_only,
        "task_scores": task_scores,
        "task_metrics": task_metrics,
        "answer_type_scores": {
            key: mean(values) for key, values in sorted(by_answer_type.items())
        },
        "annotation_status_scores": {
            key: mean(values) for key, values in sorted(by_annotation.items())
        },
        "source_type_scores": {
            key: mean(values) for key, values in sorted(by_source.items())
        },
        "warning": (
            "No eligible scorable items were found. Check review status and reference schema."
            if not scores
            else (
                "Scores using weak or legacy-adapted references are provisional; "
                "use --official-only for reviewed or manifest-derived labels."
                if weak_references or legacy_references
                else None
            )
        ),
    }
    return summary, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Benchmark JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument("--profile", choices=["auto", "tee", "waveform"], default="auto")
    parser.add_argument("--official-only", action="store_true", help="Exclude weak labels")
    parser.add_argument("--clinical-facts-gold", type=Path, help="Frozen ClinicalFact sidecar for functional and abnormality tasks")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--output", type=Path, help="Write per-item audit JSONL")
    parser.add_argument("--summary-output", type=Path, help="Write summary JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold = read_jsonl(args.gold)
    predictions = read_jsonl(args.predictions)
    profile = infer_profile(args.gold, gold) if args.profile == "auto" else args.profile
    if args.clinical_facts_gold:
        from eval.clinical_fact_benchmark import evaluate_with_clinical_facts
        summary, audit = evaluate_with_clinical_facts(
            gold, predictions, read_jsonl(args.clinical_facts_gold),
            profile, args.official_only, args.language, hashlib.sha256(args.gold.read_bytes()).hexdigest(),
        )
    else:
        summary, audit = evaluate(gold, predictions, profile, args.official_only)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for item in audit:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
