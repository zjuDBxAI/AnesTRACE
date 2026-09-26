#!/usr/bin/env python3
"""Evaluate structured waveform predictions exported in the WaveQA schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

try:
    from .waveform_metrics import evaluate_records
    from .waveform_schema import BenchmarkSample, PredictionRecord
except ImportError:  # Support ``python level1/evaluation/...py``.
    from waveform_metrics import evaluate_records
    from waveform_schema import BenchmarkSample, PredictionRecord


ModelT = TypeVar("ModelT", BenchmarkSample, PredictionRecord)


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid record: {exc}") from exc
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--abs-tol", type=float, default=1.0)
    parser.add_argument("--rel-tol", type=float, default=0.02)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_jsonl(args.manifest, BenchmarkSample)
    if args.split:
        samples = [sample for sample in samples if sample.split == args.split]
    predictions = read_jsonl(args.predictions, PredictionRecord)
    metrics: dict[str, Any] = evaluate_records(
        samples,
        predictions,
        abs_tol=args.abs_tol,
        rel_tol=args.rel_tol,
        iou_threshold=args.iou_threshold,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
