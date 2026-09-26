#!/usr/bin/env python3
"""Reparse existing Level Two outputs without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .run_extended import parse_model_response
except ImportError:
    from run_extended import parse_model_response


PARTS = ("b1", "b2", "b3", "b4")


def has_complete_prediction(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(value.get(part), str) and value[part].strip() for part in PARTS
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reparse_file(path: Path, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(path)
    changed = 0
    already_complete = 0
    still_invalid = 0
    recovered_ids: list[str] = []
    invalid_ids: list[str] = []
    for line_number, row in enumerate(rows, 1):
        if has_complete_prediction(row.get("prediction")):
            already_complete += 1
            continue
        raw = row.get("prediction_text")
        if not isinstance(raw, str) or not raw.strip():
            still_invalid += 1
            invalid_ids.append(str(row.get("sample_id", f"line:{line_number}")))
            continue
        parsed = parse_model_response(raw, language)
        if not has_complete_prediction(parsed.prediction):
            still_invalid += 1
            invalid_ids.append(str(row.get("sample_id", f"line:{line_number}")))
            continue

        truncated = bool(row.get("truncated"))
        format_errors = list(parsed.format_errors)
        truncation_error = "generation reached max_new_tokens without EOS"
        if truncated and truncation_error not in format_errors:
            format_errors.append(truncation_error)
        row.update(
            {
                "final_answer_text": parsed.final_answer_text,
                "prediction": parsed.prediction,
                "prediction_sections": parsed.prediction_sections,
                "prediction_extraction_method": parsed.extraction_method,
                "strict_output_format_valid": (
                    parsed.strict_output_format_valid and not truncated
                ),
                "response_format_valid": not truncated,
                "format_errors": format_errors,
                "prediction_error": parsed.prediction_error,
                "status": "invalid_response" if truncated else "ok",
            }
        )
        changed += 1
        recovered_ids.append(str(row.get("sample_id", f"line:{line_number}")))

    return rows, {
        "path": str(path),
        "rows": len(rows),
        "already_complete": already_complete,
        "recovered": changed,
        "still_invalid": still_invalid,
        "recovered_ids": recovered_ids,
        "invalid_ids": invalid_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--language", choices=("zh", "en"), default="en")
    parser.add_argument(
        "--in-place", action="store_true",
        help="Write recovered fields back to each input; otherwise only report.",
    )
    parser.add_argument(
        "--backup-suffix", default=".before-semantic-reparse.bak",
        help="Suffix for an untouched backup created before an in-place update.",
    )
    parser.add_argument(
        "--show-ids", action="store_true",
        help="Include recovered and invalid sample ID lists in the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for raw_path in args.paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        rows, report = reparse_file(path, args.language)
        changed = int(report["recovered"])
        if args.in_place and changed:
            backup = path.with_name(path.name + args.backup_suffix)
            if backup.exists():
                raise FileExistsError(
                    f"backup already exists; refusing to overwrite: {backup}"
                )
            shutil.copy2(path, backup)
            write_atomic(path, rows)
            report["backup"] = str(backup)
            report["written"] = True
        else:
            report["written"] = False
        if not args.show_ids:
            report.pop("recovered_ids", None)
            report.pop("invalid_ids", None)
        reports.append(report)
    print(json.dumps({"files": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
