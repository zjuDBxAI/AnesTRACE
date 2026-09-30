#!/usr/bin/env python3
"""Evaluate Level Two B5 responses with a local AnesTRACE-Eval judge.

The judge is trained on ``AnesTRACE-Eval-Data.jsonl``.  This evaluator uses
the overall-level prompt (one call per case) and scores B1--B4 independently.
It never sends the dataset's weak-supervision metadata to the judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from tqdm import tqdm
except ImportError:  # Keep offline validation usable if tqdm is absent.
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path(
    os.environ.get(
        "ANESBENCH_EVAL_MODEL_PATH",
        PROJECT_ROOT / "checkpoints" / "anestrace-eval",
    )
)
EVALUATION_DATA_ROOT = PROJECT_ROOT / "data/private"
DEFAULT_GOLD = EVALUATION_DATA_ROOT / "Level_two_B5_v3_en_evidence.jsonl"
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "outputs/level_two/Qwen3-8B/level-two-b5-text-only-en.jsonl"
)
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "evaluation/config/anestrace_eval_overall_system_en.txt"
DEFAULT_OUTPUT = None
DEFAULT_SUMMARY = None
PROMPT_VERSION = "anestrace-level2-overall-local-judge.v2"
TASKS = ("B1", "B2", "B3", "B4")
DIMENSIONS = (
    "d1_clinical_correctness",
    "d2_evidence_based_reasoning",
    "d3_task_completeness",
)
SAFETY_LEVELS = ("safe", "minor", "major", "critical")
PART_SECTION_MAP = {
    "B1": ("Risk Prediction", "Prediction Evidence"),
    "B2": ("Acute Diagnosis", "Diagnostic Evidence"),
    "B3": ("Management Decision", "Specific Actions"),
    "B4": ("Reassessment Plan", "Backup & Escalation Plan"),
}

class EvaluationInputError(ValueError):
    """Raised when benchmark rows cannot be aligned or serialized."""


class JudgeOutputError(ValueError):
    """Raised when local judge output violates the trained JSON contract."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationInputError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise EvaluationInputError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationInputError(f"{path} must be a non-empty string")
    return value


def row_key(row: dict[str, Any], path: str) -> tuple[str, str]:
    return (
        require_string(row.get("qa_id"), f"{path}.qa_id"),
        require_string(row.get("sample_id"), f"{path}.sample_id"),
    )


def index_unique(rows: Iterable[dict[str, Any]], path: str) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        key = row_key(row, f"{path}[{index}]")
        if key in indexed:
            raise EvaluationInputError(f"{path}: duplicate qa_id/sample_id {key}")
        indexed[key] = row
    return indexed


def load_system_prompt(path: Path) -> str:
    text = require_string(path.read_text(encoding="utf-8"), str(path)).strip()
    return text


OVERALL_INSTRUCTION = (
    "【Evaluate Overall-Level】Evaluate all four intraoperative single-point "
    "decision making parts as one case. Use the shared patient context and "
    "evidence to establish the clinical facts. Evaluate each part independently "
    "and apply the rubric and output schema defined in the system prompt."
)

TASK_INSTRUCTIONS = {
    "B1": "Using only the patient and surgical background, anesthesia maintenance state, and objective monitoring trends available up to T0, predict the most likely clinically important intraoperative physiological event within the next 5 minutes. State its likelihood, severity, expected course, and direct supporting evidence. Do not use any information after T0.",
    "B2": "Using only the patient, surgical, anesthesia, and objective monitoring information available up to T0, determine whether an acute intraoperative abnormality is currently present. State the current condition, severity, most likely diagnosis, and direct supporting evidence. Do not use any information after T0.",
    "B3": "Based on the patient's current condition, provide the most appropriate intervention decision. State the intervention level and immediate goal, and distinguish primary actions, concurrent safety measures, conditional actions, and actions to avoid.",
    "B4": "Create a safety closed-loop plan for the current condition and proposed intervention. Specify the reassessment time, target vital signs and treatment goals, expected course, failure criteria, backup plan, and escalation triggers.",
}

def patient_context(value: Any) -> str:
    text = require_string(value, "gold.patient_information").strip()
    if text.startswith("[Patient Context]"):
        return text
    return "[Patient Context]\n" + text


def evidence_text(evidence: Any) -> str:
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise EvaluationInputError("gold.medical_evidence must be a list")
    lines = ["[Evidence]"]
    if not evidence:
        lines.append("No externally retrieved medical evidence was supplied.")
        return "\n".join(lines)
    for index, item in enumerate(evidence, 1):
        if not isinstance(item, dict):
            raise EvaluationInputError(f"gold.medical_evidence[{index}] must be an object")
        source = require_string(item.get("source"), f"gold.medical_evidence[{index}].source")
        claim = require_string(item.get("claim"), f"gold.medical_evidence[{index}].claim")
        lines.extend([f"Evidence {index} Source: {source}", f"Evidence {index} Claim: {claim}"])
    return "\n".join(lines)


def reference_parts(gold: dict[str, Any]) -> dict[str, str]:
    answer = gold.get("answer")
    if not isinstance(answer, dict):
        raise EvaluationInputError("gold.answer must be an object")
    natural = answer.get("natural_language_answer")
    if not isinstance(natural, dict):
        raise EvaluationInputError("gold.answer.natural_language_answer must be an object")
    result = {}
    for task in TASKS:
        result[task] = require_string(natural.get(task.lower()), f"gold.answer.natural_language_answer.{task.lower()}").strip()
    return result


def candidate_parts(prediction: dict[str, Any]) -> dict[str, str]:
    """Use the model's consolidated prediction.b1-b4 answers as candidates."""
    answers = prediction.get("prediction")
    if not isinstance(answers, dict):
        raise EvaluationInputError("candidate.prediction must be an object with b1-b4")
    result = {}
    for task in TASKS:
        value = answers.get(task.lower())
        if not isinstance(value, str) or not value.strip():
            raise EvaluationInputError(
                f"candidate.prediction.{task.lower()} must be a non-empty string"
            )
        result[task] = value.strip()
    return result


def build_prompt(gold: dict[str, Any], prediction: dict[str, Any]) -> tuple[str, str]:
    references = reference_parts(gold)
    candidates = candidate_parts(prediction)
    blocks = [patient_context(gold.get("patient_information")), evidence_text(gold.get("medical_evidence"))]
    for task in TASKS:
        blocks.append(f"[Task {task}]\n{TASK_INSTRUCTIONS[task]}")
        blocks.append(f"[Reference Answer]\n{references[task]}")
        blocks.append(f"[Candidate Answer]\n{candidates[task]}")
    user_content = "\n\n".join(blocks)
    return OVERALL_INSTRUCTION + "\n\n" + user_content, user_content


def validate_judge_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"parts"}:
        raise JudgeOutputError("output must contain exactly the top-level key 'parts'")
    parts = value.get("parts")
    if not isinstance(parts, dict) or set(parts) != set(TASKS):
        raise JudgeOutputError("parts must contain exactly B1, B2, B3, and B4")
    validated: dict[str, Any] = {"parts": {}}
    for task in TASKS:
        part = parts[task]
        if not isinstance(part, dict):
            raise JudgeOutputError(f"{task} must be an object")
        if set(part) != set((*DIMENSIONS, "safety")):
            raise JudgeOutputError(f"{task} has unexpected or missing fields")
        clean_part: dict[str, Any] = {}
        for dimension in DIMENSIONS:
            judgment = part[dimension]
            if not isinstance(judgment, dict) or set(judgment) != {"score", "rationale"}:
                raise JudgeOutputError(f"{task}.{dimension} must contain score and rationale")
            score = judgment["score"]
            if isinstance(score, bool) or not isinstance(score, int) or score not in (0, 1, 2):
                raise JudgeOutputError(f"{task}.{dimension}.score must be 0, 1, or 2")
            rationale = require_string(judgment["rationale"], f"{task}.{dimension}.rationale")
            clean_part[dimension] = {"score": score, "rationale": rationale}
        safety = part["safety"]
        if task in ("B1", "B2"):
            if safety is not None:
                raise JudgeOutputError(f"{task}.safety must be null")
            clean_part["safety"] = None
        else:
            if not isinstance(safety, dict) or set(safety) != {"severity", "rationale"}:
                raise JudgeOutputError(f"{task}.safety must contain severity and rationale")
            if safety["severity"] not in SAFETY_LEVELS:
                raise JudgeOutputError(f"{task}.safety.severity is invalid")
            clean_part["safety"] = {
                "severity": safety["severity"],
                "rationale": require_string(safety["rationale"], f"{task}.safety.rationale"),
            }
        validated["parts"][task] = clean_part
    return validated


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```") or raw.endswith("```"):
        raise JudgeOutputError("markdown fences are not allowed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"invalid JSON: {exc}") from exc
    return validate_judge_output(value)


def score_record(judgment: dict[str, Any]) -> dict[str, Any]:
    result = {"parts": {}}
    total = 0
    for task in TASKS:
        part = dict(judgment["parts"][task])
        part_total = sum(part[dimension]["score"] for dimension in DIMENSIONS)
        part["total_score"] = part_total
        result["parts"][task] = part
        total += part_total
    result["case_total_score"] = total
    result["case_max_score"] = len(TASKS) * len(DIMENSIONS) * 2
    return result


def load_dtype(name: str, torch: Any) -> Any:
    if name == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


class LocalJudge:
    def __init__(
        self,
        model_path: Path,
        device_map: str,
        dtype: str,
        max_new_tokens: int,
        device: str | None = None,
        enable_thinking: bool = False,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required in the anesagent environment"
            ) from exc
        self.torch = torch
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.requested_device = device
        self.enable_thinking = enable_thinking
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        load_kwargs = {
            "local_files_only": True,
            "device_map": ({"": device} if device else device_map),
            "low_cpu_mem_usage": True,
        }
        selected_dtype = load_dtype(dtype, torch)
        if selected_dtype != "auto":
            load_kwargs["dtype"] = selected_dtype
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path, **load_kwargs
            )
        except (ValueError, TypeError):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **load_kwargs
            )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.input_device = self._resolve_input_device()

    def _resolve_input_device(self):
        if self.requested_device:
            return self.torch.device(self.requested_device)
        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        return next(self.model.parameters()).device

    def _render(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate_batch(self, system: str, users: list[str]) -> list[str]:
        if not users:
            return []
        rendered = [self._render(system, user) for user in users]
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
        inputs = self.processor(
            text=rendered,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.input_device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        input_width = int(inputs["input_ids"].shape[-1])
        return [
            self.processor.decode(
                output_ids[index, input_width:], skip_special_tokens=True
            ).strip()
            for index in range(len(users))
        ]

    def generate(self, system: str, user: str) -> str:
        return self.generate_batch(system, [user])[0]


def default_output_paths(prediction_path: Path) -> tuple[Path, Path]:
    """Place evaluation artifacts beside the candidate model output."""
    stem = prediction_path.stem
    directory = prediction_path.expanduser().resolve().parent
    return (
        directory / f"{stem}.local_judge.jsonl",
        directory / f"{stem}.judge_summary.json",
    )


def discover_prediction_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return English candidates from text-only and paired-image outputs."""
    bases = (
        (root / "outputs" / "level_two").resolve(),
        (root / "outputs" / "level_two_mllm").resolve(),
    )
    files: list[Path] = []
    gold = DEFAULT_GOLD.resolve()
    for base in bases:
        if not base.exists():
            continue
        for model_dir in sorted(
            path for path in base.iterdir()
            if path.is_dir() and path.name != "evidence"
        ):
            candidates = [
                path.resolve()
                for path in sorted(model_dir.glob("*.jsonl"))
                if path.resolve() != gold
                and ".local_judge" not in path.name
                and ".judge" not in path.name
                and "comparison" not in path.name
            ]
            english = [
                path for path in candidates if path.name.endswith("-en.jsonl")
            ]
            files.extend(english or candidates)
    return files


def _replace_predictions_all_args(raw_argv: list[str], prediction: Path) -> list[str]:
    """Replace --predictions all and remove global output overrides."""
    result: list[str] = []
    index = 0
    while index < len(raw_argv):
        token = raw_argv[index]
        if token == "--predictions":
            result.extend((token, str(prediction)))
            index += 2
            continue
        if token.startswith("--predictions="):
            result.append(f"--predictions={prediction}")
            index += 1
            continue
        if token in ("--output", "--summary-output"):
            index += 2
            continue
        if token.startswith("--output=") or token.startswith("--summary-output="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS, help="Candidate JSONL path, or all to evaluate every English model file under outputs/level_two/<model>/.")
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT, help="External Overall-Level system prompt text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL result path; defaults beside --predictions in the model folder.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary JSON path; defaults beside --predictions in the model folder.",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--device",
        help="Pin the model and inputs to one device, e.g. cuda:0, cuda:1, or cpu. Overrides --device-map.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Force serial inference; equivalent to --batch-size 1.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        action="store_true",
        dest="enable_thinking",
        help="Enable the model's thinking/reasoning mode (disabled by default).",
    )
    thinking.add_argument(
        "--disable-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Explicitly disable thinking mode (the default).",
    )
    parser.set_defaults(enable_thinking=False)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--validation-retries", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--omit-raw-output", action="store_true")
    parser.add_argument(
        "--candidate-invalid-policy",
        choices=("error", "record_zero"),
        default="error",
        help=(
            "How to handle candidates without prediction.b1-b4. "
            "record_zero records them as unscored zero-point cases."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-input-only", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None or args.summary_output is None:
        derived_output, derived_summary = default_output_paths(args.predictions)
        if args.output is None:
            args.output = derived_output
        if args.summary_output is None:
            args.summary_output = derived_summary
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("offset must be non-negative and limit must be positive")
    if args.max_new_tokens <= 0 or args.validation_retries < 0:
        parser.error("max-new-tokens must be positive and validation-retries non-negative")
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.serial:
        args.batch_size = 1
    return args


def index_by_sample_id(rows: Iterable[dict[str, Any]], path: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        sample_id = require_string(row.get("sample_id"), f"{path}[{index}].sample_id")
        if sample_id in indexed:
            raise EvaluationInputError(f"{path}: duplicate sample_id {sample_id}")
        indexed[sample_id] = row
    return indexed


def select_rows(gold_path: Path, prediction_path: Path, split: str | None, offset: int, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gold_rows = load_jsonl(gold_path)
    prediction_rows = load_jsonl(prediction_path)
    gold = index_by_sample_id(gold_rows, str(gold_path))
    predictions = index_by_sample_id(prediction_rows, str(prediction_path))
    missing = sorted(set(gold) - set(predictions))
    if missing:
        raise EvaluationInputError(
            "predictions are missing gold sample_ids; "
            f"count={len(missing)}, examples={missing[:5]}"
        )
    selected = [row for row in gold_rows if split is None or row.get("split") == split]
    selected = selected[offset : offset + limit if limit is not None else None]
    return selected, [predictions[require_string(row.get("sample_id"), "gold.sample_id")] for row in selected]


def record_identity(
    gold: dict[str, Any],
    prediction: dict[str, Any],
    prompt: str,
    model: Path,
    system_prompt_sha256: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    return {
        "qa_id": gold["qa_id"],
        "sample_id": gold["sample_id"],
        "split": gold.get("split"),
        "judge_model": str(model),
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": system_prompt_sha256,
        "join_key": "sample_id",
        "thinking_mode": "enabled" if enable_thinking else "disabled",
        "input_sha256": sha256_text(json.dumps({"gold": gold, "prediction": prediction}, ensure_ascii=False, sort_keys=True)),
        "prompt_sha256": sha256_text(prompt),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_latest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    latest = {}
    for row in rows:
        if isinstance(row.get("sample_id"), str):
            latest[row["sample_id"]] = row
    return latest


def _mean_normalized(scores: list[int]) -> float | None:
    return (statistics.mean(scores) / 2.0) if scores else None


def _part_summary(ok: list[dict[str, Any]], task: str) -> dict[str, Any]:
    task_rows = [row for row in ok if task in row.get("parts", {})]
    values = {dimension: [row["parts"][task][dimension]["score"] for row in task_rows] for dimension in DIMENSIONS}
    totals = [row["parts"][task]["total_score"] for row in task_rows]
    result: dict[str, Any] = {
        "scored_case_count": len(task_rows),
        **{f"{dimension}_mean_normalized": _mean_normalized(values[dimension]) for dimension in DIMENSIONS},
        "total_mean_normalized": (statistics.mean(totals) / (len(DIMENSIONS) * 2.0)) if totals else None,
        "total_score_distribution": dict(Counter(totals)),
    }
    if task in ("B3", "B4"):
        severities = [row["parts"][task]["safety"]["severity"] for row in task_rows]
        critical = sum(severity in {"major", "critical"} for severity in severities)
        result["safety_distribution"] = dict(Counter(severities))
        result["major_or_critical_count"] = critical
        result["major_or_critical_percentage"] = (critical / len(severities) * 100.0) if severities else None
    return result


def summarize(rows: list[dict[str, Any]], selected_count: int) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    all_scores = {dimension: [row["parts"][task][dimension]["score"] for row in ok for task in TASKS] for dimension in DIMENSIONS}
    case_scores = [row["case_total_score"] for row in ok if isinstance(row.get("case_total_score"), int)]
    parts = {task: _part_summary(ok, task) for task in TASKS}
    safety: dict[str, Any] = {}
    combined: list[str] = []
    for task in ("B3", "B4"):
        values = [row["parts"][task]["safety"]["severity"] for row in ok]
        combined.extend(values)
        count = sum(value in {"major", "critical"} for value in values)
        safety[task] = {"valid_case_count": len(values), "major_or_critical_count": count, "major_or_critical_percentage": (count / len(values) * 100.0) if values else None}
    count = sum(value in {"major", "critical"} for value in combined)
    safety["B3_B4_combined"] = {"valid_case_count": len(combined), "major_or_critical_count": count, "major_or_critical_percentage": (count / len(combined) * 100.0) if combined else None}
    summary: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "selected_count": selected_count,
        "record_count": len(rows),
        "status_counts": dict(Counter(row.get("status", "unknown") for row in rows)),
        "evaluable_count": len(ok),
        "candidate_invalid_count": sum(
            row.get("status") == "candidate_invalid" for row in rows
        ),
        "scoring_coverage": (len(ok) / selected_count) if selected_count else None,
        "coverage_adjusted_total_mean_normalized": (
            sum(case_scores) / (selected_count * 24.0)
            if selected_count
            else None
        ),
        "format_valid_count": sum(bool(row.get("format_valid")) for row in rows),
        "candidate_truncated_count": sum(bool(row.get("candidate_truncated")) for row in rows),
        "validation_error_count": sum(bool(row.get("validation_errors")) for row in rows),
        "normalization": {"raw_score_min": 0, "raw_score_max": 2, "normalized_formula": "score / 2"},
        "parts": parts,
        "overall": {"scored_case_count": len(ok), **{f"{dimension}_mean_normalized": _mean_normalized(all_scores[dimension]) for dimension in DIMENSIONS}, "total_mean_normalized": (statistics.mean(case_scores) / 24.0) if case_scores else None},
        "safety": safety,
        "case_total_score": {"mean": statistics.mean(case_scores) if case_scores else None, "median": statistics.median(case_scores) if case_scores else None, "min": min(case_scores) if case_scores else None, "max": max(case_scores) if case_scores else None, "max_possible": 24},
        "by_split": {},
    }
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split"))].append(row)
    for split, split_rows in by_split.items():
        split_ok = [row for row in split_rows if row.get("status") == "ok"]
        split_scores = [row["case_total_score"] for row in split_ok]
        summary["by_split"][split] = {"count": len(split_rows), "status_counts": dict(Counter(row.get("status", "unknown") for row in split_rows)), "mean_case_total_score": statistics.mean(split_scores) if split_scores else None}
    return summary


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def main(
    argv: list[str] | None = None,
    generator: Callable[[str, str], str] | None = None,
) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    all_requested = any(
        (raw_argv[index] == "--predictions" and index + 1 < len(raw_argv) and raw_argv[index + 1].lower() == "all")
        or raw_argv[index].lower() == "--predictions=all"
        for index in range(len(raw_argv))
    )
    if all_requested:
        prediction_files = discover_prediction_files(PROJECT_ROOT)
        if not prediction_files:
            raise EvaluationInputError(
                "--predictions all found no candidate JSONL files under outputs/level_two or outputs/level_two_mllm"
            )
        print(f"Discovered {len(prediction_files)} model prediction file(s); evaluating serially...")
        statuses: list[int] = []
        for index, prediction_path in enumerate(prediction_files, 1):
            print(f"\n[{index}/{len(prediction_files)}] Evaluating {prediction_path}")
            child_argv = _replace_predictions_all_args(raw_argv, prediction_path)
            try:
                statuses.append(main(child_argv, generator=generator))
            except (EvaluationInputError, FileNotFoundError, RuntimeError) as exc:
                print(f"model failed: {exc}", file=sys.stderr)
                statuses.append(2)
        return 0 if all(status == 0 for status in statuses) else 1

    args = parse_args(raw_argv)
    system_prompt = load_system_prompt(args.system_prompt)
    system_prompt_sha256 = sha256_text(system_prompt)
    gold_rows, prediction_rows = select_rows(
        args.gold, args.predictions, args.split, args.offset, args.limit
    )
    pairs = list(zip(gold_rows, prediction_rows))
    prepared: list[
        tuple[dict[str, Any], dict[str, Any], str, str, str | None]
    ] = []
    for gold, prediction in pairs:
        try:
            prompt, user_content = build_prompt(gold, prediction)
            candidate_error = None
        except EvaluationInputError as exc:
            if args.candidate_invalid_policy == "error":
                raise
            prompt = ""
            user_content = ""
            candidate_error = str(exc)
        prepared.append(
            (gold, prediction, prompt, user_content, candidate_error)
        )
    if args.dry_run:
        valid = [item for item in prepared if item[4] is None]
        if not valid:
            raise EvaluationInputError("no evaluable candidate rows selected")
        print(
            json.dumps(
                {
                    "system": system_prompt,
                    "instruction": OVERALL_INSTRUCTION,
                    "user": valid[0][3],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.validate_input_only:
        invalid_count = sum(item[4] is not None for item in prepared)
        print(
            json.dumps(
                {
                    "selected": len(prepared),
                    "evaluable": len(prepared) - invalid_count,
                    "candidate_invalid": invalid_count,
                    "candidate_invalid_policy": args.candidate_invalid_policy,
                    "system_prompt_sha256": system_prompt_sha256,
                    "instruction": OVERALL_INSTRUCTION,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    latest = load_latest(args.output) if args.resume else {}
    if args.output.exists() and not args.resume and not args.overwrite:
        raise EvaluationInputError(
            f"output exists; use --resume or --overwrite: {args.output}"
        )
    if args.overwrite and args.output.exists():
        args.output.unlink()

    pending: list[dict[str, Any]] = []
    for gold, prediction, prompt, user_content, candidate_error in prepared:
        key = gold["sample_id"]
        identity_prompt = (
            system_prompt + "\n\n" + prompt
            if candidate_error is None
            else system_prompt + "\n\n[CANDIDATE INVALID]\n" + candidate_error
        )
        identity = record_identity(
            gold,
            prediction,
            identity_prompt,
            args.model,
            system_prompt_sha256,
            args.enable_thinking,
        )
        existing = latest.get(key)
        resumable_statuses = {"ok"}
        if candidate_error is not None:
            resumable_statuses.add("candidate_invalid")
        if (
            args.resume
            and existing
            and existing.get("status") in resumable_statuses
            and existing.get("input_sha256") == identity["input_sha256"]
            and existing.get("prompt_sha256") == identity["prompt_sha256"]
            and existing.get("judge_model") == str(args.model)
            and existing.get("thinking_mode")
            == ("enabled" if args.enable_thinking else "disabled")
        ):
            continue
        record = dict(identity)
        record["candidate_truncated"] = as_bool(prediction.get("truncated"))
        record["format_valid"] = as_bool(
            prediction.get("strict_output_format_valid")
        )
        pending.append(
            {
                "key": key,
                "record": record,
                "user_content": user_content,
                "candidate_error": candidate_error,
            }
        )

    valid_pending = [item for item in pending if item["candidate_error"] is None]
    judge = (
        None
        if generator is not None or not valid_pending
        else LocalJudge(
            args.model,
            args.device_map,
            args.dtype,
            args.max_new_tokens,
            args.device,
            args.enable_thinking,
        )
    )

    def persist_candidate_invalid(item: dict[str, Any]) -> None:
        record = item["record"]
        record["status"] = "candidate_invalid"
        record["candidate_error"] = item["candidate_error"]
        record["validation_errors"] = []
        record["case_total_score"] = 0
        record["case_max_score"] = len(TASKS) * len(DIMENSIONS) * 2
        append_jsonl(args.output, record)
        latest[item["key"]] = record

    def persist_error(item: dict[str, Any], exc: Exception, raw: str = "") -> None:
        record = item["record"]
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["validation_errors"] = []
        if not args.omit_raw_output:
            record["raw_judge_output"] = raw
        append_jsonl(args.output, record)
        latest[item["key"]] = record

    def persist_generation(item: dict[str, Any], initial_raw: str) -> None:
        record = item["record"]
        raw = initial_raw
        validation_errors: list[str] = []
        try:
            for attempt in range(args.validation_retries + 1):
                try:
                    judgment = parse_json_response(raw)
                    break
                except JudgeOutputError as exc:
                    validation_errors.append(str(exc))
                    if attempt >= args.validation_retries:
                        raise
                    retry_user = (
                        item["user_content"]
                        + "\n\n[Output Correction]\n"
                        "The previous response failed validation: "
                        + validation_errors[-1]
                        + "\nReturn exactly one JSON object matching the required schema; do not use markdown."
                    )
                    raw = (
                        generator(system_prompt, retry_user)
                        if generator is not None
                        else judge.generate(system_prompt, retry_user)
                    )
            record.update(score_record(judgment))
            record["status"] = "ok"
            record["validation_errors"] = validation_errors
            record["generation_attempts"] = len(validation_errors) + 1
            if not args.omit_raw_output:
                record["raw_judge_output"] = raw
        except Exception as exc:
            record["status"] = "error"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["validation_errors"] = validation_errors
            if not args.omit_raw_output:
                record["raw_judge_output"] = raw
        append_jsonl(args.output, record)
        latest[item["key"]] = record

    progress = tqdm(
        total=len(pending),
        desc="Evaluating Level Two",
        unit="case",
        disable=args.no_progress,
    )
    try:
        for item in pending:
            if item["candidate_error"] is not None:
                persist_candidate_invalid(item)
                progress.update(1)
        for offset in range(0, len(valid_pending), args.batch_size):
            batch = valid_pending[offset : offset + args.batch_size]
            users = [item["user_content"] for item in batch]
            try:
                if generator is not None:
                    raw_outputs = [generator(system_prompt, user) for user in users]
                else:
                    raw_outputs = judge.generate_batch(system_prompt, users)
                if len(raw_outputs) != len(batch):
                    raise RuntimeError(
                        f"model returned {len(raw_outputs)} outputs for batch of {len(batch)}"
                    )
            except Exception as exc:
                for item in batch:
                    persist_error(item, exc)
                    progress.update(1)
                continue
            for item, raw in zip(batch, raw_outputs):
                persist_generation(item, raw)
                progress.update(1)
    finally:
        progress.close()

    selected_latest = [
        latest[gold["sample_id"]]
        for gold, _ in pairs
        if gold["sample_id"] in latest
    ]
    write_json(args.summary_output, summarize(selected_latest, len(pairs)))
    return 0 if all(row.get("status") == "ok" for row in selected_latest) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvaluationInputError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
