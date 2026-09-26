#!/usr/bin/env python3
"""Run single-pass Level Two B5 decisions with paired monitoring images.

The model receives the static patient context plus the ordered 180-second trend
and 60-second waveform-context images. The reference trend text, answers,
identifiers, review metadata, and local media paths are never placed in model
messages.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from . import run_extended as l2
except ImportError:
    import run_extended as l2
from inference_adapters import (
    ADAPTER_NAMES,
    GenerationRequest,
    create_adapter,
    resolve_adapter_spec,
)


DEFAULT_INPUT = PROJECT_ROOT / "level_two/Level_two_B5_v3_en_evidence.jsonl"
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "ANESBENCH_L2_MLLM_MODEL_PATH",
        str(PROJECT_ROOT / "checkpoints" / "Qwen" / "Qwen3-VL-32B-Instruct"),
    )
)
DEFAULT_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "doc/Level_Two_B5_Multimodal_Prompt_en.md"
)
TREND_SECTION_HEADING = "[Recent Vital-Sign Trends (Past 180 s)]"
EXPECTED_IMAGE_TYPES = ("trend", "waveform_context")
PROMPT_VERSION = "anesbench-l2-b5-paired-images-en.v1"
OUTPUT_SCHEMA_VERSION = "anesbench-level-two-b5-multimodal-prediction.v1"
SYSTEM_PROMPT = """You are an AI system performing intraoperative anesthesia decision-making from patient context and two monitoring images. Use only the information visible in the user message and images. Do not use sample identifiers, filenames, paths, reference answers, or unavailable information. Output only the eight required clinical sections, without hidden reasoning, a preamble, code fences, or JSON."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def memory_limit(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:GiB|MiB|GB|MB)", value):
        raise argparse.ArgumentTypeError(
            "must be a positive memory value such as 70GiB or 24000MiB"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run English Level Two B5 inference with static context and paired "
            "vital-sign images."
        )
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--model-adapter", choices=ADAPTER_NAMES, default="auto"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Prediction JSONL. Defaults to outputs/level_two_mllm/"
            "<model>/level-two-b5-paired-images-en.jsonl."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=l2.SUPPORTED_SPLITS,
        help="Keep one split; repeat to keep several (default: all).",
    )
    parser.add_argument("--offset", type=non_negative_int, default=0)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--max-new-tokens", type=positive_int, default=3072)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-per-gpu", type=memory_limit)
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--temperature", type=non_negative_float, default=0.0)
    parser.add_argument("--top-p", type=probability, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE
    )
    parser.add_argument("--video-num-frames", type=positive_int, default=16)
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "decord", "opencv", "torchvision", "torchcodec"),
        default="pyav",
    )
    parser.add_argument(
        "--llava-video-backend", choices=("frames", "codec"), default="frames"
    )
    parser.add_argument("--video-max-pixels", type=positive_int, default=200704)
    parser.add_argument("--vision-input-size", type=positive_int, default=448)
    parser.add_argument("--image-max-tiles", type=positive_int, default=12)
    parser.add_argument(
        "--video-max-tiles-per-frame", type=positive_int, default=1
    )
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate records, prompts, and both image paths without loading a model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retry only prior inference errors; retain format-invalid model "
            "responses (default: true)."
        ),
    )
    args = parser.parse_args(argv)
    if args.output is None:
        model_name = args.model_path.expanduser().resolve().name
        args.output = (
            PROJECT_ROOT
            / "outputs"
            / "level_two_mllm"
            / model_name
            / "level-two-b5-paired-images-en.jsonl"
        )
    return args


def load_prompt_template(path: Path) -> tuple[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"prompt template does not exist: {resolved}")
    template = resolved.read_text(encoding="utf-8")
    count = template.count("{{patient_information}}")
    if count != 1:
        raise ValueError(
            "prompt template must contain exactly one "
            f"{{{{patient_information}}}} placeholder, found {count}"
        )
    return template, hashlib.sha256(template.encode("utf-8")).hexdigest()


def strip_vital_trend_section(patient_information: str) -> str:
    if not isinstance(patient_information, str) or not patient_information.strip():
        raise ValueError("patient_information must be a non-empty string")
    count = patient_information.count(TREND_SECTION_HEADING)
    if count != 1:
        raise ValueError(
            "patient_information must contain exactly one "
            f"{TREND_SECTION_HEADING!r} heading, found {count}"
        )
    static_context, trend_text = patient_information.split(
        TREND_SECTION_HEADING, 1
    )
    static_context = static_context.rstrip()
    if not static_context:
        raise ValueError("patient context before the vital-sign section is empty")
    if not trend_text.strip():
        raise ValueError("vital-sign trend section is empty")
    return static_context


def validate_record(record: dict[str, Any], where: str = "record") -> None:
    l2.validate_record(record, where)
    strip_vital_trend_section(record["patient_information"])
    media = record.get("media")
    if not isinstance(media, dict):
        raise ValueError(f"{where}: media must be an object")
    if media.get("input_mode") != "paired_images":
        raise ValueError(f"{where}: media.input_mode must be 'paired_images'")
    images = media.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{where}: media.images must be a list")
    found: dict[str, int] = {}
    for image in images:
        if not isinstance(image, dict):
            raise ValueError(f"{where}: every media image must be an object")
        image_type = image.get("image_type")
        if image_type in EXPECTED_IMAGE_TYPES:
            found[image_type] = found.get(image_type, 0) + 1
    for image_type in EXPECTED_IMAGE_TYPES:
        if found.get(image_type, 0) != 1:
            raise ValueError(
                f"{where}: expected exactly one {image_type!r} image, "
                f"found {found.get(image_type, 0)}"
            )


def load_items(path: Path) -> list[l2.WorkItem]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {resolved}")
    items: list[l2.WorkItem] = []
    seen: dict[str, int] = {}
    for line_number, record in l2.read_jsonl(resolved):
        validate_record(record, f"{resolved}:{line_number}")
        qa_id = record["qa_id"]
        if qa_id in seen:
            raise ValueError(
                f"duplicate qa_id {qa_id!r} at lines {seen[qa_id]} and {line_number}"
            )
        seen[qa_id] = line_number
        items.append(l2.WorkItem(resolved, line_number, record))
    return items


def resolve_paired_images(
    record: dict[str, Any], project_root: Path = PROJECT_ROOT
) -> tuple[Path, Path]:
    validate_record(record)
    by_type = {
        image["image_type"]: image
        for image in record["media"]["images"]
        if image.get("image_type") in EXPECTED_IMAGE_TYPES
    }
    paths: list[Path] = []
    for image_type in EXPECTED_IMAGE_TYPES:
        raw_path = by_type[image_type].get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"media image {image_type!r} requires a path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{image_type} image does not exist: {path}")
        paths.append(path)
    return paths[0], paths[1]


def build_user_prompt(record: dict[str, Any], template: str) -> str:
    static_context = strip_vital_trend_section(record["patient_information"])
    return template.replace("{{patient_information}}", static_context, 1)


def build_generation_request(
    item: l2.WorkItem, template: str
) -> GenerationRequest:
    media_paths = resolve_paired_images(item.record)
    return GenerationRequest(
        media_type="image",
        media_path=media_paths[0],
        media_paths=media_paths,
        system_prompt=SYSTEM_PROMPT,
        question_prompt=build_user_prompt(item.record, template),
    )


def is_truncated(generated: Any, max_new_tokens: int) -> bool:
    count = generated.generated_tokens
    return isinstance(count, int) and count >= max_new_tokens


def successful_result(
    item: l2.WorkItem,
    args: argparse.Namespace,
    adapter: Any,
    generated: Any,
    prompt_hash: str,
    elapsed: float,
    requested_batch_size: int,
) -> dict[str, Any]:
    parsed = l2.parse_model_response(generated.text, "en")
    truncated = is_truncated(generated, args.max_new_tokens)
    format_errors = list(parsed.format_errors)
    if truncated:
        format_errors.append("generation reached max_new_tokens without EOS")
    response_valid = parsed.prediction is not None and not truncated
    strict_valid = parsed.strict_output_format_valid and not truncated
    status = "ok" if response_valid else "invalid_response"
    native_batch = adapter.adapter_name == "transformers_vlm"
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "qa_id": item.record["qa_id"],
        "sample_id": item.record["sample_id"],
        "task_group": item.record.get("task_group"),
        "task_name": item.record["task_name"],
        "answer_type": item.record["answer_type"],
        "split": item.record["split"],
        "language": "en",
        "model_path": str(args.model_path),
        "model_type": adapter.model_type,
        "model_adapter": adapter.adapter_name,
        "input_mode": "paired_images",
        "media_included": True,
        "media_types": list(EXPECTED_IMAGE_TYPES),
        "vital_trend_text_included": False,
        "batch_size": args.batch_size,
        "batch_size_requested_for_call": requested_batch_size,
        "batch_size_effective": requested_batch_size if native_batch else 1,
        "batch_execution": (
            "native_multimodal_batch" if native_batch else "sequential_multi_image"
        ),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "status": status,
        "prediction_text": generated.text,
        "final_answer_text": parsed.final_answer_text,
        "prediction": parsed.prediction,
        "prediction_sections": parsed.prediction_sections,
        "prediction_extraction_method": parsed.extraction_method,
        "strict_output_format_valid": strict_valid,
        "response_format_valid": response_valid,
        "format_errors": format_errors,
        "prediction_error": parsed.prediction_error,
        "truncated": truncated,
        "input_tokens": generated.input_tokens,
        "generated_tokens": generated.generated_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def error_result(
    item: l2.WorkItem,
    args: argparse.Namespace,
    spec: Any,
    exc: Exception,
    prompt_hash: str,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "qa_id": item.record["qa_id"],
        "sample_id": item.record["sample_id"],
        "task_group": item.record.get("task_group"),
        "task_name": item.record["task_name"],
        "answer_type": item.record["answer_type"],
        "split": item.record["split"],
        "language": "en",
        "model_path": str(args.model_path),
        "model_type": spec.model_type,
        "model_adapter": spec.adapter_name,
        "input_mode": "paired_images",
        "media_included": True,
        "media_types": list(EXPECTED_IMAGE_TYPES),
        "vital_trend_text_included": False,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "elapsed_seconds": round(elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_one(
    item: l2.WorkItem,
    args: argparse.Namespace,
    adapter: Any,
    template: str,
    prompt_hash: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    l2._seed_for_item(args.seed, item.record["qa_id"])
    generated = adapter.generate(build_generation_request(item, template))
    return successful_result(
        item,
        args,
        adapter,
        generated,
        prompt_hash,
        time.perf_counter() - started,
        1,
    )


def run_many(
    items: list[l2.WorkItem],
    args: argparse.Namespace,
    adapter: Any,
    template: str,
    prompt_hash: str,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    l2._seed_for_item(
        args.seed, "\0".join(item.record["qa_id"] for item in items)
    )
    requests = [build_generation_request(item, template) for item in items]
    generated = list(adapter.generate_batch(requests))
    if len(generated) != len(items):
        raise RuntimeError(
            f"adapter returned {len(generated)} outputs for {len(items)} requests"
        )
    elapsed = (time.perf_counter() - started) / len(items)
    return [
        successful_result(
            item,
            args,
            adapter,
            result,
            prompt_hash,
            elapsed,
            len(items),
        )
        for item, result in zip(items, generated)
    ]


def acquire_output_lock(path: Path) -> Any:
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"prediction output is already being written: {path}"
        ) from exc
    return handle


def write_run_config(
    args: argparse.Namespace,
    spec: Any,
    selected_count: int,
    pending_count: int,
    prompt_hash: str,
) -> None:
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "language": "en",
        "model_path": str(args.model_path),
        "model_adapter_requested": args.model_adapter,
        "model_adapter": spec.adapter_name,
        "model_type": spec.model_type,
        "model_architectures": list(spec.architectures),
        "input": str(args.input),
        "output": str(args.output),
        "splits": args.split or list(l2.SUPPORTED_SPLITS),
        "offset": args.offset,
        "limit": args.limit,
        "selected_count_before_resume": selected_count,
        "pending_count": pending_count,
        "input_mode": "paired_images",
        "media_included": True,
        "media_types": list(EXPECTED_IMAGE_TYPES),
        "image_order": ["trend_180s", "waveform_context_60s"],
        "vital_trend_text_included": False,
        "output_persistence": "latest_per_qa_id_atomic",
        "retry_policy": "errors_only",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "attn_implementation": args.attn_implementation,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSION,
        "prompt_template": str(args.prompt_template),
        "prompt_sha256": prompt_hash,
    }
    path = args.output.with_suffix(args.output.suffix + ".config.json")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_inference_locked(args: argparse.Namespace) -> int:
    template, prompt_hash = load_prompt_template(args.prompt_template)
    all_items = load_items(args.input)
    items = l2.select_items(all_items, args)
    ordered_qa_ids = [item.record["qa_id"] for item in all_items]
    allowed_qa_ids = set(ordered_qa_ids)
    if args.validate_input_only:
        for item in items:
            resolve_paired_images(item.record)
            build_user_prompt(item.record, template)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "selected": len(items),
                    "input_mode": "paired_images",
                    "media_types": list(EXPECTED_IMAGE_TYPES),
                    "vital_trend_text_included": False,
                    "prompt_sha256": prompt_hash,
                },
                indent=2,
            )
        )
        return 0
    spec = resolve_adapter_spec(args.model_path, args.model_adapter)

    if args.overwrite and args.output.exists():
        args.output.unlink()
    latest = l2.load_latest_results(args.output, allowed_qa_ids)
    if args.output.exists():
        l2.write_latest_results(args.output, latest, ordered_qa_ids)
    statuses = {
        qa_id: record["status"]
        for qa_id, record in latest.items()
        if isinstance(record.get("status"), str)
    }
    completed = l2.completed_qa_ids(statuses, args.retry_errors)
    pending = [item for item in items if item.record["qa_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(
        args, spec, selected_count=len(items), pending_count=len(pending),
        prompt_hash=prompt_hash
    )
    if not pending:
        print(
            f"No pending items. Selected={len(items)}, already recorded={len(items)}"
        )
        return 0

    print(
        f"Loading local model {args.model_path} with {spec.adapter_name} "
        f"for {len(pending)} paired-image item(s) "
        f"({len(items) - len(pending)} resumed)...",
        flush=True,
    )
    adapter = create_adapter(args, spec)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "tqdm is required; install requirements-inference.txt"
        ) from exc

    counts = {"ok": 0, "invalid_response": 0, "error": 0}
    batch_fallbacks = 0
    progress = tqdm(
        total=len(pending),
        desc="AnesBench Level Two MLLM",
        unit="item",
        dynamic_ncols=True,
    )
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_started = time.perf_counter()
        try:
            results = run_many(
                batch, args, adapter, template, prompt_hash
            )
        except KeyboardInterrupt:
            raise
        except Exception as batch_exc:
            if len(batch) == 1:
                results = [
                    error_result(
                        batch[0], args, spec, batch_exc, prompt_hash,
                        time.perf_counter() - batch_started
                    )
                ]
            else:
                batch_fallbacks += 1
                progress.write(
                    f"Batch of {len(batch)} failed with "
                    f"{type(batch_exc).__name__}: {batch_exc}; "
                    "retrying items individually."
                )
                torch_module = getattr(adapter, "torch", None)
                if torch_module is not None and torch_module.cuda.is_available():
                    torch_module.cuda.empty_cache()
                results = []
                for item in batch:
                    item_started = time.perf_counter()
                    try:
                        results.append(
                            run_one(
                                item, args, adapter, template, prompt_hash
                            )
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        results.append(
                            error_result(
                                item, args, spec, exc, prompt_hash,
                                time.perf_counter() - item_started
                            )
                        )
        for result in results:
            counts[result["status"]] += 1
            latest[result["qa_id"]] = result
        l2.write_latest_results(args.output, latest, ordered_qa_ids)
        progress.update(len(batch))
        progress.set_postfix(
            ok=counts["ok"],
            invalid=counts["invalid_response"],
            errors=counts["error"],
            batch_fallbacks=batch_fallbacks,
            refresh=False,
        )
    progress.close()
    print(
        f"Finished: ok={counts['ok']}, invalid={counts['invalid_response']}, "
        f"errors={counts['error']}, total_records={len(latest)}, "
        f"batch_fallbacks={batch_fallbacks}, output={args.output}"
    )
    return 0 if counts["error"] == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.input = args.input.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.prompt_template = args.prompt_template.expanduser().resolve()
    if args.validate_input_only:
        return run_inference_locked(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_lock = acquire_output_lock(args.output)
    try:
        return run_inference_locked(args)
    finally:
        output_lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
