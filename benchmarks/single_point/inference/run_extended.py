#!/usr/bin/env python3
"""Run AnesBench Level Two B5 text-only inference.

The default backend is a local Transformers model. With ``--api-provider`` the
same English prompt and output schema are evaluated through an OpenAI-compatible
Gemini/GPT/Claude endpoint. Only ``patient_information`` is inserted into the
model prompt; media, answers, ground truth, review metadata, sample identifiers,
and source paths are never included in model messages.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUPPORTED_LANGUAGES = ("zh", "en")
DEFAULT_INPUTS_BY_LANGUAGE = {
    "zh": PROJECT_ROOT / "data/private/Level_two_B5_v2_zh.jsonl",
    "en": PROJECT_ROOT / "data/private/Level_two_B5_v3_en_evidence.jsonl",
}
DEFAULT_INPUT = DEFAULT_INPUTS_BY_LANGUAGE["zh"]
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "ANESBENCH_L2_MODEL_PATH",
        str(PROJECT_ROOT / "checkpoints" / "Qwen" / "Qwen3-8B"),
    )
)
DEFAULT_PROMPT_TEMPLATES_BY_LANGUAGE = {
    "zh": PROJECT_ROOT / "benchmarks/single_point/prompts/Level_Two_B5_Text_Only_Prompt_zh.md",
    "en": PROJECT_ROOT / "benchmarks/single_point/prompts/Level_Two_B5_Text_Only_Prompt_en.md",
}
DEFAULT_PROMPT_TEMPLATE = DEFAULT_PROMPT_TEMPLATES_BY_LANGUAGE["zh"]
PROMPT_VERSIONS = {
    "zh": "anesbench-l2-b5-text-zh.v1",
    "en": "anesbench-l2-b5-text-en.v1",
}
PROMPT_VERSION = PROMPT_VERSIONS["zh"]
OUTPUT_SCHEMA_VERSION = "anesbench-level-two-b5-prediction.v1"
API_PROVIDER_DEFAULT_MODELS = {
    "gemini": "gemini-3-pro-preview",
    "gpt": "gpt-6-astra",
    "claude": "claude-fable-5",
}
API_PROVIDER_DEFAULT_KEY_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
DEFAULT_API_BASE_URL = "https://www.micuapi.ai/v1"
SUPPORTED_SPLITS = ("train", "validation", "test")
EXPECTED_TASK_NAME = "B5_complete_plan"
EXPECTED_ANSWER_TYPE = "open_ended"
EXPECTED_HEADINGS = (
    "风险预测",
    "预测依据",
    "诊断结论",
    "诊断依据",
    "决策结论",
    "具体动作",
    "复评计划",
    "备用与升级方案",
)
EXPECTED_HEADINGS_EN = (
    "Risk Prediction",
    "Prediction Evidence",
    "Acute Diagnosis",
    "Diagnostic Evidence",
    "Management Decision",
    "Specific Actions",
    "Reassessment Plan",
    "Backup & Escalation Plan",
)
EXPECTED_HEADINGS_BY_LANGUAGE = {
    "zh": EXPECTED_HEADINGS,
    "en": EXPECTED_HEADINGS_EN,
}
SECTION_PAIRS = {
    "b1": ("风险预测", "预测依据"),
    "b2": ("诊断结论", "诊断依据"),
    "b3": ("决策结论", "具体动作"),
    "b4": ("复评计划", "备用与升级方案"),
}
SECTION_PAIRS_EN = {
    "b1": ("Risk Prediction", "Prediction Evidence"),
    "b2": ("Acute Diagnosis", "Diagnostic Evidence"),
    "b3": ("Management Decision", "Specific Actions"),
    "b4": ("Reassessment Plan", "Backup & Escalation Plan"),
}
SECTION_PAIRS_BY_LANGUAGE = {
    "zh": SECTION_PAIRS,
    "en": SECTION_PAIRS_EN,
}
SYSTEM_PROMPTS = {
    "zh": """你是一名执行术中麻醉临床决策任务的人工智能模型。仅使用用户消息中的当前患者信息作答，不得利用样本标识或任何未提供的信息。严格按模板要求输出八个标题及最终临床结论，不输出思维过程、前言、代码块或JSON。""",
    "en": """You are an AI system performing intraoperative anesthesia decision-making. Use only the current patient information in the user message; do not use sample identifiers or any unavailable information. Follow the template exactly and output the eight required sections only, without chain-of-thought, a preamble, code fences, or JSON.""",
}
SYSTEM_PROMPT = SYSTEM_PROMPTS["zh"]

HEADING_LINE_RE = re.compile(r"(?m)^[ \t]*【([^】\r\n]+)】[ \t]*\r?$")
MARKDOWN_HEADING_RE = re.compile(
    r"(?m)^[ \t]*#{1,6}[ \t]+([^\r\n]+?)[ \t]*\r?$"
)
OUTER_FENCE_RE = re.compile(
    r"\A[ \t]*```(?:markdown|md|text)?[ \t]*\r?\n?(.*?)\r?\n?```[ \t]*\Z",
    flags=re.DOTALL | re.IGNORECASE,
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
GPT_OSS_FINAL_MARKER = "<|channel|>final<|message|>"
GPT_OSS_END_MARKERS = ("<|end|>", "<|return|>")
FORBIDDEN_TAIL_STOP_STRINGS_BY_LANGUAGE = {
    "zh": ("\n### 最终临床结论", "\n## 最终临床结论"),
    "en": ("\n### Final Clinical Conclusion", "\n## Final Clinical Conclusion"),
}
MORPHEUS_TAIL_STOP_STRINGS_BY_LANGUAGE = {
    "zh": (
        "\n### 临床决策分析",
        "\n### 最终临床结论",
        "\n#### **风险预测**",
        "\n### 风险预测",
        "\n### 【风险预测】",
        "\n#### 【风险预测】",
    ),
    "en": (
        "\n### Clinical Decision Analysis",
        "\n### Final Clinical Conclusion",
        "\n#### **Risk Prediction**",
        "\n### Risk Prediction",
        "\n### 【Risk Prediction】",
        "\n#### 【Risk Prediction】",
    ),
}
COMMON_FORBIDDEN_TAIL_STOP_STRINGS = ("\n```json", "\n```JSON")
# Backward-compatible aliases for callers that import the original Chinese rules.
FORBIDDEN_TAIL_STOP_STRINGS = (
    *FORBIDDEN_TAIL_STOP_STRINGS_BY_LANGUAGE["zh"],
    *COMMON_FORBIDDEN_TAIL_STOP_STRINGS,
)
MORPHEUS_TAIL_STOP_STRINGS = MORPHEUS_TAIL_STOP_STRINGS_BY_LANGUAGE["zh"]


@dataclass(frozen=True)
class WorkItem:
    source_path: Path
    source_line: int
    record: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    architectures: tuple[str, ...]
    frontend_kind: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    generated_tokens: int
    truncated: bool


@dataclass(frozen=True)
class ParsedResponse:
    final_answer_text: str
    prediction_sections: dict[str, str] | None
    prediction: dict[str, str] | None
    extraction_method: str | None
    strict_output_format_valid: bool
    format_errors: tuple[str, ...]
    prediction_error: str | None


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run text-only Level Two B5 inference locally or with an "
            "OpenAI-compatible Gemini/GPT/Claude API."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Level Two B5 JSONL (default: selected by --language).",
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="zh",
        help="Prompt, parser, and default dataset language (default: zh).",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--api-provider",
        choices=tuple(API_PROVIDER_DEFAULT_MODELS),
        default=None,
        help=(
            "Use an OpenAI-compatible remote model: gemini, gpt, or claude. "
            "Remote evaluation is English-only and reads the key from an environment variable."
        ),
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Remote model name (default depends on --api-provider).",
    )
    parser.add_argument(
        "--api-base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"OpenAI-compatible API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the remote API key.",
    )
    parser.add_argument("--api-timeout", type=positive_int, default=300)
    parser.add_argument("--api-max-retries", type=non_negative_int, default=2)
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=4,
        help="Concurrent remote API requests (ignored for local inference; default: 4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Prediction JSONL. Defaults to "
            "outputs/level_two/<model-basename>/level-two-b5-text-only[-en].jsonl."
        ),
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=SUPPORTED_SPLITS,
        help="Keep one dataset split; repeat to keep several (default: all).",
    )
    parser.add_argument("--offset", type=non_negative_int, default=0)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--max-new-tokens", type=positive_int, default=3072)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help=(
            "Number of cases generated together (default: 1). Increase gradually "
            "until GPU memory is saturated."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the model chat-template thinking mode when supported (default: false).",
    )
    parser.add_argument(
        "--temperature",
        type=non_negative_float,
        default=None,
        help="Sampling temperature; resolved to 0 normally or 0.6 with thinking.",
    )
    parser.add_argument(
        "--top-p",
        type=probability,
        default=None,
        help="Sampling top-p; resolved to 0.95 with thinking.",
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=None,
        help="Sampling top-k; resolved to 20 with thinking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=None,
        help="Prompt Markdown file (default: selected by --language).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output instead of resuming it.",
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry only inference errors on resume; keep format-invalid responses (default: true).",
    )
    args = parser.parse_args(argv)
    if args.api_provider and args.language != "en":
        parser.error("remote Gemini/GPT/Claude evaluation supports English only; use --language en")
    if args.api_provider:
        args.api_model = args.api_model or API_PROVIDER_DEFAULT_MODELS[args.api_provider]
        args.api_key_env = args.api_key_env or API_PROVIDER_DEFAULT_KEY_ENVS[args.api_provider]
    if args.input is None:
        args.input = DEFAULT_INPUTS_BY_LANGUAGE[args.language]
    if args.prompt_template is None:
        args.prompt_template = DEFAULT_PROMPT_TEMPLATES_BY_LANGUAGE[args.language]
    if args.output is None:
        if args.api_provider:
            model_name = args.api_model.replace("/", "--")
        else:
            model_name = args.model_path.expanduser().resolve().name
        output_name = (
            "level-two-b5-text-only-en.jsonl"
            if args.language == "en"
            else "level-two-b5-text-only.jsonl"
        )
        args.output = (
            PROJECT_ROOT
            / "outputs"
            / "level_two"
            / model_name
            / output_name
        )
    return args


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def validate_record(record: dict[str, Any], where: str = "record") -> None:
    for key in ("qa_id", "sample_id", "patient_information"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{where}: {key} must be a non-empty string")
    if record.get("task_name") != EXPECTED_TASK_NAME:
        raise ValueError(
            f"{where}: expected task_name {EXPECTED_TASK_NAME!r}, "
            f"got {record.get('task_name')!r}"
        )
    if record.get("answer_type") != EXPECTED_ANSWER_TYPE:
        raise ValueError(
            f"{where}: expected answer_type {EXPECTED_ANSWER_TYPE!r}, "
            f"got {record.get('answer_type')!r}"
        )
    if record.get("split") not in SUPPORTED_SPLITS:
        raise ValueError(
            f"{where}: split must be one of {SUPPORTED_SPLITS}, "
            f"got {record.get('split')!r}"
        )


def load_items(raw_path: Path) -> list[WorkItem]:
    path = raw_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input JSONL does not exist: {path}")
    items: list[WorkItem] = []
    seen_ids: dict[str, int] = {}
    for line_number, record in read_jsonl(path):
        validate_record(record, f"{path}:{line_number}")
        qa_id = record["qa_id"]
        if qa_id in seen_ids:
            raise ValueError(
                f"duplicate qa_id {qa_id!r} at lines "
                f"{seen_ids[qa_id]} and {line_number}"
            )
        seen_ids[qa_id] = line_number
        items.append(WorkItem(path, line_number, record))
    return items


def select_items(items: list[WorkItem], args: argparse.Namespace) -> list[WorkItem]:
    selected = items
    if args.split:
        allowed = set(args.split)
        selected = [item for item in selected if item.record["split"] in allowed]
    selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


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
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return template, digest


def build_user_prompt(record: dict[str, Any], template: str) -> str:
    validate_record(record)
    return template.replace(
        "{{patient_information}}", record["patient_information"], 1
    )


def build_messages(
    record: dict[str, Any], template: str, language: str = "zh"
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[language]},
        {"role": "user", "content": build_user_prompt(record, template)},
    ]


def frontend_kind_for_model_type(model_type: str) -> str:
    if model_type == "qwen2":
        return "qwen2_causal_lm"
    if model_type == "qwen3":
        return "qwen3_causal_lm"
    if model_type in {"qwen3_5", "qwen3_5_moe"}:
        return "qwen3_5_image_text_to_text_text_only"
    if model_type == "gpt_oss":
        return "gpt_oss_causal_lm"
    if model_type == "gemma3_text":
        return "gemma3_text_causal_lm"
    # GLM-4.7-Flash is a native Transformers causal-LM model. Its checkpoint
    # exposes model_type=glm4_moe_lite and a tokenizer chat template (including
    # the enable_thinking switch), so it uses the text-only decoder path.
    if model_type == "glm4_moe_lite":
        return "glm4_moe_lite_causal_lm"
    raise ValueError(
        f"unsupported model_type {model_type!r}; expected qwen2, qwen3, "
        "qwen3_5, gpt_oss, gemma3_text, or glm4_moe_lite"
    )


def resolve_model_spec(model_path: Path) -> ModelSpec:
    resolved = model_path.expanduser().resolve()
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config JSON: {config_path}: {exc}") from exc
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"model config has no string model_type: {config_path}")
    architectures = config.get("architectures", [])
    if not isinstance(architectures, list) or not all(
        isinstance(value, str) for value in architectures
    ):
        raise ValueError(f"model config architectures must be a string list: {config_path}")
    return ModelSpec(
        model_type=model_type,
        architectures=tuple(architectures),
        frontend_kind=frontend_kind_for_model_type(model_type),
    )


def resolve_generation_options(args: argparse.Namespace) -> dict[str, Any]:
    temperature = args.temperature
    if temperature is None:
        temperature = 0.6 if args.enable_thinking else 0.0
    top_p = args.top_p
    if top_p is None:
        top_p = 0.95 if args.enable_thinking else 1.0
    top_k = args.top_k
    if top_k is None:
        top_k = 20 if args.enable_thinking else 0
    do_sample = temperature > 0
    options: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        options.update(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    return options


def resolve_attn_implementation(
    requested: str, spec: ModelSpec
) -> str:
    # Transformers 5.14 does not expose SDPA for GptOssForCausalLM.
    if spec.model_type == "gpt_oss":
        return "eager"
    return requested


def resolve_assistant_prefill(
    model_path: Path, spec: ModelSpec, language: str = "zh"
) -> str:
    if spec.model_type == "qwen2" and "morpheus" in model_path.name.lower():
        return f"【{EXPECTED_HEADINGS_BY_LANGUAGE[language][0]}】\n"
    return ""


def _get_config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def resolve_context_window(config: Any) -> int | None:
    candidates: list[int] = []
    for current in (config, _get_config_value(config, "text_config")):
        if current is None:
            continue
        for key in (
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "model_max_length",
        ):
            value = _get_config_value(current, key)
            if isinstance(value, int) and 0 < value < 100_000_000:
                candidates.append(value)
    return min(candidates) if candidates else None


def ensure_context_capacity(
    input_tokens: int, max_new_tokens: int, context_window: int | None
) -> None:
    if context_window is None:
        raise ValueError("model context window could not be determined safely")
    requested = input_tokens + max_new_tokens
    if requested > context_window:
        raise ValueError(
            f"input ({input_tokens}) + max_new_tokens ({max_new_tokens}) "
            f"exceeds model context window ({context_window}); input was not truncated"
        )


def ensure_gpt_oss_mxfp4_runtime(
    spec: ModelSpec,
    availability_check: Any | None = None,
    installed_version: str | None = None,
    minimum_version: str | None = None,
    maximum_version: str | None = None,
) -> None:
    """Reject the unstable BF16 fallback when GPT-OSS MXFP4 kernels are unavailable."""
    if spec.model_type != "gpt_oss":
        return
    if availability_check is None:
        from transformers.utils import is_kernels_available
        from transformers.utils.import_utils import (
            KERNELS_MAX_VERSION,
            KERNELS_MIN_VERSION,
        )

        availability_check = is_kernels_available
        minimum_version = KERNELS_MIN_VERSION
        maximum_version = KERNELS_MAX_VERSION
    if availability_check():
        return
    if installed_version is None:
        try:
            installed_version = importlib.metadata.version("kernels")
        except importlib.metadata.PackageNotFoundError:
            installed_version = "not installed"
    version_window = (
        f"{minimum_version} <= kernels < {maximum_version}"
        if minimum_version and maximum_version
        else "the version range required by the installed Transformers"
    )
    raise RuntimeError(
        "GPT-OSS uses MXFP4 weights and requires a compatible kernels package; "
        f"found kernels {installed_version}, but Transformers requires {version_window}. "
        "Refusing the automatic BF16 dequantization fallback because it can fail "
        "during CUDA weight loading. Install matching versions from the repository "
        "requirements and retry in a fresh Python process."
    )


def _dtype_value(name: str, torch_module: Any) -> Any:
    if name == "auto":
        return "auto"
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def _chat_template_kwargs(
    processor: Any, enable_thinking: bool, model_type: str
) -> dict[str, Any]:
    template = getattr(processor, "chat_template", None)
    tokenizer = getattr(processor, "tokenizer", None)
    if template is None and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
    kwargs: dict[str, Any] = {}
    if model_type == "glm4_moe_lite":
        # GLM-4.7-Flash's native template exposes this argument even though
        # some older tokenizer metadata does not advertise it explicitly.
        kwargs["enable_thinking"] = enable_thinking
    elif isinstance(template, str) and "enable_thinking" in template:
        kwargs["enable_thinking"] = enable_thinking
    if model_type == "gpt_oss" and isinstance(template, str) and "reasoning_effort" in template:
        kwargs["reasoning_effort"] = "medium" if enable_thinking else "low"
    return kwargs


def _input_device(model: Any) -> Any:
    try:
        device = model.get_input_embeddings().weight.device
        if getattr(device, "type", None) != "meta":
            return device
    except (AttributeError, NotImplementedError):
        pass
    return model.device


def _eos_token_ids(model: Any, tokenizer_or_processor: Any) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
        value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if isinstance(item, int)}
    return set()


def _generated_token_count(
    token_ids: Any, eos_ids: set[int], pad_token_id: int | None
) -> int:
    """Exclude batch padding while retaining a terminal EOS token."""
    for index, token_id in enumerate(token_ids):
        value = int(token_id)
        if value in eos_ids:
            return index + 1
        if pad_token_id is not None and value == pad_token_id:
            return index
    return len(token_ids)


class TransformersTextFrontend:
    """Minimal text-only frontend for supported local decoder-only models."""

    def __init__(self, args: argparse.Namespace, spec: ModelSpec, *,
                 assistant_prefill: str | None = None,
                 stop_strings: tuple[str, ...] | None = None):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required; install requirements-inference.txt"
            ) from exc

        self.torch = torch
        self.model_type = spec.model_type
        self.frontend_kind = spec.frontend_kind
        ensure_gpt_oss_mxfp4_runtime(spec)
        model_path = args.model_path.expanduser().resolve()
        self.assistant_prefill = resolve_assistant_prefill(
            model_path, spec, args.language
        ) if assistant_prefill is None else assistant_prefill
        self.output_stop_strings = stop_strings
        load_kwargs = {
            "local_files_only": True,
            "dtype": _dtype_value(args.dtype, torch),
            "device_map": args.device_map,
            "attn_implementation": resolve_attn_implementation(
                args.attn_implementation, spec
            ),
            "low_cpu_mem_usage": True,
        }
        if spec.frontend_kind != "qwen3_5_image_text_to_text_text_only":
            self.processor = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )
            self.processor_mode = "tokenizer"
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, **load_kwargs
            )
            self.processor_mode = "processor"
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if hasattr(tokenizer, "padding_side"):
            # Decoder-only generation must read the final prompt token in every
            # row. Right padding breaks that invariant for uneven prompt lengths.
            tokenizer.padding_side = "left"
        self.model.eval()
        self.context_window = resolve_context_window(self.model.config)

    def generate(
        self,
        messages: list[dict[str, str]],
        args: argparse.Namespace,
    ) -> GenerationResult:
        return self.generate_batch([messages], args)[0]

    def generate_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        args: argparse.Namespace,
    ) -> list[GenerationResult]:
        if not messages_batch:
            return []
        template_kwargs = _chat_template_kwargs(
            self.processor, args.enable_thinking, self.model_type
        )
        rendered = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            + self.assistant_prefill
            for messages in messages_batch
        ]
        if self.processor_mode == "tokenizer":
            inputs = self.processor(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        else:
            # A text list deliberately avoids all image/video processor inputs.
            inputs = self.processor(
                text=rendered,
                return_tensors="pt",
                padding=True,
            )
        if "pixel_values" in inputs or "image_grid_thw" in inputs:
            raise RuntimeError("text-only processor unexpectedly created vision inputs")
        inputs = inputs.to(_input_device(self.model))
        prompt_width = int(inputs["input_ids"].shape[1])
        ensure_context_capacity(
            prompt_width, args.max_new_tokens, self.context_window
        )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            input_tokens = [prompt_width] * len(messages_batch)
        else:
            input_tokens = [
                int(count) for count in attention_mask.sum(dim=1).detach().cpu().tolist()
            ]
        options = resolve_generation_options(args)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        stop_strings = list(
            FORBIDDEN_TAIL_STOP_STRINGS_BY_LANGUAGE[args.language]
        )
        stop_strings.extend(COMMON_FORBIDDEN_TAIL_STOP_STRINGS)
        if self.assistant_prefill:
            stop_strings.extend(
                MORPHEUS_TAIL_STOP_STRINGS_BY_LANGUAGE[args.language]
            )
        override = getattr(self, "output_stop_strings", None)
        if override is not None:
            stop_strings = list(override)
        if stop_strings:
            options["stop_strings"] = list(dict.fromkeys(stop_strings))
        options["tokenizer"] = tokenizer
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            options["pad_token_id"] = pad_token_id
        with self.torch.inference_mode():
            sequences = self.model.generate(**inputs, **options)
        generated_ids = sequences[:, prompt_width:]
        eos_ids = _eos_token_ids(self.model, self.processor)
        results = []
        for row, token_ids in enumerate(generated_ids):
            generated_tokens = _generated_token_count(
                token_ids, eos_ids, pad_token_id
            )
            usable_ids = token_ids[:generated_tokens]
            if self.model_type == "gpt_oss":
                # Keep Harmony channel tokens so analysis and final can be split
                # without relying on ambiguous plain-text words.
                text = tokenizer.decode(
                    usable_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                text = self.processor.batch_decode(
                    [usable_ids],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                text = self.assistant_prefill + text
            last_token = int(usable_ids[-1]) if generated_tokens else None
            truncated = (
                generated_tokens >= args.max_new_tokens
                and (last_token is None or last_token not in eos_ids)
            )
            results.append(
                GenerationResult(
                    text, input_tokens[row], generated_tokens, truncated
                )
            )
        return results


def create_frontend(args: argparse.Namespace, spec: ModelSpec) -> TransformersTextFrontend:
    return TransformersTextFrontend(args, spec)


def _api_usage_value(usage: Any, name: str) -> int:
    value = getattr(usage, name, None) if usage is not None else None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(usage, dict):
        value = usage.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _api_message_content(message: Any) -> str:
    """Normalize OpenAI-compatible string or content-block responses."""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
                continue
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
            else:
                value = getattr(block, "text", None) or getattr(block, "content", None)
            if isinstance(value, str):
                pieces.append(value)
        return "".join(pieces)
    return ""


class OpenAICompatibleFrontend:
    """Thread-safe OpenAI-compatible chat frontend for hosted model evaluation."""

    model_type = "openai_compatible"
    frontend_kind = "openai_compatible_chat"

    def __init__(self, args: argparse.Namespace):
        if not args.api_provider:
            raise ValueError("--api-provider is required for the remote frontend")
        key_env = args.api_key_env or API_PROVIDER_DEFAULT_KEY_ENVS[args.api_provider]
        key_candidates = [key_env]
        # Accept concise provider-specific names as a convenience, while still
        # allowing --api-key-env to select any project-specific variable.
        alias = {
            "gemini": "GEMINI_API_KEY",
            "gpt": "GPT_API_KEY",
            "claude": "CLAUDE_API_KEY",
        }[args.api_provider]
        if alias not in key_candidates:
            key_candidates.append(alias)
        api_key = next((os.environ.get(name) for name in key_candidates if os.environ.get(name)), None)
        if not api_key:
            raise RuntimeError(
                "API key environment variable is not set (tried "
                + ", ".join(repr(name) for name in key_candidates)
                + "); keys are never read from command-line values or source files"
            )
        key_env = next(name for name in key_candidates if os.environ.get(name))
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The remote frontend requires the openai package; "
                "install it in the active environment"
            ) from exc
        self.provider = args.api_provider
        self.model_type = f"api_{self.provider}"
        self.model_identifier = args.api_model
        self.api_base_url = args.api_base_url
        self.key_env = key_env
        self.client = OpenAI(
            base_url=args.api_base_url,
            api_key=api_key,
            timeout=args.api_timeout,
            max_retries=0,
        )
        self.max_retries = args.api_max_retries

    def generate(
        self, messages: list[dict[str, str]], args: argparse.Namespace
    ) -> GenerationResult:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.model_identifier,
                    "messages": messages,
                    "max_tokens": args.max_new_tokens,
                }
                # Omit sampling controls by default: hosted GPT/Claude/Gemini
                # endpoints differ in which controls they accept.
                if args.temperature is not None:
                    request["temperature"] = args.temperature
                response = self.client.chat.completions.create(**request)
                choices = (
                    response.get("choices") if isinstance(response, dict)
                    else getattr(response, "choices", None)
                )
                if not choices:
                    raise ValueError("API response contains no choices")
                choice = choices[0]
                message = (
                    choice.get("message") if isinstance(choice, dict)
                    else getattr(choice, "message", None)
                )
                text = _api_message_content(message)
                if not text.strip():
                    finish = (
                        choice.get("finish_reason") if isinstance(choice, dict)
                        else getattr(choice, "finish_reason", None)
                    )
                    raise ValueError(
                        "API response returned empty content"
                        + (f" (finish_reason={finish!r})" if finish else "")
                    )
                usage = (
                    response.get("usage") if isinstance(response, dict)
                    else getattr(response, "usage", None)
                )
                input_tokens = _api_usage_value(usage, "prompt_tokens")
                generated_tokens = _api_usage_value(usage, "completion_tokens")
                finish_reason = (
                    choice.get("finish_reason") if isinstance(choice, dict)
                    else getattr(choice, "finish_reason", None)
                )
                truncated = finish_reason in {"length", "max_tokens"}
                return GenerationResult(
                    text=text,
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    truncated=truncated,
                )
            except Exception as exc:  # retry transient and empty-content responses
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        assert last_error is not None
        raise RuntimeError(
            f"{self.provider} API request failed after {self.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def generate_batch(
        self, messages_batch: list[list[dict[str, str]]], args: argparse.Namespace
    ) -> list[GenerationResult]:
        # Hosted chat APIs do not provide the local tensor batch contract.
        return [self.generate(messages, args) for messages in messages_batch]


def _remove_thinking(text: str) -> tuple[str, bool, str | None]:
    lowered = text.lower()
    has_open = "<think>" in lowered
    has_close = "</think>" in lowered
    if has_open and not has_close:
        return text.strip(), False, "unclosed <think> block"
    if has_close and not has_open:
        # DeepSeek-R1 templates prefill <think> in the prompt, so generated
        # tokens contain reasoning followed only by the closing tag.
        close_end = lowered.index("</think>") + len("</think>")
        return text[close_end:].strip(), True, None
    if not has_open:
        return text.strip(), False, None
    cleaned, count = THINK_BLOCK_RE.subn("", text)
    if "<think>" in cleaned.lower() or "</think>" in cleaned.lower():
        return cleaned.strip(), True, "malformed or nested <think> block"
    return cleaned.strip(), count > 0, None


def _extract_gpt_oss_final(text: str) -> tuple[str, bool, str | None]:
    if GPT_OSS_FINAL_MARKER not in text:
        if "<|channel|>analysis" in text:
            return text.strip(), False, "gpt-oss response has analysis but no final channel"
        return text.strip(), False, None
    final = text.rsplit(GPT_OSS_FINAL_MARKER, 1)[1]
    ends = [final.find(marker) for marker in GPT_OSS_END_MARKERS]
    ends = [position for position in ends if position >= 0]
    if ends:
        final = final[: min(ends)]
    return final.strip(), True, None


def _markdown_heading_name(raw: str) -> str:
    name = raw.strip()
    if name.startswith("【") and name.endswith("】"):
        name = name[1:-1].strip()
    return name


def _recover_markdown_headings(
    text: str, expected_headings: tuple[str, ...]
) -> tuple[str, bool, list[str]]:
    matches = list(MARKDOWN_HEADING_RE.finditer(text))
    if not matches:
        return text, False, []
    names = [_markdown_heading_name(match.group(1)) for match in matches]
    expected_matches = [
        (match, name)
        for match, name in zip(matches, names)
        if name in expected_headings
    ]
    if [name for _, name in expected_matches] != list(expected_headings):
        return text, False, []
    if len(expected_matches) != len(expected_headings):
        return text, False, []

    last_expected_end = expected_matches[-1][0].end()
    cut_at = len(text)
    notes = ["recovered Markdown-style section headings"]
    for match, name in zip(matches, names):
        if match.start() > last_expected_end and name not in expected_headings:
            cut_at = match.start()
            notes.append(f"discarded tail beginning at Markdown heading: {name}")
            break
    if cut_at == len(text):
        for marker in ("\n```json", "\n```JSON"):
            position = text.find(marker, last_expected_end)
            if position >= 0:
                cut_at = min(cut_at, position)
                notes.append("discarded forbidden JSON tail")

    source = text[:cut_at]
    pieces: list[str] = []
    cursor = 0
    for match, name in expected_matches:
        if match.start() >= cut_at:
            break
        pieces.append(source[cursor : match.start()])
        pieces.append(f"【{name}】")
        cursor = match.end()
    pieces.append(source[cursor:])

    return "".join(pieces).strip(), True, notes


def _normalize_flexible_heading(raw: str) -> str:
    name = raw.strip()
    if name.startswith("【") and name.endswith("】"):
        name = name[1:-1].strip()
    if name.startswith("**") and name.endswith("**"):
        name = name[2:-2].strip()
    return name.rstrip(":").strip()


def _recover_flexible_english_sections(
    text: str, expected_headings: tuple[str, ...]
) -> tuple[str, bool, list[str]]:
    """Recover semantically unambiguous English heading variants.

    Generic Evidence labels are mapped only inside Risk Prediction or Acute
    Diagnosis. Other generic labels remain part of their surrounding content.
    """
    if tuple(expected_headings) != EXPECTED_HEADINGS_EN:
        return text, False, []
    exact_names = [
        match.group(1).strip() for match in HEADING_LINE_RE.finditer(text)
    ]
    if exact_names == list(expected_headings):
        return text, False, []

    canonical = {name.casefold(): name for name in expected_headings}
    aliases = {
        "evidence supporting risk prediction": "Prediction Evidence",
        "risk prediction evidence": "Prediction Evidence",
        "evidence supporting diagnosis": "Diagnostic Evidence",
        "diagnosis evidence": "Diagnostic Evidence",
    }
    main_names = {
        "Risk Prediction", "Acute Diagnosis", "Management Decision",
        "Reassessment Plan",
    }
    tail_names = {
        "Clinical Decision Analysis", "Final Clinical Conclusion",
        "Final Output", "Final Decision Summary", "Note",
    }
    markers: list[dict[str, object]] = []
    tail_positions: list[int] = []

    for line_match in re.finditer(r"(?m)^.*(?:\r?\n|\Z)", text):
        raw_line = line_match.group(0).rstrip("\r\n")
        if not raw_line and line_match.start() == len(text):
            continue
        base = line_match.start()
        stripped = raw_line.strip()
        label: str | None = None
        content_start: int | None = None
        bracket = re.fullmatch(r"[ \t]*【([^】\r\n]+)】[ \t]*", raw_line)
        markdown = re.fullmatch(r"[ \t]*#{1,6}[ \t]+(.+?)[ \t]*", raw_line)
        bold = re.fullmatch(
            r"[ \t]*\*\*(?P<label>[^*\r\n]+?)\*\*[ \t]*(?P<rest>.*?)[ \t]*",
            raw_line,
        )
        plain = re.fullmatch(
            r"[ \t]*(?P<label>Evidence|Prediction Evidence|Diagnostic Evidence|"
            r"Risk Prediction|Acute Diagnosis|Management Decision|Specific Actions|"
            r"Reassessment Plan|Backup & Escalation Plan)[ \t]*:[ \t]*(?P<rest>.*)",
            raw_line,
            flags=re.IGNORECASE,
        )
        if bracket:
            label = bracket.group(1)
            content_start = base + len(raw_line)
        elif markdown:
            label = markdown.group(1)
            content_start = base + len(raw_line)
        elif bold:
            label = bold.group("label")
            content_start = base + bold.start("rest")
        elif plain:
            label = plain.group("label")
            content_start = base + plain.start("rest")
        elif stripped.lower().startswith("```json"):
            tail_positions.append(base)
            continue
        else:
            continue

        normalized = _normalize_flexible_heading(label)
        key = normalized.casefold()
        if key in canonical:
            name = canonical[key]
        elif key in aliases:
            name = aliases[key]
        elif key == "evidence":
            name = "Evidence"
        elif normalized in tail_names:
            tail_positions.append(base)
            continue
        else:
            continue
        markers.append({"name": name, "start": base, "content_start": content_start})

    explicit = [marker for marker in markers if marker["name"] != "Evidence"]

    for marker in markers:
        if marker["name"] != "Evidence":
            continue
        position = int(marker["start"])
        previous_main = next((
            candidate for candidate in reversed(explicit)
            if int(candidate["start"]) < position and candidate["name"] in main_names
        ), None)
        next_main = next((
            candidate for candidate in explicit
            if int(candidate["start"]) > position and candidate["name"] in main_names
        ), None)
        if previous_main is None:
            continue
        target = {
            "Risk Prediction": "Prediction Evidence",
            "Acute Diagnosis": "Diagnostic Evidence",
        }.get(previous_main["name"])
        if target is None:
            continue
        interval_end = int(next_main["start"]) if next_main else len(text)
        has_explicit_target = any(
            candidate["name"] == target
            and int(previous_main["start"]) < int(candidate["start"]) < interval_end
            for candidate in explicit
        )
        if not has_explicit_target:
            marker["name"] = target

    required = [marker for marker in markers if marker["name"] in expected_headings]
    first_by_name: dict[str, dict[str, object]] = {}
    for marker in required:
        first_by_name.setdefault(str(marker["name"]), marker)
    if set(first_by_name) != set(expected_headings):
        return text, False, []

    last_first = max(int(marker["start"]) for marker in first_by_name.values())
    duplicate_cycles = [
        int(marker["start"]) for marker in required
        if int(marker["start"]) > last_first
        and marker is not first_by_name[str(marker["name"])]
    ]
    cutoff_candidates = [p for p in tail_positions if p > last_first] + duplicate_cycles
    cutoff = min(cutoff_candidates, default=len(text))

    selected: list[dict[str, object]] = []
    for name in expected_headings:
        candidates = [
            marker for marker in required
            if marker["name"] == name and int(marker["start"]) < cutoff
        ]
        if not candidates:
            return text, False, []
        selected.append(candidates[0])

    document_order = sorted(selected, key=lambda marker: int(marker["start"]))
    bodies: dict[str, str] = {}
    for index, marker in enumerate(document_order):
        body_end = (
            int(document_order[index + 1]["start"])
            if index + 1 < len(document_order) else cutoff
        )
        body = text[int(marker["content_start"]):body_end].strip()
        if not body:
            return text, False, []
        bodies[str(marker["name"])] = body

    recovered = "\n\n".join(
        f"【{name}】\n{bodies[name]}" for name in expected_headings
    )
    notes = ["recovered semantically mapped English section labels"]
    if [marker["name"] for marker in document_order] != list(expected_headings):
        notes.append("restored required sections to canonical order")
    if cutoff < len(text):
        notes.append("discarded non-schema tail after recovered sections")
    return recovered, True, notes


def _strip_tail_after_exact_sections(
    text: str, expected_headings: tuple[str, ...], language: str
) -> tuple[str, bool, str | None]:
    exact_matches = list(HEADING_LINE_RE.finditer(text))
    if [match.group(1).strip() for match in exact_matches] != list(expected_headings):
        return text, False, None
    last_heading_end = exact_matches[-1].end()
    tail_labels = (
        ("临床决策分析", "最终临床结论")
        if language == "zh"
        else ("Clinical Decision Analysis", "Final Clinical Conclusion")
    )
    label_pattern = "|".join(re.escape(label) for label in tail_labels)
    first_heading = re.escape(expected_headings[0])
    tail_patterns = (
        rf"(?m)^[ \t]*#{{1,6}}[ \t]+(?:{label_pattern})[ \t]*$",
        rf"(?m)^[ \t]*#{{1,6}}[ \t]+\*\*{first_heading}\*\*[ \t]*$",
        rf"(?m)^[ \t]*#{{1,6}}[ \t]+【{first_heading}】[ \t]*$",
        r"(?m)^[ \t]*```(?:json)?[ \t]*$",
    )
    positions = []
    for pattern in tail_patterns:
        match = re.search(pattern, text[last_heading_end:], flags=re.IGNORECASE)
        if match:
            positions.append(last_heading_end + match.start())
    if not positions:
        return text, False, None
    cut_at = min(positions)
    return (
        text[:cut_at].rstrip(),
        True,
        "discarded forbidden tail after the eight required sections",
    )


def _combine_prediction(
    sections: dict[str, str], section_pairs: dict[str, tuple[str, str]]
) -> dict[str, str]:
    prediction: dict[str, str] = {}
    for key, pair in section_pairs.items():
        prediction[key] = "\n\n".join(
            f"【{heading}】\n{sections[heading]}" for heading in pair
        )
    return prediction


def parse_model_response(text: str, language: str = "zh") -> ParsedResponse:
    expected_headings = EXPECTED_HEADINGS_BY_LANGUAGE[language]
    section_pairs = SECTION_PAIRS_BY_LANGUAGE[language]
    format_errors: list[str] = []
    channel_text, removed_channel, channel_error = _extract_gpt_oss_final(text)
    if channel_error:
        return ParsedResponse(
            channel_text,
            None,
            None,
            None,
            False,
            (channel_error,),
            channel_error,
        )
    if removed_channel:
        format_errors.append("extracted gpt-oss final channel")
    final_answer_text, removed_thinking, think_error = _remove_thinking(channel_text)
    if think_error:
        format_errors.append(think_error)
        return ParsedResponse(
            final_answer_text,
            None,
            None,
            None,
            False,
            tuple(format_errors),
            think_error,
        )
    if removed_thinking:
        format_errors.append("response contained a <think> block")

    extraction_text = final_answer_text
    fence_match = OUTER_FENCE_RE.fullmatch(extraction_text)
    if fence_match:
        extraction_text = fence_match.group(1).strip()
        format_errors.append("response was wrapped in a Markdown code fence")

    extraction_text, stripped_tail, tail_note = _strip_tail_after_exact_sections(
        extraction_text, expected_headings, language
    )
    if stripped_tail and tail_note:
        format_errors.append(tail_note)

    extraction_text, recovered_markdown, markdown_notes = _recover_markdown_headings(
        extraction_text, expected_headings
    )
    if recovered_markdown:
        format_errors.extend(markdown_notes)

    extraction_text, recovered_flexible, flexible_notes = (
        _recover_flexible_english_sections(extraction_text, expected_headings)
    )
    if recovered_flexible:
        format_errors.extend(flexible_notes)

    matches = list(HEADING_LINE_RE.finditer(extraction_text))
    names = [match.group(1).strip() for match in matches]
    structural_errors: list[str] = []
    for heading in expected_headings:
        count = names.count(heading)
        if count == 0:
            structural_errors.append(f"missing heading: 【{heading}】")
        elif count > 1:
            structural_errors.append(f"duplicate heading: 【{heading}】")
    unexpected = [name for name in names if name not in expected_headings]
    if unexpected:
        structural_errors.append(
            "unexpected headings: "
            + ", ".join(f"【{name}】" for name in unexpected)
        )
    if not structural_errors and names != list(expected_headings):
        structural_errors.append("headings are out of order")
    if matches and extraction_text[: matches[0].start()].strip():
        structural_errors.append("unexpected text before the first heading")

    if structural_errors:
        format_errors.extend(structural_errors)
        error = "; ".join(structural_errors)
        return ParsedResponse(
            final_answer_text,
            None,
            None,
            None,
            False,
            tuple(format_errors),
            error,
        )

    sections: dict[str, str] = {}
    empty_headings: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(extraction_text)
        body = extraction_text[match.end() : end].strip()
        heading = match.group(1).strip()
        sections[heading] = body
        if not body:
            empty_headings.append(heading)
    if empty_headings:
        error = "empty heading bodies: " + ", ".join(
            f"【{heading}】" for heading in empty_headings
        )
        format_errors.append(error)
        return ParsedResponse(
            final_answer_text,
            None,
            None,
            None,
            False,
            tuple(format_errors),
            error,
        )

    strict = not format_errors
    if strict:
        method = "strict_eight_heading_v1"
    else:
        recoveries = []
        if removed_channel:
            recoveries.append("gpt_oss_final_channel")
        if removed_thinking:
            recoveries.append("think")
        if fence_match:
            recoveries.append("code_fence")
        if stripped_tail:
            recoveries.append("discarded_tail")
        if recovered_markdown:
            recoveries.append("markdown_headings")
        if recovered_flexible:
            recoveries.append("semantic_headings")
        method = "recovered_" + "_and_".join(recoveries) + "_eight_heading_v1"
    return ParsedResponse(
        final_answer_text=final_answer_text,
        prediction_sections=sections,
        prediction=_combine_prediction(sections, section_pairs),
        extraction_method=method,
        strict_output_format_valid=strict,
        format_errors=tuple(format_errors),
        prediction_error=None,
    )


def _seed_for_item(base_seed: int, qa_id: str) -> None:
    try:
        import torch
    except ImportError:
        return
    digest = hashlib.sha256(qa_id.encode("utf-8")).digest()
    seed = (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configured_model_reference(args: argparse.Namespace) -> str:
    if getattr(args, "api_provider", None):
        return str(args.api_model)
    return str(args.model_path.expanduser().resolve())


def _result_from_generated(
    item: WorkItem,
    args: argparse.Namespace,
    frontend: Any,
    generated: GenerationResult,
    prompt_hash: str,
    elapsed: float,
) -> dict[str, Any]:
    parsed = parse_model_response(generated.text, args.language)
    strict_valid = parsed.strict_output_format_valid and not generated.truncated
    response_valid = parsed.prediction is not None and not generated.truncated
    format_errors = list(parsed.format_errors)
    if generated.truncated:
        format_errors.append("generation reached max_new_tokens without EOS")
    status = "ok" if response_valid else "invalid_response"
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "qa_id": item.record["qa_id"],
        "sample_id": item.record["sample_id"],
        "task_group": item.record.get("task_group"),
        "task_name": item.record["task_name"],
        "answer_type": item.record["answer_type"],
        "split": item.record["split"],
        "language": args.language,
        "model_path": _configured_model_reference(args),
        "model_type": frontend.model_type,
        "model_frontend": frontend.frontend_kind,
        "input_mode": "text_only",
        "media_included": False,
        "batch_size": args.batch_size,
        "prompt_version": PROMPT_VERSIONS[args.language],
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
        "truncated": generated.truncated,
        "input_tokens": generated.input_tokens,
        "generated_tokens": generated.generated_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_one(
    item: WorkItem,
    args: argparse.Namespace,
    frontend: Any,
    template: str,
    prompt_hash: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    _seed_for_item(args.seed, item.record["qa_id"])
    messages = build_messages(item.record, template, args.language)
    generated = frontend.generate(messages, args)
    return _result_from_generated(
        item, args, frontend, generated, prompt_hash, time.perf_counter() - started
    )


def run_many(
    items: list[WorkItem],
    args: argparse.Namespace,
    frontend: Any,
    template: str,
    prompt_hash: str,
) -> list[dict[str, Any]]:
    if not items:
        return []
    started = time.perf_counter()
    _seed_for_item(args.seed, "\0".join(item.record["qa_id"] for item in items))
    messages_batch = [
        build_messages(item.record, template, args.language) for item in items
    ]
    generated_batch = frontend.generate_batch(messages_batch, args)
    if len(generated_batch) != len(items):
        raise RuntimeError(
            f"batch generator returned {len(generated_batch)} results for {len(items)} items"
        )
    elapsed_per_item = (time.perf_counter() - started) / len(items)
    return [
        _result_from_generated(
            item, args, frontend, generated, prompt_hash, elapsed_per_item
        )
        for item, generated in zip(items, generated_batch)
    ]


def error_result(
    item: WorkItem,
    args: argparse.Namespace,
    exc: Exception,
    elapsed: float,
    spec: ModelSpec,
    prompt_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "qa_id": item.record["qa_id"],
        "sample_id": item.record["sample_id"],
        "task_group": item.record.get("task_group"),
        "task_name": item.record["task_name"],
        "answer_type": item.record["answer_type"],
        "split": item.record["split"],
        "language": args.language,
        "model_path": _configured_model_reference(args),
        "model_type": spec.model_type,
        "model_frontend": spec.frontend_kind,
        "input_mode": "text_only",
        "media_included": False,
        "batch_size": args.batch_size,
        "prompt_version": PROMPT_VERSIONS[args.language],
        "prompt_sha256": prompt_hash,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "elapsed_seconds": round(elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def load_latest_results(
    path: Path, allowed_qa_ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for _, record in read_jsonl(path):
        qa_id = record.get("qa_id")
        if not isinstance(qa_id, str):
            continue
        if allowed_qa_ids is not None and qa_id not in allowed_qa_ids:
            continue
        latest[qa_id] = record
    return latest


def write_latest_results(
    path: Path,
    latest: dict[str, dict[str, Any]],
    ordered_qa_ids: list[str],
) -> None:
    """Atomically persist at most one, latest result per dataset item."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for qa_id in ordered_qa_ids:
                record = latest.get(qa_id)
                if record is not None:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_previous_statuses(path: Path) -> dict[str, str]:
    statuses = {}
    for qa_id, record in load_latest_results(path).items():
        status = record.get("status")
        if isinstance(status, str):
            statuses[qa_id] = status
    return statuses


def completed_qa_ids(statuses: dict[str, str], retry_errors: bool) -> set[str]:
    if not retry_errors:
        return set(statuses)
    return {qa_id for qa_id, status in statuses.items() if status != "error"}


def write_run_config(
    args: argparse.Namespace,
    spec: ModelSpec,
    selected_count: int,
    pending_count: int,
    prompt_hash: str,
) -> None:
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    generation = resolve_generation_options(args)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "model_path": str(args.model_path.expanduser().resolve()),
        "model_type": spec.model_type,
        "model_architectures": list(spec.architectures),
        "model_frontend": spec.frontend_kind,
        "input": str(args.input.expanduser().resolve()),
        "output": str(args.output),
        "language": args.language,
        "splits": args.split or list(SUPPORTED_SPLITS),
        "offset": args.offset,
        "limit": args.limit,
        "selected_count_before_resume": selected_count,
        "pending_count": pending_count,
        "input_mode": "text_only",
        "media_included": False,
        "output_persistence": "latest_per_qa_id_atomic",
        "retry_policy": "errors_only",
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "effective_attn_implementation": resolve_attn_implementation(
            args.attn_implementation, spec
        ),
        "assistant_prefill": resolve_assistant_prefill(
            args.model_path, spec, args.language
        ),
        "enable_thinking": args.enable_thinking,
        "generation_options": generation,
        "seed": args.seed,
        "prompt_version": PROMPT_VERSIONS[args.language],
        "prompt_template": str(args.prompt_template.expanduser().resolve()),
        "prompt_sha256": prompt_hash,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_api_run_config(
    args: argparse.Namespace,
    selected_count: int,
    pending_count: int,
    prompt_hash: str,
) -> None:
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "backend": "openai_compatible",
        "api_provider": args.api_provider,
        "api_model": args.api_model,
        "api_base_url": args.api_base_url,
        "api_key_env": args.api_key_env,
        "input": str(args.input),
        "output": str(args.output),
        "language": args.language,
        "splits": args.split or list(SUPPORTED_SPLITS),
        "offset": args.offset,
        "limit": args.limit,
        "selected_count_before_resume": selected_count,
        "pending_count": pending_count,
        "input_mode": "text_only",
        "media_included": False,
        "output_persistence": "latest_per_qa_id_atomic",
        "retry_policy": "errors_only",
        "max_workers": args.max_workers,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "api_timeout": args.api_timeout,
        "api_max_retries": args.api_max_retries,
        "enable_thinking": False,
        "prompt_version": PROMPT_VERSIONS[args.language],
        "prompt_template": str(args.prompt_template),
        "prompt_sha256": prompt_hash,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_api_inference(args: argparse.Namespace) -> int:
    if args.language != "en":
        raise ValueError("remote Gemini/GPT/Claude evaluation is English-only")
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.prompt_template = args.prompt_template.expanduser().resolve()
    template, prompt_hash = load_prompt_template(args.prompt_template)
    all_items = load_items(args.input)
    items = select_items(all_items, args)
    ordered_qa_ids = [item.record["qa_id"] for item in all_items]
    allowed_qa_ids = set(ordered_qa_ids)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    latest_results = load_latest_results(args.output, allowed_qa_ids)
    # A resumed record is reusable only for the same API model and prompt.
    model_ref = str(args.api_model)
    latest_results = {
        qa_id: record
        for qa_id, record in latest_results.items()
        if record.get("model_path") == model_ref
        and record.get("model_type") == f"api_{args.api_provider}"
        and record.get("prompt_sha256") == prompt_hash
        and record.get("language") == "en"
    }
    if args.output.exists():
        write_latest_results(args.output, latest_results, ordered_qa_ids)
    previous = {
        qa_id: record["status"]
        for qa_id, record in latest_results.items()
        if isinstance(record.get("status"), str)
    }
    completed = completed_qa_ids(previous, args.retry_errors)
    pending = [item for item in items if item.record["qa_id"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_api_run_config(args, len(items), len(pending), prompt_hash)
    if not pending:
        print(
            f"No pending remote items. Selected={len(items)}, "
            f"already recorded={len(items)}"
        )
        return 0

    spec = ModelSpec(
        model_type=f"api_{args.api_provider}",
        architectures=(),
        frontend_kind="openai_compatible_chat",
    )
    frontend = OpenAICompatibleFrontend(args)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required; install requirements-inference.txt") from exc

    ok_count = invalid_count = error_count = 0
    progress = tqdm(
        total=len(pending), desc=f"AnesBench Level Two ({args.api_provider})",
        unit="item", dynamic_ncols=True,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_workers,
            thread_name_prefix=f"l2-{args.api_provider}",
        ) as executor:
            futures = {
                executor.submit(
                    run_one, item, args, frontend, template, prompt_hash
                ): item
                for item in pending
            }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                started = time.perf_counter()
                try:
                    result = future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    result = error_result(
                        item, args, exc, time.perf_counter() - started, spec, prompt_hash
                    )
                latest_results[result["qa_id"]] = result
                if result["status"] == "ok":
                    ok_count += 1
                elif result["status"] == "error":
                    error_count += 1
                else:
                    invalid_count += 1
                write_latest_results(args.output, latest_results, ordered_qa_ids)
                progress.update(1)
                progress.set_postfix(
                    ok=ok_count, invalid=invalid_count, errors=error_count, refresh=False
                )
    finally:
        progress.close()
    print(
        f"Finished remote {args.api_provider}: ok={ok_count}, "
        f"invalid={invalid_count}, errors={error_count}, "
        f"total_records={len(latest_results)}, output={args.output}"
    )
    return 0 if error_count == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.input = args.input.expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.prompt_template = args.prompt_template.expanduser().resolve()

    if args.api_provider:
        return run_api_inference(args)

    template, prompt_hash = load_prompt_template(args.prompt_template)
    all_items = load_items(args.input)
    items = select_items(all_items, args)
    ordered_qa_ids = [item.record["qa_id"] for item in all_items]
    allowed_qa_ids = set(ordered_qa_ids)
    spec = resolve_model_spec(args.model_path)
    if args.overwrite and args.output.exists():
        args.output.unlink()
    latest_results = load_latest_results(args.output, allowed_qa_ids)
    if args.output.exists():
        write_latest_results(args.output, latest_results, ordered_qa_ids)
    previous = {
        qa_id: record["status"]
        for qa_id, record in latest_results.items()
        if isinstance(record.get("status"), str)
    }
    completed = completed_qa_ids(previous, args.retry_errors)
    pending = [item for item in items if item.record["qa_id"] not in completed]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(args, spec, len(items), len(pending), prompt_hash)
    if not pending:
        print(
            f"No pending items. Selected={len(items)}, already recorded={len(items)}"
        )
        return 0

    print(
        f"Loading local model {args.model_path} with {spec.frontend_kind} "
        f"for {len(pending)} text-only item(s) "
        f"({len(items) - len(pending)} resumed)...",
        flush=True,
    )
    frontend = create_frontend(args, spec)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "tqdm is required; install requirements-inference.txt"
        ) from exc

    ok_count = 0
    invalid_count = 0
    error_count = 0
    progress = tqdm(
        total=len(pending),
        desc="AnesBench Level Two",
        unit="item",
        dynamic_ncols=True,
    )
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_started = time.perf_counter()
        try:
            results = run_many(batch, args, frontend, template, prompt_hash)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            elapsed_per_item = (time.perf_counter() - batch_started) / len(batch)
            results = [
                error_result(
                    item,
                    args,
                    exc,
                    elapsed_per_item,
                    spec,
                    prompt_hash,
                )
                for item in batch
            ]
        for result in results:
            if result["status"] == "ok":
                ok_count += 1
            elif result["status"] == "error":
                error_count += 1
            else:
                invalid_count += 1
            latest_results[result["qa_id"]] = result
        write_latest_results(args.output, latest_results, ordered_qa_ids)
        progress.update(len(batch))
        progress.set_postfix(
            ok=ok_count,
            invalid=invalid_count,
            errors=error_count,
            refresh=False,
        )

    print(
        f"Finished: ok={ok_count}, invalid={invalid_count}, "
        f"errors={error_count}, total_records={len(latest_results)}, "
        f"output={args.output}"
    )
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
