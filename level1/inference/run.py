#!/usr/bin/env python3
"""Run frozen AnesBench Visual Perception TEE generation with a local VLM.

This script intentionally does not score predictions. It never includes the
benchmark answer, ground truth, review metadata, or manifest context in model
inputs. Predictions can be joined to the source records later by ``qa_id``.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import json
import mimetypes
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.clinical_facts import (extract_clinical_facts, infer_assessment_target, parse_controlled_response)
from inference_adapters import (
    ADAPTER_NAMES,
    GenerationRequest,
    create_adapter,
    resolve_adapter_spec,
    thinking_template_kwargs,
)


DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "ANESBENCH_MODEL_PATH",
        str(PROJECT_ROOT / "checkpoints" / "Qwen" / "Qwen3-VL-32B-Instruct"),
    )
)
SUPPORTED_LANGUAGES = ("zh", "en")
DEFAULT_INPUTS_BY_LANGUAGE = {
    "zh": (PROJECT_ROOT / "level_one/TEE/Visual Perception_TEE_zh.jsonl",),
    "en": (PROJECT_ROOT / "level_one/TEE/Visual Perception_TEE_en.jsonl",),
}
DEFAULT_OUTPUTS_BY_LANGUAGE = {
    language: PROJECT_ROOT
    / f"outputs/qwen3-vl-32b/visual-perception-tee-{language}.jsonl"
    for language in SUPPORTED_LANGUAGES
}
PROMPT_TEMPLATE_PATHS = {
    language: PROJECT_ROOT
    / f"doc/Visual_Perception_TEE_Prompt_Templates_{language}.md"
    for language in SUPPORTED_LANGUAGES
}
SUPPORTED_ANSWER_TYPES = {
    "multiple_choice",
    "bbox_or_point",
    "structured_label_plus_natural_language",
}
CONTROLLED_RESPONSE_CONTRACT = "controlled_natural_language_v1"
CONTROLLED_RESPONSE_TASKS = {"functional_assessment", "abnormality_detection"}
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

SYSTEM_PROMPT = """你是一名接受过经食管超声心动图（TEE）训练的医学视觉分析模型。请仅依据当前提供的图像或视频回答问题，不得利用文件名、路径、样本编号或其他隐藏元数据推断答案。

若任务提供候选选项，必须从当前题目的选项中选择，不得根据其他题目的 option_id 推断本题答案。若任务指定输出协议，必须严格遵守对应格式。

只输出任务指定的最终答案，不输出 Markdown、代码块、分析过程、思维链、前后缀或额外解释。不要虚构图像中不可见的信息。"""

SYSTEM_PROMPT_EN = """You are a medical visual analysis model trained in transesophageal echocardiography (TEE). Answer the question using only the image or video currently provided. Do not infer answers from filenames, paths, sample identifiers, or other hidden metadata.

When candidate options are provided, you must select from the options for the current question. Do not infer an answer from option_id values used in other questions. When an output contract is specified, follow that format exactly.

Output only the requested final answer. Do not output Markdown, code fences, analysis, chain of thought, prefixes, suffixes, or additional explanations. Do not fabricate information that is not visible in the image or video."""

CHOICE_TASK_INSTRUCTIONS = {
    "probe_position": "请观察当前 TEE 图像，根据主要解剖结构、近远场关系和成像视角判断探头位置。",
    "imaging_axis": "请观察当前 TEE 图像，并严格以问题指定的主要目标结构为参照判断轴向或心腔视图。不要把探头位置、标准切面名称和目标结构轴向混为一谈。",
    "cfd_presence": "请完整观察当前 TEE 视频，判断视频中是否实际出现彩色血流多普勒（Color Flow Doppler, CFD）叠加。灰阶回声、伪彩界面、测量标记和单纯高回声结构不等同于 CFD。",
    "standard_imaging_plane": "请综合观察当前 TEE 视频中的稳定解剖结构、结构间相对位置和随心动周期的动态表现，判断最符合的标准 TEE 成像切面。以视频整体为依据，不要只依据单个模糊帧。若当前视频不符合候选标准切面的关键解剖标志，应选择题目中对应的‘其他或非标准 TEE 切面’。",
    "structure_visibility_judgment": "请判断问题指定的目标解剖结构在当前 TEE 图像中是否能够被视觉识别。只有存在足够明确的解剖边界或位置依据时才选择‘可见’；明确未进入视野时选择‘不可见’；疑似存在但无法可靠辨认时选择‘不确定’。",
    "assessability_judgment": """请完整观察当前 TEE 视频，判断视频是否足以回答问题指定的功能或结构评估目标。

判定原则：
- 可评估：目标结构或功能显示充分，可以形成可靠判断；
- 评估受限：可获得部分有效信息，但成像范围、图像质量、动态长度或 CFD 覆盖不足；
- 不可评估：目标结构或必要信息基本未显示，无法形成有效判断。""",
}

CHOICE_TASK_INSTRUCTIONS_EN = {
    "probe_position": "Examine the current TEE image and determine the probe position from the principal anatomical structures, near-field and far-field relationships, and imaging perspective.",
    "imaging_axis": "Examine the current TEE image and determine the axis or chamber view strictly with reference to the primary target structure named in the question. Do not confuse probe position, standard imaging-plane name, and the axis of the target structure.",
    "cfd_presence": "Review the entire current TEE video and determine whether a Color Flow Doppler (CFD) overlay is actually present. Grayscale echocardiography, pseudocolor interface elements, measurement markers, and isolated hyperechoic structures are not CFD.",
    "standard_imaging_plane": "Review the stable anatomical structures, their relative positions, and their dynamic appearance throughout the cardiac cycle in the current TEE video. Determine the best-matching standard TEE imaging plane from the video as a whole, rather than from a single unclear frame. If the video lacks the key anatomical landmarks of the candidate standard views, select the option corresponding to 'other or nonstandard TEE view.'",
    "structure_visibility_judgment": "Determine whether the target anatomical structure named in the question can be visually identified in the current TEE image. Select 'visible' only when sufficiently clear anatomical boundaries or positional evidence are present. Select 'not visible' when the structure is clearly outside the field of view. Select 'uncertain' when the structure may be present but cannot be identified reliably.",
    "assessability_judgment": """Review the entire current TEE video and determine whether it is sufficient to answer the functional or structural assessment target named in the question.

Decision criteria:
- Assessable: the target structure or function is displayed sufficiently to support a reliable judgment.
- Limited: some useful information is available, but imaging coverage, image quality, dynamic duration, or CFD coverage is insufficient.
- Not assessable: the target structure or required information is essentially absent, so no valid judgment can be made.""",
}

TASK_CONTRACTS = {
    "probe_position": ("multiple_choice", "single_representative_image"),
    "imaging_axis": ("multiple_choice", "single_representative_image"),
    "cfd_presence": ("multiple_choice", "video"),
    "standard_imaging_plane": ("multiple_choice", "video"),
    "structure_visibility_judgment": ("multiple_choice", "single_frame_image"),
    "single_structure_localization": ("bbox_or_point", "single_frame_image"),
    "assessability_judgment": (
        "multiple_choice",
        "video_with_view_and_cfd_context",
    ),
    "functional_assessment": (
        "structured_label_plus_natural_language",
        "video_with_view_and_cfd_context",
    ),
    "abnormality_detection": (
        "structured_label_plus_natural_language",
        "video_with_view_and_cfd_context",
    ),
}

LOCALIZATION_PROMPT = """请在当前 TEE 图像中定位问题指定的单个目标结构。输出覆盖该结构可见区域的最小合理矩形框，并给出矩形框中心点。

问题：{question}

坐标要求：
1. 使用归一化坐标，左上角为 (0,0)，右下角为 (1,1)。
2. bbox 格式为 [x1,y1,x2,y2]，必须满足 0<=x1<x2<=1 且 0<=y1<y2<=1。
3. point 格式为 [x,y]，应为 bbox 的中心点。
4. bboxes 和 points 分别只包含上述单个框和单个中心点。
5. target_structure 使用指定英文代码。

结构代码：
主动脉瓣=AV；房间隔=IAS；下腔静脉=IVC；室间隔=IVS；左心房=LA；左心耳=LAA；左心室=LV；左室流出道=LVOT；二尖瓣=MV；右心房=RA；右心室=RV；右室流出道=RVOT；上腔静脉=SVC；三尖瓣=TV；主动脉弓=aortic_arch；主动脉根部=aortic_root；升主动脉=ascending_aorta；降主动脉=descending_aorta；乳头肌=papillary_muscles。

只输出以下 JSON 对象，不得增加、删除或重命名字段：
{{"target_structure":"指定英文代码","bbox":[x1,y1,x2,y2],"point":[x,y],"coordinate_system":"normalized_xyxy_and_xy_in_0_1","bboxes":[[x1,y1,x2,y2]],"points":[[x,y]]}}"""

FUNCTIONAL_ASSESSMENT_PROMPT = """请完整观察当前 TEE 视频，只评估问题指定的目标。请使用标准医学术语描述视频直接支持的结构、运动和血流事实，不扩展到题目目标以外的诊断。

问题：{question}

严格输出两行纯文本，不要输出 JSON、Markdown 或额外内容：
可评估性：可评估|评估受限|不可评估
结论：使用简洁但完整的标准医学术语，列出问题目标范围内所有可明确判断的关键阳性和阴性结构、运动及血流事实；多个事实使用分号分隔

可评估性只能从三个标签中选择一个。可评估或评估受限时，必须报告可判断部分中所有由视频直接支持的关键阳性和阴性事实；不得用“正常”“异常”或“未见明显异常”等笼统表述替代具体事实。能够判断程度或确定性时应明确说明。不可评估时，结论只说明直接可见的成像限制，不得给出确定的目标功能事实。"""

ABNORMALITY_DETECTION_PROMPT = """请完整观察当前 TEE 视频，检测当前可见范围内的视觉异常。只报告视频直接支持的结构、运动、血流、积液、占位或器械相关事实；不要根据模糊帧过度诊断。

问题：{question}

严格输出两行纯文本，不要输出 JSON、Markdown 或额外内容：
明显异常：有|无|不确定|不可评估
结论：使用简洁但完整的标准医学术语，列出当前可见范围内所有由视频直接支持的关键阳性和阴性结构、运动、血流、积液、占位及器械相关事实；多个事实使用分号分隔

状态只能从四个标签中选择一个。“有”时不得遗漏明确可见的主要异常；“无”时应报告具体的关键阴性发现，不得只写“未见明显异常”；“不确定”时只能将疑似异常表述为可能或疑似，不得陈述确定性阳性异常；“不可评估”时只说明重要成像限制，不得陈述阳性异常。能够判断程度或确定性时应明确说明。"""

LOCALIZATION_PROMPT_EN = """Locate the single target structure named in the question in the current TEE image. Output the smallest reasonable rectangle covering the visible portion of the structure and the center point of that rectangle.

Question: {question}

Coordinate requirements:
1. Use normalized coordinates with (0,0) at the upper-left and (1,1) at the lower-right.
2. bbox must use [x1,y1,x2,y2], with 0<=x1<x2<=1 and 0<=y1<y2<=1.
3. point must use [x,y] and must be the center of bbox.
4. bboxes and points must each contain only the single box and center point above.
5. target_structure must use the specified English code.

Structure codes:
aortic valve=AV; interatrial septum=IAS; inferior vena cava=IVC; interventricular septum=IVS; left atrium=LA; left atrial appendage=LAA; left ventricle=LV; left ventricular outflow tract=LVOT; mitral valve=MV; right atrium=RA; right ventricle=RV; right ventricular outflow tract=RVOT; superior vena cava=SVC; tricuspid valve=TV; aortic arch=aortic_arch; aortic root=aortic_root; ascending aorta=ascending_aorta; descending aorta=descending_aorta; papillary muscles=papillary_muscles.

Output only the following JSON object. Do not add, remove, or rename fields:
{{"target_structure":"specified English code","bbox":[x1,y1,x2,y2],"point":[x,y],"coordinate_system":"normalized_xyxy_and_xy_in_0_1","bboxes":[[x1,y1,x2,y2]],"points":[[x,y]]}}"""

FUNCTIONAL_ASSESSMENT_PROMPT_EN = """Review the entire current TEE video and assess only the target named in the question. Use standard clinical terminology for structural, motion, and flow facts directly supported by the video. Do not extend the diagnosis beyond the requested target.

Question: {question}

Output exactly two plain-text lines. Do not output JSON, Markdown, or additional content:
Assessability: assessable|limited|not assessable
Conclusion: concise but complete standard clinical description of all key positive and negative structural, motion, and flow facts directly supported within the requested target; separate multiple facts with semicolons

Select exactly one assessability label. For assessable or limited, report every key positive and negative fact directly supported in the portions that can be assessed; do not replace specific facts with generic statements such as "normal," "abnormal," or "no obvious abnormality." State severity or certainty when it can be determined. For not assessable, describe only the directly visible imaging limitation and do not assert a definite target-function fact."""

ABNORMALITY_DETECTION_PROMPT_EN = """Review the entire current TEE video and detect visually apparent abnormalities within the visible field. Report only structural, motion, flow, effusion, mass, or device-related facts directly supported by the video. Do not overdiagnose an unclear frame.

Question: {question}

Output exactly two plain-text lines. Do not output JSON, Markdown, or additional content:
Visually apparent abnormality: yes|no|uncertain|not assessable
Conclusion: concise but complete standard clinical description of all key positive and negative structural, motion, flow, effusion, mass, and device-related facts directly supported within the visible field; separate multiple facts with semicolons

Select exactly one status label. With yes, do not omit any clearly visible major abnormality. With no, report concrete key negative findings rather than only saying "no obvious abnormality." With uncertain, describe suspected abnormalities only as possible or probable and do not assert a definite positive abnormality. With not assessable, describe only important imaging limitations and do not assert a positive abnormality. State severity or certainty when it can be determined."""

SYSTEM_PROMPTS = {"zh": SYSTEM_PROMPT, "en": SYSTEM_PROMPT_EN}
CHOICE_INSTRUCTIONS_BY_LANGUAGE = {
    "zh": CHOICE_TASK_INSTRUCTIONS,
    "en": CHOICE_TASK_INSTRUCTIONS_EN,
}
LOCALIZATION_PROMPTS = {"zh": LOCALIZATION_PROMPT, "en": LOCALIZATION_PROMPT_EN}
FUNCTIONAL_ASSESSMENT_PROMPTS = {
    "zh": FUNCTIONAL_ASSESSMENT_PROMPT,
    "en": FUNCTIONAL_ASSESSMENT_PROMPT_EN,
}
ABNORMALITY_DETECTION_PROMPTS = {
    "zh": ABNORMALITY_DETECTION_PROMPT,
    "en": ABNORMALITY_DETECTION_PROMPT_EN,
}


@dataclass(frozen=True)
class WorkItem:
    source_path: Path
    source_line: int
    dataset: str
    record: dict[str, Any]


@dataclass(frozen=True)
class ParsedPrediction:
    raw_prediction: Any
    prediction: dict[str, Any] | None
    extraction_method: str | None
    strict_json_format_valid: bool | None
    error: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen AnesBench Visual Perception TEE predictions "
            "with an automatically selected local multimodal model adapter."
        )
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="zh",
        help=(
            "Benchmark and prompt language. Selects language-specific default "
            "input, output, and prompt template paths (default: zh)."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Input JSONL files. When omitted, uses the frozen Visual Perception "
            "TEE file matching --language."
        ),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--api-provider",
        choices=tuple(API_PROVIDER_DEFAULT_MODELS),
        default=None,
        help=(
            "Use an OpenAI-compatible multimodal API for TEE: gemini, gpt, or claude. "
            "The API key is read only from an environment variable."
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
        help="Concurrent remote TEE API requests (default: 4).",
    )
    parser.add_argument(
        "--api-vision-format",
        choices=("openai", "anthropic"),
        default="openai",
        help=(
            "Image block format for the remote endpoint. The Micu OpenAI-compatible "
            "endpoint uses openai image_url blocks by default; anthropic is available "
            "for a native-compatible Claude endpoint."
        ),
    )
    parser.add_argument(
        "--api-image-max-dimension",
        type=positive_int,
        default=1280,
        help="Resize remote image/video frames so the longest side is at most this value.",
    )
    parser.add_argument(
        "--model-adapter",
        choices=ADAPTER_NAMES,
        default="auto",
        help=(
            "Model inference protocol. auto selects from the local config.json "
            "(default: auto)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction JSONL. When omitted, uses a language-specific path "
            "under outputs/qwen3-vl-32b."
        ),
    )
    parser.add_argument("--limit", type=positive_int, help="Run at most N pending items.")
    parser.add_argument("--offset", type=non_negative_int, default=0)
    parser.add_argument(
        "--answer-type",
        action="append",
        choices=sorted(SUPPORTED_ANSWER_TYPES),
        help="Keep one answer type; repeat this option to keep several.",
    )
    parser.add_argument(
        "--task-name", action="append", help="Keep one task_name; repeat to keep several."
    )
    parser.add_argument("--max-new-tokens", type=positive_int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help="Number of same-media items per model.generate call (default: 1).",
    )
    parser.add_argument("--video-num-frames", type=positive_int, default=16)
    parser.add_argument(
        "--video-backend",
        choices=("pyav", "decord", "opencv", "torchvision", "torchcodec"),
        default="pyav",
    )
    parser.add_argument(
        "--llava-video-backend",
        choices=("frames", "codec"),
        default="frames",
        help="LLaVA-OneVision2 video processor backend (default: frames).",
    )
    parser.add_argument(
        "--video-max-pixels",
        type=positive_int,
        default=200704,
        help="LLaVA-OneVision2 per-frame video pixel budget.",
    )
    parser.add_argument(
        "--vision-input-size",
        type=positive_int,
        default=448,
        help="Fleming/InternVL-chat square tile size.",
    )
    parser.add_argument(
        "--image-max-tiles",
        type=positive_int,
        default=12,
        help="Maximum dynamic tiles for a Fleming/InternVL-chat image.",
    )
    parser.add_argument(
        "--video-max-tiles-per-frame",
        type=positive_int,
        default=1,
        help="Maximum dynamic tiles per Fleming/InternVL-chat video frame.",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map, for example auto, balanced, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        type=memory_limit,
        default=None,
        help="Optional Accelerate memory limit for every visible GPU, e.g. 70GiB.",
    )
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable model thinking when its chat template supports it "
            "(default: false)."
        ),
    )
    parser.add_argument("--temperature", type=non_negative_float, default=0.0)
    parser.add_argument("--top-p", type=probability, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file. Otherwise successful records are resumed.",
    )
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry prior error records when resuming (default: true).",
    )
    args = parser.parse_args(argv)
    if args.api_provider:
        args.api_model = args.api_model or API_PROVIDER_DEFAULT_MODELS[args.api_provider]
        args.api_key_env = args.api_key_env or API_PROVIDER_DEFAULT_KEY_ENVS[args.api_provider]
    if not args.inputs:
        args.inputs = list(DEFAULT_INPUTS_BY_LANGUAGE[args.language])
    if args.output is None:
        if args.api_provider:
            model_name = args.api_model.replace("/", "--")
            args.output = (
                PROJECT_ROOT
                / "outputs"
                / model_name
                / f"visual-perception-tee-{args.language}.jsonl"
            )
        else:
            args.output = DEFAULT_OUTPUTS_BY_LANGUAGE[args.language]
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def memory_limit(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*(?:\.[0-9]+)?(?:GiB|MiB|GB|MB)", value):
        raise argparse.ArgumentTypeError(
            "must be a positive memory value such as 70GiB or 24000MiB"
        )
    return value


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


def dataset_name(path: Path) -> str:
    lower = str(path).lower()
    if "waveform" in lower:
        return "Waveform"
    if "tee" in lower:
        return "TEE"
    return path.stem


def validate_task_contract(record: dict[str, Any], where: str = "record") -> None:
    task_name = record.get("task_name")
    if task_name not in TASK_CONTRACTS:
        raise ValueError(f"{where}: unsupported task_name {task_name!r}")
    expected_answer_type, expected_input_mode = TASK_CONTRACTS[task_name]
    answer_type = record.get("answer_type")
    if answer_type != expected_answer_type:
        raise ValueError(
            f"{where}: task {task_name!r} requires answer_type "
            f"{expected_answer_type!r}, got {answer_type!r}"
        )
    media = record.get("media")
    if not isinstance(media, dict):
        raise ValueError(f"{where}: record has no media object")
    input_mode = media.get("input_mode")
    if input_mode != expected_input_mode:
        raise ValueError(
            f"{where}: task {task_name!r} requires media.input_mode "
            f"{expected_input_mode!r}, got {input_mode!r}"
        )


def load_items(paths: list[Path]) -> list[WorkItem]:
    items: list[WorkItem] = []
    seen_ids: dict[str, tuple[Path, int]] = {}
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"input JSONL does not exist: {path}")
        for line_number, record in read_jsonl(path):
            qa_id = record.get("qa_id")
            if not isinstance(qa_id, str) or not qa_id:
                raise ValueError(f"{path}:{line_number}: missing qa_id")
            if qa_id in seen_ids:
                old_path, old_line = seen_ids[qa_id]
                raise ValueError(
                    f"duplicate qa_id {qa_id!r}: {old_path}:{old_line} and {path}:{line_number}"
                )
            seen_ids[qa_id] = (path, line_number)
            answer_type = record.get("answer_type")
            if answer_type not in SUPPORTED_ANSWER_TYPES:
                raise ValueError(f"{path}:{line_number}: unsupported answer_type {answer_type!r}")
            validate_task_contract(record, f"{path}:{line_number}")
            items.append(WorkItem(path, line_number, dataset_name(path), record))
    return items


def load_previous_statuses(path: Path) -> dict[str, str]:
    statuses: dict[str, str] = {}
    if not path.exists():
        return statuses
    for _, record in read_jsonl(path):
        qa_id = record.get("qa_id")
        status = record.get("status")
        if isinstance(qa_id, str) and isinstance(status, str):
            if status == "ok" and record.get("response_format_valid") is False:
                status = "invalid_response"
            if (
                status == "ok"
                and record.get("task_name") in CONTROLLED_RESPONSE_TASKS
                and record.get("response_contract") != CONTROLLED_RESPONSE_CONTRACT
            ):
                status = "invalid_response"
            statuses[qa_id] = status
    return statuses


def resolve_media(record: dict[str, Any], project_root: Path = PROJECT_ROOT) -> tuple[str, Path]:
    media = record.get("media")
    if not isinstance(media, dict):
        raise ValueError("record has no media object")
    input_mode = media.get("input_mode")
    use_video = input_mode in {"video", "video_with_view_and_cfd_context"}
    key = "video_path" if use_video else "image_path"
    raw_path = media.get(key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"media.{key} is required for input_mode={input_mode!r}")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"media file does not exist: {path}")
    return ("video" if use_video else "image"), path


def build_question_prompt(record: dict[str, Any], language: str = "zh") -> str:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported prompt language: {language!r}")
    validate_task_contract(record)
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("record has no question")
    question = question.strip()
    task_name = record["task_name"]

    choice_instructions = CHOICE_INSTRUCTIONS_BY_LANGUAGE[language]
    if task_name in choice_instructions:
        options = record.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"task {task_name!r} requires non-empty options")
        seen_option_ids: set[str] = set()
        for option in options:
            if not isinstance(option, dict):
                raise ValueError("each option must be an object")
            option_id = option.get("option_id")
            text = option.get("text")
            if not isinstance(option_id, str) or not option_id:
                raise ValueError("each option requires a non-empty option_id")
            if not isinstance(text, str) or not text:
                raise ValueError("each option requires non-empty text")
            if option_id in seen_option_ids:
                raise ValueError(f"duplicate option_id {option_id!r}")
            seen_option_ids.add(option_id)
        options_json = json.dumps(options, ensure_ascii=False, separators=(",", ":"))
        if language == "en":
            return (
                f"{choice_instructions[task_name]}\n\n"
                f"Question: {question}\n"
                f"Candidate options: {options_json}\n\n"
                "Select exactly one option and output only the following JSON object:\n"
                '{"option_id":"option letter","text":"exact full text of the selected option"}'
            )
        return (
            f"{choice_instructions[task_name]}\n\n"
            f"问题：{question}\n"
            f"候选选项：{options_json}\n\n"
            "请严格选择一个选项，并只输出以下 JSON 对象：\n"
            '{"option_id":"选项字母","text":"所选选项的完整原文"}'
        )
    if task_name == "single_structure_localization":
        return LOCALIZATION_PROMPTS[language].format(question=question)
    if task_name == "functional_assessment":
        return FUNCTIONAL_ASSESSMENT_PROMPTS[language].format(question=question)
    if task_name == "abnormality_detection":
        return ABNORMALITY_DETECTION_PROMPTS[language].format(question=question)
    raise ValueError(f"unsupported task_name: {task_name!r}")


def build_messages(
    record: dict[str, Any],
    media_type: str,
    media_value: Any,
    language: str = "zh",
) -> list[dict[str, Any]]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported prompt language: {language!r}")
    media_block = {"type": media_type, media_type: media_value}
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPTS[language]}],
        },
        {
            "role": "user",
            "content": [
                media_block,
                {
                    "type": "text",
                    "text": build_question_prompt(record, language=language),
                },
            ],
        },
    ]


def extract_json(text: str) -> Any | None:
    stripped = text.strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"[\[{]", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
                return value
            except json.JSONDecodeError:
                continue
    return None


def adapt_prediction(
    record: dict[str, Any], raw_prediction: Any
) -> tuple[dict[str, Any] | None, str | None]:
    """Wrap the prompt response in the benchmark answer schema without repairing it."""
    if not isinstance(raw_prediction, dict):
        return None, "model response does not contain a JSON object"

    answer_type = record["answer_type"]
    if answer_type == "multiple_choice":
        choice = raw_prediction.get("choice")
        if isinstance(choice, dict):
            selected = choice
        elif "option_id" in raw_prediction or "text" in raw_prediction:
            selected = raw_prediction
        else:
            return None, "choice response requires option_id and text"

        option_id = selected.get("option_id")
        option_text = selected.get("text")
        valid_options = {
            option["option_id"]: option["text"] for option in record.get("options", [])
        }
        if not isinstance(option_id, str) or option_id not in valid_options:
            return None, f"invalid option_id {option_id!r}"
        if not isinstance(option_text, str) or option_text != valid_options[option_id]:
            return None, "option text does not exactly match the selected current option"
        return (
            {
                "choice": {"option_id": option_id, "text": option_text},
                "localization": None,
                "structured_labels": None,
                "natural_language": None,
            },
            None,
        )

    if answer_type == "bbox_or_point":
        localization = raw_prediction.get("localization")
        if not isinstance(localization, dict):
            localization = raw_prediction
        return (
            {
                "choice": None,
                "localization": localization,
                "structured_labels": None,
                "natural_language": None,
            },
            None,
        )

    if answer_type == "structured_label_plus_natural_language":
        if not isinstance(raw_prediction.get("structured_labels"), dict):
            return None, "structured response requires a structured_labels object"
        if not isinstance(raw_prediction.get("natural_language"), str):
            return None, "structured response requires natural_language text"
        return raw_prediction, None

    return None, f"unsupported answer_type {answer_type!r}"


def _option_aliases(option_text: str) -> list[str]:
    aliases = {option_text.strip()}
    prefix = re.split(r"[（(]", option_text, maxsplit=1)[0].strip()
    if prefix:
        aliases.add(prefix)
    for parenthetical in re.findall(r"[（(]([^）)]+)[）)]", option_text):
        aliases.add(parenthetical.strip())
        aliases.update(
            part.strip() for part in re.split(r"[,，/]", parenthetical) if part.strip()
        )
    return sorted(
        (alias for alias in aliases if len(alias) >= 2),
        key=len,
        reverse=True,
    )


def _match_choice_segment(
    record: dict[str, Any], segment: str
) -> dict[str, str] | None:
    cleaned = segment.strip().strip("*_\x60\"'“”‘’[]()（）")
    cleaned = re.sub(
        r"^(?:(?:应该|应当|应)?(?:选择|选)?(?:是|为)?\s*[:：]?\s*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(?:(?:the\s+)?(?:final\s+)?(?:answer|choice|option)"
        r"(?:\s+(?:is|would\s+be))?\s*[:：]?\s*)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    option_by_id = {
        option["option_id"].upper(): option
        for option in record.get("options", [])
        if isinstance(option, dict)
        and isinstance(option.get("option_id"), str)
        and isinstance(option.get("text"), str)
    }
    id_match = re.match(
        r"^(?:(?:选项|option)\s*)?([A-Z])(?=$|[\s.、:：，,）)])",
        cleaned,
        flags=re.IGNORECASE,
    )
    if id_match:
        option = option_by_id.get(id_match.group(1).upper())
        if option is not None:
            return {"option_id": option["option_id"], "text": option["text"]}

    normalized = re.sub(r"\s+", " ", cleaned).casefold()
    alias_candidates: list[tuple[str, dict[str, Any]]] = []
    for option in option_by_id.values():
        alias_candidates.extend(
            (re.sub(r"\s+", " ", alias).casefold(), option)
            for alias in _option_aliases(option["text"])
        )
    for alias, option in sorted(alias_candidates, key=lambda item: len(item[0]), reverse=True):
        if not normalized.startswith(alias):
            continue
        if alias[-1:].isascii() and alias[-1:].isalnum():
            following = normalized[len(alias) : len(alias) + 1]
            if following and following.isascii() and following.isalnum():
                continue
        return {"option_id": option["option_id"], "text": option["text"]}
    return None


def extract_multiple_choice_by_rule(
    record: dict[str, Any], text: str
) -> dict[str, str] | None:
    stripped = text.strip()
    if len(stripped) <= 120:
        direct = _match_choice_segment(record, stripped)
        if direct is not None:
            return direct

    cue_pattern = re.compile(
        r"(?:最终答案|最终选择|答案|应该选择|应选择|我选择|结论)"
        r"\s*(?:应该|应当|应)?(?:选择|选)?(?:是|为)?\s*[:：]?\s*"
        r"([^\n。；]{1,120})",
        flags=re.IGNORECASE,
    )
    for match in reversed(list(cue_pattern.finditer(text))):
        choice = _match_choice_segment(record, match.group(1))
        if choice is not None:
            return choice
    english_cue_pattern = re.compile(
        r"(?:final\s+answer|final\s+choice|answer|choice|I\s+choose|conclusion)"
        r"\s*(?:is|would\s+be)?\s*[:：]?\s*([^\n.;]{1,120})",
        flags=re.IGNORECASE,
    )
    for match in reversed(list(english_cue_pattern.finditer(text))):
        choice = _match_choice_segment(record, match.group(1))
        if choice is not None:
            return choice
    return None


def parse_model_response(
    record: dict[str, Any], text: str, language: str = "zh"
) -> ParsedPrediction:
    task_name = str(record.get("task_name") or "")
    if task_name in {"functional_assessment", "abnormality_detection"}:
        controlled = parse_controlled_response(text, task_name, language)
        if not controlled.valid or controlled.conclusion is None:
            return ParsedPrediction(None, None, None, None, controlled.error)
        target = (
            infer_assessment_target(str(record.get("question") or ""), language)
            if task_name == "functional_assessment"
            else None
        )
        extracted = extract_clinical_facts(
            controlled.conclusion,
            language,
            task_name=task_name,
            assessment_target=target,
        )
        prediction = {
            "choice": None,
            "localization": None,
            "structured_labels": None,
            "natural_language": controlled.conclusion,
            "core_label": controlled.core_label,
            "clinical_facts": [fact.as_list() for fact in extracted.facts],
            "fact_extraction": {
                "warnings": list(extracted.warnings),
                "unresolved_clauses": list(extracted.unresolved_clauses),
                "duplicate_count": extracted.duplicate_count,
                "conflicts": [list(value) for value in extracted.conflicts],
            },
        }
        return ParsedPrediction(
            {"core_label": controlled.core_label, "conclusion": controlled.conclusion},
            prediction,
            "clinical_fact_rules_v1",
            None,
            None,
        )

    raw_prediction = extract_json(text)
    prediction, error = adapt_prediction(record, raw_prediction)
    if prediction is not None:
        return ParsedPrediction(raw_prediction, prediction, "json", True, None)

    if record.get("answer_type") == "multiple_choice":
        extracted_choice = extract_multiple_choice_by_rule(record, text)
        if extracted_choice is not None:
            prediction, rule_error = adapt_prediction(record, extracted_choice)
            if prediction is not None:
                return ParsedPrediction(
                    raw_prediction,
                    prediction,
                    "rule_multiple_choice",
                    False,
                    None,
                )
            error = rule_error

    return ParsedPrediction(raw_prediction, None, None, False, error)


def _api_usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(usage, dict):
        value = usage.get(name)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _api_message_content(message: Any) -> str:
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
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    pieces.append(value)
            else:
                value = getattr(block, "text", None) or getattr(block, "content", None)
                if isinstance(value, str):
                    pieces.append(value)
        return "".join(pieces)
    return ""


def _image_bytes_for_api(path: Path, max_dimension: int) -> tuple[str, bytes]:
    """Return a compact JPEG payload and MIME type for an image path."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for remote TEE image encoding") from exc
    from io import BytesIO

    with Image.open(path) as image:
        image.load()
        image = image.convert("RGB")
        if max(image.size) > max_dimension:
            scale = max_dimension / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    return "image/jpeg", output.getvalue()


def _video_frame_payloads(path: Path, args: argparse.Namespace) -> list[tuple[str, bytes]]:
    """Sample a video into JPEG frames because the compatible API accepts images."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required to send TEE videos to the remote API; "
            "install opencv-python or restrict the run to image tasks"
        ) from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for remote TEE video encoding") from exc
    from io import BytesIO

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open TEE video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"TEE video has no decodable frames: {path}")
        frame_count = min(max(1, args.video_num_frames), total)
        if frame_count == 1:
            indices = [0]
        else:
            indices = [round(i * (total - 1) / (frame_count - 1)) for i in range(frame_count)]
        payloads: list[tuple[str, bytes]] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB")
            if max(image.size) > args.api_image_max_dimension:
                scale = args.api_image_max_dimension / max(image.size)
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=82, optimize=True)
            payloads.append(("image/jpeg", output.getvalue()))
        if not payloads:
            raise RuntimeError(f"TEE video frame decoding returned no frames: {path}")
        return payloads
    finally:
        capture.release()


def _remote_media_payloads(request: GenerationRequest, args: argparse.Namespace) -> list[tuple[str, bytes]]:
    if request.media_type == "image":
        return [_image_bytes_for_api(request.media_path, args.api_image_max_dimension)]
    if request.media_type == "video":
        return _video_frame_payloads(request.media_path, args)
    raise ValueError(f"unsupported remote TEE media type: {request.media_type!r}")


def _remote_messages(
    request: GenerationRequest, args: argparse.Namespace
) -> list[dict[str, Any]]:
    payloads = _remote_media_payloads(request, args)
    content: list[dict[str, Any]] = []
    for mime_type, payload in payloads:
        encoded = base64.b64encode(payload).decode("ascii")
        if args.api_vision_format == "anthropic":
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded,
                    },
                }
            )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                    },
                }
            )
    content.append({"type": "text", "text": request.question_prompt})
    return [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": content},
    ]


class OpenAICompatibleTEEFrontend:
    """Remote TEE frontend using image_url blocks or native Anthropic image blocks."""

    adapter_name = "openai_compatible_vision"
    frontend_kind = "openai_compatible_vision"

    def __init__(self, args: argparse.Namespace):
        if not args.api_provider:
            raise ValueError("--api-provider is required for the remote TEE frontend")
        key_env = args.api_key_env or API_PROVIDER_DEFAULT_KEY_ENVS[args.api_provider]
        candidates = [key_env]
        alias = {"gemini": "GEMINI_API_KEY", "gpt": "GPT_API_KEY", "claude": "CLAUDE_API_KEY"}[args.api_provider]
        if alias not in candidates:
            candidates.append(alias)
        api_key = next((os.environ.get(name) for name in candidates if os.environ.get(name)), None)
        if not api_key:
            raise RuntimeError(
                "API key environment variable is not set (tried "
                + ", ".join(repr(name) for name in candidates)
                + "); keys are never read from command-line values or source files"
            )
        self.key_env = next(name for name in candidates if os.environ.get(name))
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The remote TEE frontend requires the openai package") from exc
        self.provider = args.api_provider
        self.model_type = f"api_{self.provider}"
        self.model_identifier = args.api_model
        self.client = OpenAI(
            base_url=args.api_base_url,
            api_key=api_key,
            timeout=args.api_timeout,
            max_retries=0,
        )
        self.max_retries = args.api_max_retries

    def generate(self, request: GenerationRequest) -> Any:
        messages = _remote_messages(request, self.args)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_identifier,
                    messages=messages,
                    max_tokens=self.args.max_new_tokens,
                )
                choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
                if not choices:
                    raise ValueError("API response contains no choices")
                choice = choices[0]
                message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
                text = _api_message_content(message)
                if not text.strip():
                    raise ValueError("API response returned empty content")
                usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
                return GenerationResult(
                    text=text,
                    input_tokens=_api_usage_value(usage, "prompt_tokens"),
                    generated_tokens=_api_usage_value(usage, "completion_tokens"),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
        assert last_error is not None
        raise RuntimeError(
            f"{self.provider} TEE API request failed after {self.max_retries + 1} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def set_args(self, args: argparse.Namespace) -> None:
        self.args = args


def build_generation_request(
    item: WorkItem, args: argparse.Namespace
) -> GenerationRequest:
    media_type, media_path = resolve_media(item.record)
    return GenerationRequest(
        media_type=media_type,
        media_path=media_path,
        system_prompt=SYSTEM_PROMPTS[args.language],
        question_prompt=build_question_prompt(
            item.record, language=args.language
        ),
    )


def successful_result(
    item: WorkItem,
    args: argparse.Namespace,
    adapter: Any,
    generated: Any,
    elapsed: float,
    batch_size_used: int,
    batch_elapsed: float,
) -> dict[str, Any]:
    text = generated.text
    parsed = parse_model_response(item.record, text, language=args.language)
    return {
        "qa_id": item.record["qa_id"],
        "language": args.language,
        "sample_id": item.record.get("sample_id"),
        "dataset": item.dataset,
        "task_group": item.record.get("task_group"),
        "task_name": item.record.get("task_name"),
        "answer_type": item.record.get("answer_type"),
        "model_adapter": adapter.adapter_name,
        "model_type": adapter.model_type,
        "status": "ok",
        "prediction_text": text,
        "raw_prediction": parsed.raw_prediction,
        "prediction": parsed.prediction,
        "prediction_extraction_method": parsed.extraction_method,
        "strict_json_format_valid": parsed.strict_json_format_valid,
        "response_contract": (
            "controlled_natural_language_v1"
            if item.record.get("task_name")
            in {"functional_assessment", "abnormality_detection"}
            else "json_or_task_rule_v1"
        ),
        "response_format_valid": parsed.prediction is not None,
        "prediction_error": parsed.error,
        "input_tokens": generated.input_tokens,
        "generated_tokens": generated.generated_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "batch_size_used": batch_size_used,
        "batch_elapsed_seconds": round(batch_elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
    }


def run_one(
    item: WorkItem,
    args: argparse.Namespace,
    adapter: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    generated = adapter.generate(build_generation_request(item, args))
    elapsed = time.perf_counter() - started
    return successful_result(
        item, args, adapter, generated, elapsed, 1, elapsed
    )


def run_batch(
    items: list[WorkItem],
    args: argparse.Namespace,
    adapter: Any,
) -> list[dict[str, Any]]:
    requests = [build_generation_request(item, args) for item in items]
    started = time.perf_counter()
    generate_batch = getattr(adapter, "generate_batch", None)
    if callable(generate_batch):
        generated = list(generate_batch(requests))
    else:
        generated = [adapter.generate(request) for request in requests]
    elapsed = time.perf_counter() - started
    if len(generated) != len(items):
        raise RuntimeError(
            f"adapter returned {len(generated)} results for {len(items)} requests"
        )
    per_item_elapsed = elapsed / len(items)
    return [
        successful_result(
            item,
            args,
            adapter,
            result,
            per_item_elapsed,
            len(items),
            elapsed,
        )
        for item, result in zip(items, generated)
    ]


def error_result(
    item: WorkItem,
    exc: Exception,
    elapsed: float,
    language: str = "zh",
    adapter: Any | None = None,
) -> dict[str, Any]:
    return {
        "qa_id": item.record["qa_id"],
        "language": language,
        "sample_id": item.record.get("sample_id"),
        "dataset": item.dataset,
        "task_group": item.record.get("task_group"),
        "task_name": item.record.get("task_name"),
        "answer_type": item.record.get("answer_type"),
        "model_adapter": getattr(adapter, "adapter_name", None),
        "model_type": getattr(adapter, "model_type", None),
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "elapsed_seconds": round(elapsed, 4),
        "source_file": str(item.source_path),
        "source_line": item.source_line,
    }


def load_latest_prediction_records(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    latest: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    if not path.exists():
        return latest, duplicate_count
    for line_number, record in read_jsonl(path):
        qa_id = record.get("qa_id")
        if not isinstance(qa_id, str) or not qa_id:
            raise ValueError(f"{path}:{line_number}: prediction record requires qa_id")
        if qa_id in latest:
            duplicate_count += 1
        latest[qa_id] = record
    return latest, duplicate_count


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def ordered_prediction_records(
    latest: dict[str, dict[str, Any]], qa_id_order: Iterable[str]
) -> list[dict[str, Any]]:
    order = tuple(dict.fromkeys(qa_id_order))
    known_ids = set(order)
    records = [latest[qa_id] for qa_id in order if qa_id in latest]
    records.extend(
        record for qa_id, record in latest.items() if qa_id not in known_ids
    )
    return records


def compact_prediction_file(path: Path, qa_id_order: Iterable[str]) -> int:
    latest, duplicate_count = load_latest_prediction_records(path)
    if duplicate_count:
        atomic_write_jsonl(path, ordered_prediction_records(latest, qa_id_order))
    return duplicate_count


def acquire_output_lock(path: Path) -> Any:
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            f"prediction output is already being written by another process: {path}"
        ) from exc
    return handle


def select_items(items: list[WorkItem], args: argparse.Namespace) -> list[WorkItem]:
    selected = items
    if args.answer_type:
        allowed = set(args.answer_type)
        selected = [item for item in selected if item.record["answer_type"] in allowed]
    if args.task_name:
        allowed_tasks = set(args.task_name)
        selected = [item for item in selected if item.record.get("task_name") in allowed_tasks]
    selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    return selected


def iter_media_batches(
    items: list[WorkItem], batch_size: int
) -> Iterable[list[WorkItem]]:
    batch: list[WorkItem] = []
    batch_media_type: str | None = None
    for item in items:
        media = item.record.get("media") or {}
        input_mode = media.get("input_mode")
        media_type = (
            "video"
            if input_mode in {"video", "video_with_view_and_cfd_context"}
            else "image"
        )
        if batch and (
            len(batch) >= batch_size or media_type != batch_media_type
        ):
            yield batch
            batch = []
        batch.append(item)
        batch_media_type = media_type
    if batch:
        yield batch


def write_run_config(
    args: argparse.Namespace, selected_count: int, adapter_spec: Any
) -> None:
    config_path = args.output.with_suffix(args.output.suffix + ".config.json")
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": args.language,
        "model_path": str(args.model_path.expanduser().resolve()),
        "model_adapter_requested": args.model_adapter,
        "model_adapter": adapter_spec.adapter_name,
        "model_type": adapter_spec.model_type,
        "model_architectures": list(adapter_spec.architectures),
        "inputs": [str(path.expanduser().resolve()) for path in args.inputs],
        "output": str(args.output.expanduser().resolve()),
        "selected_count_before_resume": selected_count,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "video_num_frames": args.video_num_frames,
        "video_backend": args.video_backend,
        "llava_video_backend": args.llava_video_backend,
        "video_max_pixels": args.video_max_pixels,
        "vision_input_size": args.vision_input_size,
        "image_max_tiles": args.image_max_tiles,
        "video_max_tiles_per_frame": args.video_max_tiles_per_frame,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "attn_implementation": args.attn_implementation,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "prompt_template": str(PROMPT_TEMPLATE_PATHS[args.language]),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_inference_locked(args: argparse.Namespace) -> int:
    if args.overwrite and args.output.exists():
        args.output.unlink()
    all_items = load_items(args.inputs)
    items = select_items(all_items, args)
    source_qa_ids = [item.record["qa_id"] for item in all_items]
    removed_records = compact_prediction_file(args.output, source_qa_ids)
    if removed_records:
        print(
            f"Compacted {args.output}: removed {removed_records} superseded "
            "prediction record(s).",
            flush=True,
        )
    latest_predictions, _ = load_latest_prediction_records(args.output)
    previous = load_previous_statuses(args.output)
    if args.retry_errors:
        completed = {qa_id for qa_id, status in previous.items() if status == "ok"}
    else:
        completed = set(previous)
    pending = [item for item in items if item.record["qa_id"] not in completed]
    if not pending:
        print(f"No pending items. Selected={len(items)}, already recorded={len(items)}")
        return 0

    adapter_spec = resolve_adapter_spec(args.model_path, args.model_adapter)
    write_run_config(args, len(items), adapter_spec)
    print(
        f"Loading local model {args.model_path} with adapter "
        f"{adapter_spec.adapter_name} (model_type={adapter_spec.model_type}) "
        f"for {len(pending)} "
        f"{args.language} item(s) "
        f"({len(items) - len(pending)} resumed)...",
        flush=True,
    )
    print(
        f"Execution: batch_size={args.batch_size}, "
        f"device_map={args.device_map}, "
        f"max_memory_per_gpu={args.max_memory_per_gpu or 'automatic'}",
        flush=True,
    )
    adapter = create_adapter(args, adapter_spec)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required; install requirements-inference.txt") from exc

    ok_count = 0
    error_count = 0
    batch_fallbacks = 0
    progress = tqdm(
        total=len(pending),
        desc="AnesBench inference",
        unit="item",
        dynamic_ncols=True,
    )
    for batch in iter_media_batches(pending, args.batch_size):
        batch_started = time.perf_counter()
        try:
            results = run_batch(batch, args, adapter)
        except KeyboardInterrupt:
            raise
        except Exception as batch_exc:
            if len(batch) == 1:
                results = [
                    error_result(
                        batch[0],
                        batch_exc,
                        time.perf_counter() - batch_started,
                        language=args.language,
                        adapter=adapter,
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
                if (
                    torch_module is not None
                    and torch_module.cuda.is_available()
                ):
                    torch_module.cuda.empty_cache()
                results = []
                for item in batch:
                    item_started = time.perf_counter()
                    try:
                        results.append(run_one(item, args, adapter))
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        results.append(
                            error_result(
                                item,
                                exc,
                                time.perf_counter() - item_started,
                                language=args.language,
                                adapter=adapter,
                            )
                        )
        for result in results:
            if result["status"] == "ok":
                ok_count += 1
            else:
                error_count += 1
            latest_predictions[result["qa_id"]] = result
        atomic_write_jsonl(
            args.output,
            ordered_prediction_records(latest_predictions, source_qa_ids),
        )
        progress.update(len(batch))
        progress.set_postfix(
            ok=ok_count,
            errors=error_count,
            batch_fallbacks=batch_fallbacks,
            refresh=False,
        )
    progress.close()

    print(
        f"Finished: ok={ok_count}, errors={error_count}, "
        f"batch_fallbacks={batch_fallbacks}, output={args.output}"
    )
    return 0 if error_count == 0 else 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output = args.output.expanduser().resolve()
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
