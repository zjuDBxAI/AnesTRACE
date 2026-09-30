#!/usr/bin/env python3
"""Local, bilingual Level Three ablations. Dataset answers are never model inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
LEVEL_TWO_INFERENCE = ROOT / "benchmarks" / "single_point" / "inference"
if str(LEVEL_TWO_INFERENCE) not in sys.path:
    sys.path.insert(0, str(LEVEL_TWO_INFERENCE))

import run_extended as l2
VERSION = "level-three-ablation.v1"
MODES = ("current_only", "state_history", "state_action_history")
HEADINGS = {
    "zh": ("【诊断结论】", "【诊断依据】", "【具体干预】"),
    "en": ("[Clinical Diagnosis]", "[Diagnosis Evidence]", "[Specific Intervention]"),
}
CORE_HEADINGS = {
    "zh": ("【麻醉与用药状态】", "【体征趋势】", "【当前新增手术阶段】"),
    "en": ("[Anesthesia & Medication State]", "[Vital-Sign Trends]", "[New Surgical-Phase Information]"),
}
QUESTIONS = {
    "zh": "请判断当前临床状态，并给出此刻最合适的干预决策。",
    "en": "Assess the current clinical state and provide the most appropriate intervention decision at this time.",
}

# GLM-4.7-Flash is loaded through the shared Level Two Transformers frontend.
# Keep the ablation runner explicit about the native GLM path: no Qwen-style
# assistant prefill or model-specific stop strings are injected.
def create_ablation_frontend(args, spec):
    if spec.model_type == "glm4_moe_lite":
        # GLM owns the assistant boundary and termination in its native chat
        # template. Disable the extra Qwen/Morpheus decorations explicitly.
        return l2.TransformersTextFrontend(
            args, spec, assistant_prefill="", stop_strings=()
        )
    return l2.TransformersTextFrontend(
        args, spec, assistant_prefill="", stop_strings=()
    )
SYSTEM = {
    "zh": "你正在完成麻醉临床决策研究任务。仅依据所提供的当前信息及明确提供的历史信息作答，不假设未提供的检查、干预或结局。历史段落内的相对时间均相对该历史节点，不是当前节点。记录的干预不自动等于最优决策。用中文严格按以下三个标题依次输出，每段非空，不加其他标题或前后说明：\n【诊断结论】\n概括当前主要问题及严重程度。\n【诊断依据】\n给出简洁的关键临床证据，不输出内部思维过程。\n【具体干预】\n给出此刻应执行的具体处理，必要时说明剂量或参数；不编造缺失信息。",
    "en": "You are completing an anesthesia clinical decision research task. Use only the supplied current information and explicitly supplied history. Do not assume unavailable tests, interventions, or outcomes. Relative times inside historical blocks refer to that historical node, not the current node. Observed interventions are not automatically optimal decisions. Respond in English using exactly these three headings in order, with nonempty content and no additional headings or preamble:\n[Clinical Diagnosis]\nSummarize the current main problem and severity.\n[Diagnosis Evidence]\nProvide concise key clinical evidence, not internal chain-of-thought.\n[Specific Intervention]\nState concrete actions appropriate now, with doses or settings where needed; do not invent missing information.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: expected nonempty string")
    return value


def split_core(core, language):
    """Fail closed on unknown top-level sections instead of carrying them into history."""
    require_text(core, "core_context")
    matches = list(re.finditer(r"(?m)^(【[^\n]+】|\[[^\n]+\])\s*$", core))
    allowed = CORE_HEADINGS[language]
    if not matches or core[:matches[0].start()].strip():
        raise ValueError("unrecognized core_context preamble/headings")
    sections = {}
    for i, match in enumerate(matches):
        name = match.group(1)
        if name not in allowed or name in sections:
            raise ValueError(f"unknown or duplicate core heading: {name}")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(core)
        sections[name] = core[match.start():end].strip()
        require_text(core[match.end():end], name)
    if not all(name in sections for name in allowed[:2]):
        raise ValueError("missing required medication/vital-sign section")
    return "\n\n".join(sections[name] for name in allowed[:2])


def timeline(turns):
    times, ordinals = [], []
    for i, turn in enumerate(turns):
        point = turn["decision_point"]
        ordinal = require_text(point["ordinal"], "ordinal")
        if ordinal in ordinals:
            raise ValueError(f"duplicate ordinal: {ordinal}")
        expected = ("operation_start", "anesthesia_start") if i == 0 else ("previous_decision",)
        delta = point["elapsed_seconds"]
        if point["elapsed_from"] not in expected or isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError(f"invalid time at {ordinal}")
        if i and delta <= 0:
            raise ValueError(f"non-increasing time at {ordinal}")
        times.append(float(delta) + (times[-1] if i else 0))
        ordinals.append(ordinal)
    return times


def visible_actions(turns):
    """Only inspect the already visible prefix. Never read acceptable_plans."""
    seen, packages, previous = set(), [], []
    for turn in turns:
        package = turn["input"]["previous_intervention"]
        source, actions = package["source_decision_point"], package["actions"]
        if not isinstance(actions, list) or any(not isinstance(x, str) or not x.strip() for x in actions):
            raise ValueError("previous_intervention.actions must be a string list")
        if source is None:
            if actions:
                raise ValueError("actions without a source decision point")
        else:
            if source not in previous:
                raise ValueError(f"action package refers to non-prior node: {source}")
            key = (source, tuple(actions))
            if key not in seen:
                packages.append({"source_decision_point": source, "actions": actions[:]})
                seen.add(key)
        previous.append(turn["decision_point"]["ordinal"])
    order = {name: i for i, name in enumerate(previous)}
    return sorted(packages, key=lambda p: order[p["source_decision_point"]])


def build_prompt(episode, index, language, mode):
    # Slicing before time, state, and action processing enforces future exclusion.
    prefix = episode["turns"][:index + 1]
    turn = prefix[-1]
    times = timeline(prefix)
    core = require_text(turn["input"]["core_context"], "core_context")
    split_core(core, language)
    static = require_text(episode["patient_information"], "patient_information")
    lab = require_text(turn["input"]["visible_test_results"]["content"], "visible_test_results.content")
    point = turn["decision_point"]
    ordinal, elapsed = point["ordinal"], point["elapsed_seconds"]
    origin = prefix[0]["decision_point"]["elapsed_from"]
    origin_zh = "麻醉开始" if origin == "anesthesia_start" else "手术开始"
    origin_en = "anesthesia start" if origin == "anesthesia_start" else "operation start"
    if language == "zh":
        relative = "距" + origin_zh if index == 0 else "距上一决策点"
        current = f"{static}\n\n【当前决策时间】\n- 当前为{ordinal}；{relative}：{elapsed:g}秒。\n\n{core}\n\n【当前检查结果】\n{lab}"
    else:
        relative = "since " + origin_en if index == 0 else "since the previous decision point"
        current = f"{static}\n\n[Current Decision Time]\n- Current node: {ordinal}; {elapsed:g} seconds {relative}.\n\n{core}\n\n[Current Test Results]\n{lab}"
    history = []
    if mode != "current_only":
        for i, old in enumerate(prefix[:-1]):
            name = old["decision_point"]["ordinal"]
            age = times[-1] - times[i]
            if language == "zh":
                title = f"【历史节点 {name}】\n距当前{age:g}秒；距{origin_zh}{times[i]:g}秒。以下块内相对时间均相对{name}。"
            else:
                title = f"[Historical Node {name}]\n{age:g} seconds before now; {times[i]:g} seconds since {origin_en}. All relative times inside this block refer to {name}."
            history.append(title + "\n" + split_core(old["input"]["core_context"], language))
    packages = visible_actions(prefix) if mode == "state_action_history" and index else []
    if mode == "state_action_history" and index:
        action_lines = ["【已公开实际干预历史】" if language == "zh" else "[Previously Published Observed Interventions]"]
        empty = "未记录实际干预" if language == "zh" else "No observed intervention recorded"
        for package in packages:
            action_lines.append(f"{package['source_decision_point']}:\n" + "\n".join("- " + x for x in (package["actions"] or [empty])))
        if not packages:
            action_lines.append(empty)
        history.append("\n".join(action_lines))
    user = "\n\n".join([current, *history, QUESTIONS[language]])
    messages = [{"role": "system", "content": SYSTEM[language]}, {"role": "user", "content": user}]
    return {
        "messages": messages, "prompt_sha256": digest(messages),
        "history_node_count": index if mode != "current_only" else 0,
        "history_action_package_count": len(packages),
        "history_action_count": sum(len(p["actions"]) for p in packages),
    }


def parse_prediction(text, language, truncated=False):
    final, harmony, error = l2._extract_gpt_oss_final(text)
    final, thinking, think_error = l2._remove_thinking(final)
    errors = [e for e in (error, think_error) if e]
    headings = HEADINGS[language]
    matches = list(re.finditer("(?m)^(" + "|".join(re.escape(h) for h in headings) + r")\s*$", final))
    prediction = None
    if [m.group(1) for m in matches] != list(headings):
        errors.append("expected exactly three headings in order")
    else:
        values = [final[m.end():matches[i + 1].start() if i + 1 < 3 else len(final)].strip() for i, m in enumerate(matches)]
        if final[:matches[0].start()].strip():
            errors.append("unexpected preamble")
        if any(not v for v in values):
            errors.append("empty section")
        if any(re.search(r"(?m)^(?:【[^\n]+】|\[[^\n]+\]|#{1,6}\s+).*$", v) for v in values):
            errors.append("unexpected additional heading")
        prediction = dict(zip(("diagnosis", "diagnosis_evidence", "specific_intervention"), values))
    if truncated:
        errors.append("generation truncated")
    # Never expose an unfinished reasoning block as a clinical prediction.
    if error or think_error:
        prediction, final = None, ""
    return {"final_answer_text": final, "prediction": prediction,
            "response_format_valid": not errors, "format_errors": errors,
            "extraction_method": "harmony_final" if harmony else "thinking_removed" if thinking else "plain"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--language", choices=("zh", "en"), default="en")
    p.add_argument("--ablation-mode", choices=MODES, default=MODES[0])
    p.add_argument("--data-source", choices=("all", "vitaldb", "mover"), default="all")
    p.add_argument("--split", action="append", choices=("train", "validation", "test"))
    p.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "Qwen" / "Qwen3-8B",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--offset", type=l2.non_negative_int, default=0)
    p.add_argument("--limit", type=l2.positive_int)
    p.add_argument("--batch-size", type=l2.positive_int, default=1)
    p.add_argument("--max-new-tokens", type=l2.positive_int, default=3072)
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--attn-implementation", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature", type=l2.non_negative_float)
    p.add_argument("--top-p", type=l2.probability)
    p.add_argument("--top-k", type=l2.positive_int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.input = (args.input or ROOT / "level_three" / f"Level_three_v3_{args.language}.jsonl").expanduser().resolve()
    args.model_path = args.model_path.expanduser().resolve()
    args.output = (args.output or ROOT / "outputs" / "level_three_ablation" / args.model_path.name / args.language / args.ablation_mode / "predictions.jsonl").expanduser().resolve()
    return args


def prepare(args):
    rows, ids, turn_ids = [], set(), set()
    for line, raw in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        episode = json.loads(raw)
        eid = require_text(episode["episode_id"], "episode_id")
        if eid in ids:
            raise ValueError(f"duplicate episode: {eid}")
        ids.add(eid)
        source = episode["data_source"]
        if source not in ("VitalDB_INSPIRE", "Mover_EPIC"):
            raise ValueError(f"unknown data source: {source}")
        wanted = {"vitaldb": "VitalDB_INSPIRE", "mover": "Mover_EPIC"}.get(args.data_source)
        if wanted and source != wanted or args.split and episode["split"] not in args.split:
            continue
        if not isinstance(episode["turns"], list) or not episode["turns"]:
            raise ValueError(f"empty turns: {eid}")
        rows.append((line, episode))
    rows = rows[args.offset:args.offset + args.limit if args.limit is not None else None]
    items = []
    for line, episode in rows:
        for i, turn in enumerate(episode["turns"]):
            tid = require_text(turn["turn_id"], "turn_id")
            if tid in turn_ids:
                raise ValueError(f"duplicate turn_id: {tid}")
            turn_ids.add(tid)
            items.append({"episode_id": episode["episode_id"], "turn_id": tid,
                          "decision_point": turn["decision_point"], "data_source": episode["data_source"],
                          "split": episode["split"], "source_line": line, "source_file": str(args.input),
                          "language": args.language, "ablation_mode": args.ablation_mode,
                          **build_prompt(episode, i, args.language, args.ablation_mode)})
    return rows, items


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_jsonl(path, records):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def run_config(args, items):
    return {"schema_version": VERSION, "prompt_version": VERSION,
            "input_path": str(args.input), "input_sha256": file_hash(args.input),
            "model_path": str(args.model_path), "language": args.language, "ablation_mode": args.ablation_mode,
            "generation_options": l2.resolve_generation_options(args),
            "enable_thinking": args.enable_thinking, "seed": args.seed,
            "dtype": args.dtype, "device_map": args.device_map, "attn_implementation": args.attn_implementation,
            "batch_size": args.batch_size, "data_source": args.data_source,
            "split": sorted(set(args.split or [])), "offset": args.offset, "limit": args.limit,
            "selection_sha256": digest([(x["turn_id"], x["prompt_sha256"]) for x in items]),
            "current_embedded_recent_medications_retained": True, "historical_labs_included": False,
            "historical_predictions_included": False, "assistant_prefill": "", "stop_strings": []}


def resume_records(path, config_path, config, items, overwrite=False):
    if overwrite:
        return {}
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            changed = sorted(k for k in config.keys() | saved.keys() if config.get(k) != saved.get(k))
            raise ValueError(f"resume configuration mismatch: {changed}; use a separate output or --overwrite")
    elif path.exists():
        raise ValueError("existing predictions have no run configuration; refusing unsafe resume")
    existing, expected = {}, {x["turn_id"]: x for x in items}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            tid = record["turn_id"]
            if tid not in expected or record["prompt_sha256"] != expected[tid]["prompt_sha256"]:
                raise ValueError(f"resume prompt mismatch: {tid}")
            if tid in existing:
                raise ValueError(f"duplicate saved turn: {tid}")
            existing[tid] = record
    return existing


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run and args.overwrite and args.output.exists():
        raise ValueError("--dry-run --overwrite cannot replace configuration for existing predictions; use a separate output")
    rows, items = prepare(args)
    if not items:
        raise ValueError("no episodes selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_suffix(".run_config.json")
    prompts_path = args.output.with_suffix(".prompts.jsonl")
    stats_path = args.output.with_suffix(".stats.json")
    for path in (args.output, config_path, prompts_path, stats_path):
        if path == args.input:
            raise ValueError("output must not overwrite input")
    config = run_config(args, items)
    # The lock prevents two processes from interleaving checkpoints for this output.
    import fcntl
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = resume_records(args.output, config_path, config, items, args.overwrite)
        write_json(config_path, config)
        write_jsonl(prompts_path, items)
        started = time.monotonic()
        if args.dry_run:
            summary = {"dry_run": True, "episodes": len(rows), "decision_points": len(items),
                       "data_sources": dict(Counter(r["data_source"] for _, r in rows)),
                       "nodes_per_episode": dict(Counter(len(r["turns"]) for _, r in rows)),
                       "prompt_characters": {"min": min(len(x["messages"][1]["content"]) for x in items),
                                             "max": max(len(x["messages"][1]["content"]) for x in items)}}
            write_json(args.output.with_suffix(".dry_run.json"), summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            return 0
        pending = [x for x in items if x["turn_id"] not in existing or
                   args.retry_errors and existing[x["turn_id"]]["status"] in ("error", "context_overflow")]
        if args.overwrite:
            write_jsonl(args.output, [])
        frontend = None
        if pending:
            spec = l2.resolve_model_spec(args.model_path)
            frontend = create_ablation_frontend(args, spec)

        def save(item, result=None, error=None, elapsed=0, batch_count=1):
            record = {k: v for k, v in item.items() if k != "messages"}
            record.update(schema_version=VERSION, model_path=str(args.model_path),
                          model_type=frontend.model_type, frontend=frontend.frontend_kind,
                          created_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
                          elapsed_seconds_semantics="batch_wall_time_divided_by_batch_count", batch_count=batch_count)
            if error is not None:
                overflow = "context window" in str(error)
                record.update(status="context_overflow" if overflow else "error", prediction_error=str(error),
                              prediction=None, prediction_text="", input_tokens=None, generated_tokens=None,
                              truncated=False, response_format_valid=False)
            else:
                parsed = parse_prediction(result.text, args.language, result.truncated)
                record.update(parsed, status="ok" if parsed["response_format_valid"] else "invalid_response",
                              prediction_text=result.text, input_tokens=result.input_tokens,
                              generated_tokens=result.generated_tokens, truncated=result.truncated)
            existing[item["turn_id"]] = record
            write_jsonl(args.output, [existing[x["turn_id"]] for x in items if x["turn_id"] in existing])
            print(f"[{len(existing)}/{len(items)}] {item['turn_id']}: {record['status']}", flush=True)

        def generate(batch):
            # A batch-level overflow/OOM must not mark its shorter siblings as failures.
            from transformers import set_seed
            seed_material = [x["turn_id"] for x in batch]
            set_seed((args.seed + int(digest(seed_material)[:8], 16)) % (2**32))
            began = time.monotonic()
            try:
                results = frontend.generate_batch([x["messages"] for x in batch], args)
                if len(results) != len(batch):
                    raise RuntimeError("generation result count mismatch")
            except Exception as exc:
                if len(batch) > 1:
                    if frontend.torch.cuda.is_available():
                        frontend.torch.cuda.empty_cache()
                    for item in batch:
                        generate([item])
                else:
                    save(batch[0], error=exc, elapsed=time.monotonic() - began)
                return
            elapsed = (time.monotonic() - began) / len(batch)
            for item, result in zip(batch, results):
                save(item, result, elapsed=elapsed, batch_count=len(batch))

        for start in range(0, len(pending), args.batch_size):
            generate(pending[start:start + args.batch_size])
        statuses = dict(Counter(x["status"] for x in existing.values()))
        write_json(stats_path, {"episodes": len(rows), "decision_points": len(items),
                               "predictions": len(existing), "status_counts": statuses,
                               "requests_this_run": len(pending), "elapsed_seconds_this_run": time.monotonic() - started,
                               "input_sha256_unchanged": file_hash(args.input) == config["input_sha256"]})
        return 1 if any(k != "ok" and n for k, n in statuses.items()) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
