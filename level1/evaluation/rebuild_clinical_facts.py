#!/usr/bin/env python3
"""Rebuild English facts and compare stored predictions under v1 and en-v2.

No inference or model judge is used. Original datasets, v1 sidecars and model
outputs are read-only inputs. Outputs are separate versioned artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.clinical_facts import (
    FACT_ROOT, extract_clinical_facts, fact_sort_key, sha256_file, validate_fact_set,
    load_fact_resources,
)
from eval.clinical_facts_en import EXTRACTOR_VERSION, extractor_sha256
from eval.clinical_fact_benchmark import evaluate_with_clinical_facts
from scripts.build_tee_clinical_facts import _eligibility, _scoring_facts, core_label

TASKS = ("functional_assessment", "abnormality_detection")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def micro(rows):
    counts = {key: sum((row["components"].get("fact_" + key) or 0) for row in rows) for key in ("tp", "fp", "fn")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {**counts, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def build(gold, old, source_hash):
    ontology, _, _ = load_fact_resources("en")
    new, changes = [], []
    old_index = {r["qa_id"]: r for r in old}
    code_hash = extractor_sha256()
    rules_hash = hashlib.sha256((sha256_file(FACT_ROOT / "rules.en.json") + sha256_file(FACT_ROOT / "learned_aliases.en.json")).encode()).hexdigest()
    for item in gold:
        task = item.get("task_name")
        if task not in TASKS:
            continue
        labels = item["answer"]["structured_labels"]
        target = labels.get("assessment_target") if task == "functional_assessment" else None
        text = item["answer"]["natural_language"]
        extracted = extract_clinical_facts(text, "en", task_name=task, assessment_target=target, extractor_version=EXTRACTOR_VERSION)
        state = core_label(item)
        facts = _scoring_facts(task, state, extracted.facts)
        eligible, reason = _eligibility(task, state, facts)
        validate_fact_set(facts, ontology)
        if eligible and (extracted.unresolved_clauses or extracted.conflicts):
            raise ValueError(f"{item['qa_id']}: unresolved eligible reference: {extracted}")
        if task == "abnormality_detection" and state == "yes" and not facts:
            raise ValueError(f"{item['qa_id']}: positive reference lost all facts")
        record = {
            **old_index[item["qa_id"]],
            "clinical_facts": [f.as_list() for f in sorted(facts, key=fact_sort_key)],
            "fact_scoring_eligible": eligible, "ineligible_reason": reason,
            "extractor_version": EXTRACTOR_VERSION, "extractor_sha256": code_hash,
            "source_dataset_sha256": source_hash, "rules_sha256": rules_hash,
            "freeze_method": "deterministic_english_v2",
        }
        new.append(record)
        before = old_index[item["qa_id"]]
        changes.append({
            "qa_id": item["qa_id"], "task_name": task, "source_text": text,
            "old_facts": before["clinical_facts"], "new_facts": record["clinical_facts"],
            "old_eligible": before["fact_scoring_eligible"], "new_eligible": eligible,
            "core_changed": {tuple(f[:3]) for f in before["clinical_facts"]} != {f.core for f in facts},
            "evidence": [e.as_dict() for e in extracted.evidence],
            "unresolved_clauses": list(extracted.unresolved_clauses),
        })
    return new, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "level_one/TEE/Visual Perception_TEE_en.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/clinical-fact-en-v2")
    parser.add_argument("--sidecar", type=Path, default=ROOT / "level_one/TEE/clinical_facts/en-v2/frozen_en.jsonl")
    parser.add_argument("--predictions", type=Path, nargs="*", help="Defaults to outputs/*/tee-en.jsonl with an existing v1 summary")
    args = parser.parse_args()
    old = read_jsonl(FACT_ROOT / "frozen_en.jsonl")
    gold = read_jsonl(args.gold)
    source_hash = sha256_file(args.gold)
    new, changes = build(gold, old, source_hash)
    # Existing frozen artifacts may only be reused byte-for-byte, never replaced.
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
    if args.sidecar.exists() and args.sidecar.read_text(encoding="utf-8") != payload:
        raise ValueError(f"Frozen artifact differs: {args.sidecar}; use a new version/path")
    if not args.sidecar.exists():
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        with args.sidecar.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    write_jsonl(args.output_dir / "reference-audit.jsonl", changes)
    paths = args.predictions if args.predictions is not None else sorted(
        p for p in (ROOT / "outputs").glob("*/tee-en.jsonl")
        if p.with_name("tee-en.clinical-fact.eval.summary.json").is_file())
    comparisons = []
    for path in paths:
        predictions = read_jsonl(path)
        before, audit_before = evaluate_with_clinical_facts(gold, predictions, old, "tee", False, "en", source_hash)
        after, audit_after = evaluate_with_clinical_facts(gold, predictions, new, "tee", False, "en", source_hash)
        model_dir = args.output_dir / path.parent.name
        after["english_extractor_version"] = EXTRACTOR_VERSION
        after["english_extractor_note"] = "Deterministically rebuilt English references; not an independent physician fact-set review."
        write_json(model_dir / "tee-en.clinical-fact.eval.summary.json", after)
        write_jsonl(model_dir / "tee-en.clinical-fact.eval.jsonl", audit_after)
        write_json(model_dir / "tee-en.v1-recomputed.summary.json", before)
        for task in TASKS:
            old_ids = {r["qa_id"] for r in old if r["task_name"] == task and r["fact_scoring_eligible"]}
            new_ids = {r["qa_id"] for r in new if r["task_name"] == task and r["fact_scoring_eligible"]}
            shared = old_ids & new_ids
            comparisons.append({
                "model": path.parent.name, "task": task, "predictions_sha256": sha256_file(path),
                "old_eligible": len(old_ids), "new_eligible": len(new_ids), "common_eligible": len(shared),
                "old": micro([r for r in audit_before if r["qa_id"] in old_ids]),
                "new": micro([r for r in audit_after if r["qa_id"] in new_ids]),
                "old_common": micro([r for r in audit_before if r["qa_id"] in shared]),
                "new_common": micro([r for r in audit_after if r["qa_id"] in shared]),
            })
    totals = {task: {
        "core_changed": sum(r["core_changed"] for r in changes if r["task_name"] == task),
        "old_eligible": sum(r["old_eligible"] for r in changes if r["task_name"] == task),
        "new_eligible": sum(r["new_eligible"] for r in changes if r["task_name"] == task),
    } for task in TASKS}
    write_json(args.output_dir / "comparison.json", {"extractor_sha256": extractor_sha256(), "references": totals, "models": comparisons})
    lines = ["# English ClinicalFact v2 audit", "",
             "English only. Identical stored predictions; F1 formula, exact core matching, status labels and invalid-response policy unchanged.", "",
             "References were rebuilt from the original English text. This audit verifies extraction, not medical correctness against video. No model judge or inference was run.", "",
             "## Reference changes", "", "```json", json.dumps(totals, indent=2), "```", "",
             "## Micro-F1 (0–1)", "",
             "Common columns use only cases eligible under BOTH versions. Their gold facts can still differ; these are evaluator changes, not model improvements.", "",
             "| Model | Task | v1 | en-v2 | v1 common | en-v2 common |", "|---|---|---:|---:|---:|---:|"]
    for c in comparisons:
        values = ["—" if c[k]["f1"] is None else f"{c[k]['f1']:.4f}" for k in ("old", "new", "old_common", "new_common")]
        lines.append("| " + " | ".join([c["model"], c["task"], *values]) + " |")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": str(args.sidecar), "report": str(args.output_dir / "REPORT.md"), "references": totals, "models": len(paths)}, indent=2))


if __name__ == "__main__":
    main()
