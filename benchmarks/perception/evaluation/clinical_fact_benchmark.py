"""Score predictions against explicitly supplied clinician fact triplets."""

from __future__ import annotations

from typing import Any


def evaluate_with_clinical_facts(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    sidecar: list[dict[str, Any]],
    profile: str,
    official_only: bool,
    language: str,
    source_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from .evaluate import evaluate

    facts = {row["qa_id"]: row.get("clinical_facts", []) for row in sidecar}
    missing = {row["qa_id"] for row in gold if row.get("answer_type") == "structured_label_plus_natural_language"} - facts.keys()
    if missing:
        raise ValueError(f"clinical fact references missing for {len(missing)} item(s)")
    aligned = []
    for row in gold:
        if row["qa_id"] in facts:
            row = {**row, "answer": {**row["answer"], "clinical_facts": facts[row["qa_id"]]}}
        aligned.append(row)
    summary, audit = evaluate(aligned, predictions, profile, official_only)
    summary["clinical_facts_source_sha256"] = source_hash
    summary["clinical_facts_language"] = language
    return summary, audit
