"""Public, deterministic extraction for simple clinical fact statements.

Official benchmark references are clinician annotated and are not distributed.
The extractor is intentionally conservative; unrecognized clauses are surfaced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    subject: str
    attribute: str
    value: str

    def as_list(self) -> list[str]:
        return [self.subject, self.attribute, self.value]


@dataclass(frozen=True)
class Extraction:
    facts: tuple[Fact, ...]
    warnings: tuple[str, ...] = ()
    unresolved_clauses: tuple[str, ...] = ()
    duplicate_count: int = 0
    conflicts: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class ControlledResponse:
    valid: bool
    core_label: str | None
    conclusion: str | None
    error: str | None


def parse_controlled_response(text: str, task_name: str, language: str) -> ControlledResponse:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 2 or ":" not in lines[0] and "：" not in lines[0]:
        return ControlledResponse(False, None, None, "expected two labeled lines")
    label = re.split(r"[:：]", lines[0], maxsplit=1)[-1].strip().lower()
    conclusion = re.split(r"[:：]", lines[1], maxsplit=1)[-1].strip()
    allowed = (
        {"assessable", "limited", "not assessable", "可评估", "评估受限", "不可评估"}
        if task_name == "functional_assessment"
        else {"yes", "no", "uncertain", "not assessable", "有", "无", "不确定", "不可评估"}
    )
    if label not in allowed or not conclusion:
        return ControlledResponse(False, None, None, "invalid label or empty conclusion")
    return ControlledResponse(True, label, conclusion, None)


def infer_assessment_target(question: str, language: str) -> str | None:
    match = re.search(r"\b(LV|RV|LA|RA|MV|AV|aorta|pericardi\w*)\b", question, re.I)
    return match.group(1).upper() if match else None


def extract_clinical_facts(
    text: str, language: str, *, task_name: str, assessment_target: str | None = None
) -> Extraction:
    facts: list[Fact] = []
    unresolved: list[str] = []
    for clause in re.split(r"[;；。]", text):
        clause = clause.strip()
        if not clause:
            continue
        subject = infer_assessment_target(clause, language) or assessment_target
        if not subject:
            unresolved.append(clause)
            continue
        lowered = clause.casefold()
        attribute = next((key for key in ("function", "motion", "regurgitation", "effusion", "size", "flow") if key in lowered), "observation")
        value = next((key for key in ("severely reduced", "moderately reduced", "mildly reduced", "reduced", "dilated", "normal", "absent", "present") if key in lowered), None)
        if value is None:
            unresolved.append(clause)
            continue
        facts.append(Fact(subject, attribute, value))
    unique = tuple(dict.fromkeys(facts))
    return Extraction(unique, unresolved_clauses=tuple(unresolved), duplicate_count=len(facts) - len(unique))
