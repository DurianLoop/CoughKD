"""Subgroup and clinical-caution reporting helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import binary_classification_report


FORBIDDEN_CLAIMS = [
    "diagnose",
    "diagnosis",
    "diagnostic device",
    "clinical diagnosis",
    "replaces a clinician",
]


def subgroup_report(
    labels: list[int],
    scores: list[float],
    metadata: list[dict[str, str]],
    fields: list[str],
    min_n: int = 5,
) -> dict[str, object]:
    if len(labels) != len(scores) or len(labels) != len(metadata):
        raise ValueError("labels, scores, and metadata must have the same length")
    report: dict[str, object] = {"min_n": min_n, "fields": {}}
    for field in fields:
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(metadata):
            groups[item.get(field, "unknown") or "unknown"].append(idx)
        field_report = {}
        for value, indices in sorted(groups.items()):
            if len(indices) < min_n:
                field_report[value] = {"n": len(indices), "status": "suppressed_small_n"}
                continue
            group_labels = [labels[idx] for idx in indices]
            group_scores = [scores[idx] for idx in indices]
            if len(set(group_labels)) < 2:
                field_report[value] = {"n": len(indices), "status": "suppressed_single_class"}
                continue
            metrics = binary_classification_report(group_labels, group_scores)
            field_report[value] = {"n": len(indices), "status": "reported", "metrics": metrics}
        report["fields"][field] = field_report
    return report


def assert_no_unsupported_clinical_claims(text: str) -> None:
    lowered = text.lower()
    for claim in FORBIDDEN_CLAIMS:
        if claim in lowered:
            raise ValueError(f"unsupported clinical claim found: {claim}")


def write_subgroup_report(report: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subgroup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Subgroup Report", "", "This is a screening research report, not a medical diagnosis.", ""]
    for field, groups in report["fields"].items():
        lines.append(f"## {field}")
        for value, payload in groups.items():
            lines.append(f"- {value}: {payload}")
    content = "\n".join(lines) + "\n"
    assert_no_unsupported_clinical_claims(content.replace("not a medical diagnosis", "screening research only"))
    (out_dir / "subgroup_report.md").write_text(content, encoding="utf-8")
