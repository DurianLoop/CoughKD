"""Dataset manifest schema, validation, and subject-disjoint splitting."""

from __future__ import annotations

import csv
import json
import random
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = ["recording_id", "subject_id", "dataset", "path", "label", "split"]
OPTIONAL_COLUMNS = ["age", "sex", "country", "device", "symptoms", "quality_score"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


@dataclass(frozen=True)
class ManifestIssue:
    severity: str
    message: str


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest is empty")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest missing required columns: {missing}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def write_manifest(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = [col for col in ALL_COLUMNS if any(col in row for row in rows)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def validate_manifest(manifest_path: Path, root: Path) -> tuple[list[ManifestIssue], dict[str, object]]:
    rows = read_manifest(manifest_path)
    issues: list[ManifestIssue] = []

    recording_ids = [row["recording_id"] for row in rows]
    duplicates = [item for item, count in Counter(recording_ids).items() if count > 1]
    for recording_id in duplicates:
        issues.append(ManifestIssue("error", f"duplicate recording_id: {recording_id}"))

    durations: list[float] = []
    short_examples: list[dict[str, object]] = []
    zero_duration_count = 0
    duration_lt_0_5_count = 0
    duration_lt_1_0_count = 0
    for idx, row in enumerate(rows, start=2):
        for col in REQUIRED_COLUMNS:
            if not row.get(col):
                issues.append(ManifestIssue("error", f"row {idx} missing {col}"))
        file_path = root / row.get("path", "")
        if row.get("path") and not file_path.is_file():
            issues.append(ManifestIssue("error", f"missing audio file for {row['recording_id']}: {file_path}"))
        elif row.get("path"):
            try:
                with wave.open(str(file_path), "rb") as handle:
                    duration = handle.getnframes() / handle.getframerate()
                    durations.append(duration)
                    if duration <= 0:
                        zero_duration_count += 1
                    if duration < 0.5:
                        duration_lt_0_5_count += 1
                    if duration < 1.0:
                        duration_lt_1_0_count += 1
                    if duration < 0.5 and len(short_examples) < 10:
                        short_examples.append(
                            {
                                "recording_id": row.get("recording_id", ""),
                                "duration_sec": duration,
                                "path": row.get("path", ""),
                            }
                        )
            except Exception as exc:
                issues.append(ManifestIssue("error", f"unreadable audio for {row['recording_id']}: {exc}"))

    if zero_duration_count:
        issues.append(ManifestIssue("warning", f"{zero_duration_count} recordings have non-positive duration"))
    if duration_lt_0_5_count:
        issues.append(ManifestIssue("warning", f"{duration_lt_0_5_count} recordings are shorter than 0.5 seconds"))

    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subject_id = row.get("subject_id", "")
        split = row.get("split", "")
        if subject_id and split:
            subject_splits[subject_id].add(split)
    leaked = {subject: sorted(splits) for subject, splits in subject_splits.items() if len(splits) > 1}
    for subject, splits in leaked.items():
        issues.append(ManifestIssue("error", f"subject leakage: {subject} appears in splits {splits}"))

    labels_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    datasets_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    for split in sorted({row.get("split", "") for row in rows}):
        split_rows = [row for row in rows if row.get("split", "") == split]
        labels_by_split[split] = dict(Counter(row.get("label", "") for row in split_rows))
        datasets_by_split[split] = dict(Counter(row.get("dataset", "") for row in split_rows))
        subjects_by_split[split] = {row.get("subject_id", "") for row in split_rows if row.get("subject_id", "")}

    summary: dict[str, object] = {
        "num_recordings": len(rows),
        "num_subjects": len({row.get("subject_id", "") for row in rows if row.get("subject_id", "")}),
        "datasets": dict(Counter(row.get("dataset", "") for row in rows)),
        "labels": dict(Counter(row.get("label", "") for row in rows)),
        "splits": dict(Counter(row.get("split", "") for row in rows)),
        "subjects_by_split": {split: len(subjects) for split, subjects in subjects_by_split.items()},
        "labels_by_split": dict(labels_by_split),
        "datasets_by_split": dict(datasets_by_split),
        "duration_sec": _duration_summary(durations),
        "quality": {
            "zero_duration_count": zero_duration_count,
            "duration_lt_0_5_count": duration_lt_0_5_count,
            "duration_lt_1_0_count": duration_lt_1_0_count,
            "short_examples": short_examples,
        },
        "issues": [issue.__dict__ for issue in issues],
    }
    return issues, summary


def _duration_summary(durations: list[float]) -> dict[str, float | int | None]:
    if not durations:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(durations),
        "min": min(durations),
        "max": max(durations),
        "mean": sum(durations) / len(durations),
    }


def write_validation_report(issues: list[ManifestIssue], summary: dict[str, object], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Manifest Validation Report", ""]
    lines.append(f"- Recordings: {summary['num_recordings']}")
    lines.append(f"- Subjects: {summary['num_subjects']}")
    lines.append(f"- Datasets: `{summary['datasets']}`")
    lines.append(f"- Labels: `{summary['labels']}`")
    lines.append(f"- Splits: `{summary['splits']}`")
    lines.append(f"- Subjects by split: `{summary['subjects_by_split']}`")
    lines.append(f"- Labels by split: `{summary['labels_by_split']}`")
    lines.append(f"- Datasets by split: `{summary['datasets_by_split']}`")
    lines.append(f"- Duration seconds: `{summary['duration_sec']}`")
    lines.append(f"- Quality: `{summary.get('quality', {})}`")
    lines.append("")
    if issues:
        lines.append("## Issues")
        for issue in issues:
            lines.append(f"- `{issue.severity}`: {issue.message}")
    else:
        lines.append("No issues found.")
    (out_dir / "manifest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def filter_manifest_rows(
    rows: list[dict[str, str]],
    root: Path,
    min_duration_sec: float | None = None,
    drop_labels: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, object]] = []
    label_drop = drop_labels or set()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        reason = ""
        if row.get("label", "") in label_drop:
            reason = "drop_label"
        elif min_duration_sec is not None:
            file_path = root / row.get("path", "")
            try:
                with wave.open(str(file_path), "rb") as handle:
                    duration = handle.getnframes() / handle.getframerate()
                if duration < min_duration_sec:
                    reason = "short_duration"
            except Exception:
                reason = "unreadable_audio"
        if reason:
            reason_counts[reason] += 1
            if len(dropped) < 20:
                dropped.append({"recording_id": row.get("recording_id", ""), "reason": reason, "path": row.get("path", "")})
            continue
        kept.append(dict(row))
    report = {
        "input_records": len(rows),
        "kept_records": len(kept),
        "dropped_records": len(rows) - len(kept),
        "drop_reasons": dict(reason_counts),
        "drop_examples": dropped,
    }
    return kept, report


def subject_disjoint_split(
    rows: list[dict[str, str]],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> list[dict[str, str]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_subject: list[dict[str, str]] = []
    for row in rows:
        subject_id = row.get("subject_id", "")
        if subject_id:
            by_subject[subject_id].append(row)
        else:
            missing_subject.append(row)

    subjects = sorted(by_subject)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_subjects = len(subjects)
    n_train = max(1, int(round(n_subjects * train_frac))) if n_subjects else 0
    n_val = max(1, int(round(n_subjects * val_frac))) if n_subjects >= 3 else 0
    if n_train + n_val >= n_subjects and n_subjects >= 2:
        n_train = n_subjects - 2 if n_subjects >= 4 else 1
        n_val = 1

    split_by_subject: dict[str, str] = {}
    for subject in subjects[:n_train]:
        split_by_subject[subject] = "train"
    for subject in subjects[n_train : n_train + n_val]:
        split_by_subject[subject] = "val"
    for subject in subjects[n_train + n_val :]:
        split_by_subject[subject] = "test"

    split_rows: list[dict[str, str]] = []
    for subject in subjects:
        for row in by_subject[subject]:
            updated = dict(row)
            updated["split"] = split_by_subject[subject]
            split_rows.append(updated)
    for row in missing_subject:
        updated = dict(row)
        updated["split"] = "pretrain_only"
        split_rows.append(updated)
    return sorted(split_rows, key=lambda row: row["recording_id"])


def check_external_selection_guard(rows: list[dict[str, str]], selection_splits: set[str] | None = None) -> list[ManifestIssue]:
    """Prevent external/test records from being used for model selection.

    Selection splits default to train/val. Any record marked `test`,
    `external`, or `external_test` is forbidden from selection.
    """

    allowed = selection_splits or {"train", "val"}
    forbidden = {"test", "external", "external_test"}
    issues: list[ManifestIssue] = []
    for row in rows:
        split = row.get("split", "")
        if split in forbidden and split in allowed:
            issues.append(
                ManifestIssue(
                    "error",
                    f"external/test split is included in selection scope: {row.get('recording_id', '')}",
                )
            )
    return issues
