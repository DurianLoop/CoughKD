"""Preflight checks before downloading the UK COVID-19 split audio archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_CSV = ROOT / "runs" / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_files.csv"
DEFAULT_DATASET_DIR = Path(r"D:\CoughKD\external\ukcovid_open")
DEFAULT_MANIFEST = ROOT / "manifests" / "ukcovid_open_test_external.csv"
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _archive_rows(dataset_dir: Path, plan_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for item in plan_rows:
        name = item["name"]
        expected = int(item["size_bytes"])
        path = dataset_dir / name
        actual = path.stat().st_size if path.is_file() else 0
        rows.append(
            {
                "name": name,
                "expected_bytes": expected,
                "actual_bytes": actual,
                "remaining_bytes": max(0, expected - actual),
                "exists": path.is_file(),
                "complete": actual >= expected,
            }
        )
    return rows


def _closest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _disk(path: Path) -> dict[str, Any]:
    existing = _closest_existing(path)
    usage = shutil.disk_usage(existing)
    return {
        "checked_path": str(existing),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _manifest_status(manifest: Path) -> dict[str, Any]:
    if not manifest.is_file():
        return {"present": False, "rows": 0, "subjects": 0}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    subjects = {row.get("subject_id", "") for row in rows if row.get("subject_id")}
    return {"present": True, "rows": len(rows), "subjects": len(subjects)}


def _metadata_status(dataset_dir: Path) -> list[dict[str, Any]]:
    names = ["audio_metadata.csv", "participant_metadata.csv", "train_test_splits.csv", "zenodo_record_10043978.json"]
    rows = []
    for name in names:
        path = dataset_dir / name
        rows.append({"name": name, "present": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0})
    return rows


def _gb(value: int | float) -> float:
    return float(value) / (1024**3)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument(
        "--required-free-gb",
        type=float,
        default=120.0,
        help="Conservative free-space target for archive download plus extraction workspace.",
    )
    args = parser.parse_args()

    plan_rows = _read_plan(args.plan_csv)
    archive = _archive_rows(args.dataset_dir, plan_rows)
    disk = _disk(args.dataset_dir)
    metadata = _metadata_status(args.dataset_dir)
    manifest = _manifest_status(args.manifest)

    total_archive = sum(int(row["expected_bytes"]) for row in archive)
    present_bytes = sum(int(row["actual_bytes"]) for row in archive)
    remaining_bytes = sum(int(row["remaining_bytes"]) for row in archive)
    complete_parts = sum(1 for row in archive if row["complete"])
    metadata_ready = all(row["present"] for row in metadata)
    free_gb = _gb(disk["free_bytes"])
    verdict = "READY_TO_DOWNLOAD_OR_RESUME" if metadata_ready and free_gb >= args.required_free_gb else "NOT_READY_PREFLIGHT"

    payload = {
        "verdict": verdict,
        "dataset_dir": str(args.dataset_dir),
        "plan_csv": str(args.plan_csv.relative_to(ROOT)),
        "manifest": str(args.manifest.relative_to(ROOT)) if args.manifest.is_relative_to(ROOT) else str(args.manifest),
        "required_free_gb": args.required_free_gb,
        "disk": disk,
        "archive": {
            "parts_total": len(archive),
            "parts_complete": complete_parts,
            "total_archive_bytes": total_archive,
            "present_archive_bytes": present_bytes,
            "remaining_archive_bytes": remaining_bytes,
            "total_archive_gib": _gb(total_archive),
            "present_archive_gib": _gb(present_bytes),
            "remaining_archive_gib": _gb(remaining_bytes),
        },
        "metadata": metadata,
        "metadata_ready": metadata_ready,
        "manifest": manifest,
        "archive_parts": archive,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ukcovid_audio_download_preflight.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    part_rows = [
        {
            "name": row["name"],
            "expected_gib": f"{_gb(row['expected_bytes']):.3f}",
            "actual_gib": f"{_gb(row['actual_bytes']):.3f}",
            "complete": row["complete"],
        }
        for row in archive
    ]
    metadata_rows = [{"name": row["name"], "present": row["present"], "bytes": row["bytes"]} for row in metadata]
    lines = [
        "# UKCOVID Audio Download Preflight",
        "",
        f"- Verdict: `{verdict}`",
        f"- Dataset dir: `{args.dataset_dir}`",
        f"- Disk checked at: `{disk['checked_path']}`",
        f"- Free space: `{free_gb:.2f} GiB`",
        f"- Conservative required free space: `{args.required_free_gb:.2f} GiB`",
        f"- Archive present: `{_gb(present_bytes):.2f}/{_gb(total_archive):.2f} GiB`",
        f"- Archive remaining: `{_gb(remaining_bytes):.2f} GiB`",
        f"- Archive parts complete: `{complete_parts}/{len(archive)}`",
        f"- Metadata ready: `{metadata_ready}`",
        f"- Manifest rows/subjects: `{manifest['rows']}/{manifest['subjects']}`",
        "",
        "## Metadata Files",
        "",
        _table(metadata_rows),
        "",
        "## Archive Parts",
        "",
        _table(part_rows),
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "READY_TO_DOWNLOAD_OR_RESUME":
        lines.extend(
            [
                "The metadata and free-space preflight are ready for a resumable UKCOVID audio download.",
                "",
                "Next command after explicit approval:",
                "",
                "```cmd",
                "cd /d D:\\CoughKD\\AAAI",
                "set HTTPS_PROXY=http://127.0.0.1:7897",
                "set HTTP_PROXY=http://127.0.0.1:7897",
                "runs\\ukcovid_open_metadata_audit\\download_ukcovid_audio_archive_windows.cmd D:\\CoughKD\\external\\ukcovid_open",
                "```",
            ]
        )
    else:
        lines.append("Do not start the UKCOVID audio download until missing metadata or free-space issues are resolved.")
    lines.append("")
    (OUT / "UKCOVID_AUDIO_DOWNLOAD_PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "free_gib": round(free_gb, 2),
                "required_free_gib": args.required_free_gb,
                "archive_parts_complete": f"{complete_parts}/{len(archive)}",
                "remaining_archive_gib": round(_gb(remaining_bytes), 2),
                "metadata_ready": metadata_ready,
            },
            indent=2,
        )
    )
    print(OUT / "UKCOVID_AUDIO_DOWNLOAD_PREFLIGHT.md")


if __name__ == "__main__":
    main()
