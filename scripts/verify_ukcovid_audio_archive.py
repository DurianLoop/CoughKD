"""Verify UK COVID-19 split audio archive sizes and checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLAN_CSV = ROOT / "runs" / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_files.csv"
DEFAULT_DATASET_DIR = Path(r"D:\CoughKD\external\ukcovid_open")
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024 * 8)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _expected_md5(checksum: str) -> str:
    checksum = checksum.strip()
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].lower()
    return checksum.lower()


def _verify(dataset_dir: Path, plan_rows: list[dict[str, str]], skip_hash: bool) -> list[dict[str, Any]]:
    rows = []
    for item in plan_rows:
        path = dataset_dir / item["name"]
        expected_bytes = int(item["size_bytes"])
        actual_bytes = path.stat().st_size if path.is_file() else 0
        complete_size = path.is_file() and actual_bytes == expected_bytes
        expected_hash = _expected_md5(item.get("checksum", ""))
        actual_hash = ""
        checksum_match: bool | None = None
        if complete_size and expected_hash and not skip_hash:
            actual_hash = _md5(path)
            checksum_match = actual_hash == expected_hash
        elif complete_size and expected_hash and skip_hash:
            checksum_match = None
        elif complete_size and not expected_hash:
            checksum_match = None
        else:
            checksum_match = False
        rows.append(
            {
                "name": item["name"],
                "exists": path.is_file(),
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "size_match": complete_size,
                "expected_md5": expected_hash,
                "actual_md5": actual_hash,
                "checksum_match": checksum_match,
            }
        )
    return rows


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
    parser.add_argument("--plan-csv", type=Path, default=PLAN_CSV)
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="Only check file sizes. Use for a fast dry check; do not use before extraction.",
    )
    args = parser.parse_args()

    rows = _verify(args.dataset_dir, _read_plan(args.plan_csv), args.skip_hash)
    missing_or_incomplete = [row for row in rows if not row["size_match"]]
    checksum_failures = [
        row for row in rows if row["size_match"] and row["checksum_match"] is False
    ]
    checksum_pending = [
        row for row in rows if row["size_match"] and row["expected_md5"] and row["checksum_match"] is None
    ]
    if missing_or_incomplete:
        verdict = "NOT_READY_MISSING_ARCHIVE"
    elif checksum_failures:
        verdict = "ARCHIVE_CHECKSUM_FAILED"
    elif checksum_pending:
        verdict = "CHECKSUM_NOT_RUN"
    else:
        verdict = "READY_FOR_EXTRACTION"

    payload = {
        "verdict": verdict,
        "dataset_dir": str(args.dataset_dir),
        "plan_csv": str(args.plan_csv.relative_to(ROOT)),
        "skip_hash": args.skip_hash,
        "parts_total": len(rows),
        "parts_size_complete": sum(1 for row in rows if row["size_match"]),
        "parts_checksum_match": sum(1 for row in rows if row["checksum_match"] is True),
        "missing_or_incomplete": missing_or_incomplete,
        "checksum_failures": checksum_failures,
        "checksum_pending": checksum_pending,
        "parts": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ukcovid_audio_archive_integrity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_rows = [
        {
            "name": row["name"],
            "exists": row["exists"],
            "size_match": row["size_match"],
            "checksum_match": row["checksum_match"],
            "actual_bytes": row["actual_bytes"],
        }
        for row in rows
    ]
    lines = [
        "# UKCOVID Audio Archive Integrity",
        "",
        f"- Verdict: `{verdict}`",
        f"- Dataset dir: `{args.dataset_dir}`",
        f"- Skip hash: `{args.skip_hash}`",
        f"- Size-complete parts: `{payload['parts_size_complete']}/{payload['parts_total']}`",
        f"- Checksum-matched parts: `{payload['parts_checksum_match']}/{payload['parts_total']}`",
        f"- Missing/incomplete parts: `{len(missing_or_incomplete)}`",
        f"- Checksum failures: `{len(checksum_failures)}`",
        "",
        "## Parts",
        "",
        _table(report_rows),
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "READY_FOR_EXTRACTION":
        lines.append("All split archive parts match the expected size and MD5 checksum. It is safe to extract the archive.")
    elif verdict == "CHECKSUM_NOT_RUN":
        lines.append("All split archive parts match expected sizes, but checksum verification was skipped. Run again without `--skip-hash` before extraction.")
    elif verdict == "ARCHIVE_CHECKSUM_FAILED":
        lines.append("At least one complete archive part failed MD5 verification. Redownload the failed part before extraction.")
    else:
        lines.append("Archive parts are missing or incomplete. Continue/resume the download before extraction.")
    lines.append("")
    (OUT / "UKCOVID_AUDIO_ARCHIVE_INTEGRITY.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "parts_size_complete": f"{payload['parts_size_complete']}/{payload['parts_total']}",
                "parts_checksum_match": f"{payload['parts_checksum_match']}/{payload['parts_total']}",
                "missing_or_incomplete": len(missing_or_incomplete),
                "checksum_failures": len(checksum_failures),
            },
            indent=2,
        )
    )
    print(OUT / "UKCOVID_AUDIO_ARCHIVE_INTEGRITY.md")


if __name__ == "__main__":
    main()
