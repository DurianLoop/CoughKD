"""Check whether UK COVID-19 audio archive and extracted wav files are ready."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLAN_CSV = ROOT / "runs" / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_files.csv"
TEST_MANIFEST = ROOT / "manifests" / "ukcovid_open_test_external.csv"
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _read_plan() -> list[dict[str, str]]:
    with PLAN_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _archive_status(dataset_dir: Path) -> list[dict[str, object]]:
    rows = []
    for item in _read_plan():
        path = dataset_dir / item["name"]
        expected = int(item["size_bytes"])
        actual = path.stat().st_size if path.is_file() else 0
        rows.append(
            {
                "name": item["name"],
                "expected_bytes": expected,
                "actual_bytes": actual,
                "exists": path.is_file(),
                "complete": actual >= expected,
            }
        )
    return rows


def _build_file_index(dataset_dir: Path) -> set[str]:
    names: set[str] = set()
    for ext in ("*.wav", "*.flac", "*.ogg", "*.webm", "*.mp3"):
        for path in dataset_dir.rglob(ext):
            names.add(path.name.lower())
    return names


def _manifest_audio_status(dataset_dir: Path, manifest: Path) -> dict[str, int]:
    df = pd.read_csv(manifest)
    index = _build_file_index(dataset_dir)
    expected_names = {Path(str(path).replace("\\", "/")).name.lower() for path in df["path"]}
    found = sum(1 for name in expected_names if name in index)
    return {
        "manifest_rows": int(len(df)),
        "manifest_subjects": int(df["subject_id"].nunique()),
        "unique_audio_filenames": int(len(expected_names)),
        "unique_audio_found": int(found),
        "unique_audio_missing": int(len(expected_names) - found),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\CoughKD\external\ukcovid_open"))
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    args = parser.parse_args()

    archive = _archive_status(args.dataset_dir)
    archive_complete = all(item["complete"] for item in archive)
    manifest_status = _manifest_audio_status(args.dataset_dir, args.manifest)
    audio_ready = manifest_status["unique_audio_missing"] == 0
    verdict = "READY_FOR_EVALUATION" if archive_complete and audio_ready else "NOT_READY_AUDIO_MISSING"

    payload = {
        "verdict": verdict,
        "archive_complete": archive_complete,
        "archive_files_complete": sum(1 for item in archive if item["complete"]),
        "archive_files_total": len(archive),
        "archive_missing_or_incomplete": [item for item in archive if not item["complete"]],
        "manifest_audio_status": manifest_status,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ukcovid_audio_ready_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# UK COVID-19 Audio Readiness Check",
        "",
        f"- Verdict: `{verdict}`",
        f"- Archive files complete: `{payload['archive_files_complete']}/{payload['archive_files_total']}`",
        f"- Unique manifest audio files found: `{manifest_status['unique_audio_found']}/{manifest_status['unique_audio_filenames']}`",
        f"- Manifest rows: `{manifest_status['manifest_rows']}`",
        f"- Manifest subjects: `{manifest_status['manifest_subjects']}`",
        "",
        "## Next Step",
        "",
    ]
    if verdict == "READY_FOR_EVALUATION":
        lines.extend(
            [
                "Rebuild the formal manifest without `--allow-missing-audio` and run onboarding:",
                "",
                "```powershell",
                "cd D:\\CoughKD\\AAAI",
                "D:\\conda\\envs\\CoughKD\\python.exe scripts\\build_ukcovid_manifest.py `",
                "  --audio-metadata D:\\CoughKD\\external\\ukcovid_open\\audio_metadata.csv `",
                "  --participant-metadata D:\\CoughKD\\external\\ukcovid_open\\participant_metadata.csv `",
                "  --splits D:\\CoughKD\\external\\ukcovid_open\\train_test_splits.csv `",
                "  --dataset-dir D:\\CoughKD\\external\\ukcovid_open `",
                "  --out manifests\\ukcovid_open_test_external.csv `",
                "  --keep-split-values test `",
                "  --recursive-audio-search",
                "",
                "D:\\conda\\envs\\CoughKD\\python.exe scripts\\onboard_external_target.py `",
                "  --manifest manifests\\ukcovid_open_test_external.csv `",
                "  --target-tag ukcovid_open `",
                "  --skip-existing `",
                "  --device auto `",
                "  --batch-size 16",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "Download/extract the split archive, then rerun this check:",
                "",
                "```cmd",
                "cd /d D:\\CoughKD\\AAAI",
                "set HTTPS_PROXY=http://127.0.0.1:7897",
                "set HTTP_PROXY=http://127.0.0.1:7897",
                "runs\\ukcovid_open_metadata_audit\\download_ukcovid_audio_archive_windows.cmd D:\\CoughKD\\external\\ukcovid_open",
                "```",
            ]
        )
    (OUT / "UKCOVID_AUDIO_READY_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "archive_complete": archive_complete, **manifest_status}, indent=2))
    print(OUT / "UKCOVID_AUDIO_READY_CHECK.md")


if __name__ == "__main__":
    main()
