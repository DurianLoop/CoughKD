"""Summarize UK COVID-19 Vocal Audio metadata and manifest readiness."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"D:\CoughKD\external\ukcovid_open")
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).to_dict().items()}


def _path_exists(rel_path: str) -> bool:
    path = Path(str(rel_path))
    if path.is_absolute():
        return path.is_file()
    return (Path(r"D:\CoughKD") / path).is_file()


def main() -> None:
    audio = pd.read_csv(BASE / "audio_metadata.csv", low_memory=False)
    participant = pd.read_csv(BASE / "participant_metadata.csv", low_memory=False)
    splits = pd.read_csv(BASE / "train_test_splits.csv", low_memory=False)
    all_manifest = pd.read_csv(ROOT / "manifests" / "ukcovid_open_external.csv")
    test_manifest = pd.read_csv(ROOT / "manifests" / "ukcovid_open_test_external.csv")

    path_exists = test_manifest["path"].map(_path_exists)
    summary = {
        "metadata_files": {
            "audio_metadata_rows": int(len(audio)),
            "participant_metadata_rows": int(len(participant)),
            "train_test_split_rows": int(len(splits)),
        },
        "participant_labels": _counts(participant["covid_test_result"]),
        "test_methods": _counts(participant["covid_test_method"]),
        "split_counts": _counts(splits["splits"]),
        "all_manifest": {
            "rows": int(len(all_manifest)),
            "subjects": int(all_manifest["subject_id"].nunique()),
            "labels": _counts(all_manifest["label"]),
            "splits": _counts(all_manifest["split"]),
        },
        "test_manifest": {
            "rows": int(len(test_manifest)),
            "subjects": int(test_manifest["subject_id"].nunique()),
            "labels": _counts(test_manifest["label"]),
            "splits": _counts(test_manifest["split"]),
            "audio_paths_existing": int(path_exists.sum()),
            "audio_paths_missing": int((~path_exists).sum()),
        },
        "recommended_role": "candidate_large_independent_external_after_audio_download",
        "blocking_next_step": "download_and_extract_audio_archive_or_selected_test_audio_files",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# UK COVID-19 Vocal Audio Metadata Audit",
        "",
        "## Verdict",
        "",
        "The metadata and split files are present and usable. The test manifest is large enough to be a candidate second external target, but model evaluation is still blocked until the audio archive or selected test audio files are downloaded and extracted.",
        "",
        "## Metadata",
        "",
        f"- Audio metadata rows: `{summary['metadata_files']['audio_metadata_rows']}`",
        f"- Participant metadata rows: `{summary['metadata_files']['participant_metadata_rows']}`",
        f"- Train/test split rows: `{summary['metadata_files']['train_test_split_rows']}`",
        "",
        "## Test Manifest",
        "",
        f"- Rows/clips: `{summary['test_manifest']['rows']}`",
        f"- Subjects: `{summary['test_manifest']['subjects']}`",
        f"- Labels: `{summary['test_manifest']['labels']}`",
        f"- Existing audio paths: `{summary['test_manifest']['audio_paths_existing']}`",
        f"- Missing audio paths: `{summary['test_manifest']['audio_paths_missing']}`",
        "",
        "## Paper Role",
        "",
        "If audio evaluation is completed, UK COVID-19 Vocal Audio can serve as a large public independent external target. It is stronger than Virufy for the main evidence package and cleaner than DiCOVA as an independent target.",
        "",
        "## Next Command After Audio Extraction",
        "",
        "```powershell",
        "cd D:\\CoughKD\\AAAI",
        "D:\\conda\\envs\\CoughKD\\python.exe scripts\\build_ukcovid_manifest.py `",
        "  --audio-metadata D:\\CoughKD\\external\\ukcovid_open\\audio_metadata.csv `",
        "  --participant-metadata D:\\CoughKD\\external\\ukcovid_open\\participant_metadata.csv `",
        "  --splits D:\\CoughKD\\external\\ukcovid_open\\train_test_splits.csv `",
        "  --dataset-dir D:\\CoughKD\\external\\ukcovid_open `",
        "  --out manifests\\ukcovid_open_test_external.csv `",
        "  --keep-split-values test",
        "```",
        "",
    ]
    (OUT / "UKCOVID_METADATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(OUT / "UKCOVID_METADATA_AUDIT.md")


if __name__ == "__main__":
    main()
