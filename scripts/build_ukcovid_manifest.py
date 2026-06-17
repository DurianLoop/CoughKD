"""Build a CoughKD manifest for the UK COVID-19 Vocal Audio Dataset.

The Zenodo open-access release is large, so this builder supports a
metadata-only mode via ``--allow-missing-audio``. That lets us audit labels,
splits, and candidate cough file paths before downloading the 53GB audio
archive.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ID_CANDIDATES = [
    "participant_identifier",
    "participant_id",
    "participant",
    "subject_id",
    "id",
]

LABEL_CANDIDATES = [
    "covid_test_result",
    "sars_cov_2_test_result",
    "sars_cov_2_pcr_result",
    "pcr_result",
    "test_result",
    "covid_status",
    "covid19_status",
    "covid19_test_result",
]

AGE_CANDIDATES = ["age", "age_years", "participant_age"]
SEX_CANDIDATES = ["sex", "gender", "participant_gender"]


def _norm(value: object) -> str:
    return str(value).strip()


def _norm_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).fillna("")
    df = df.rename(columns={col: _norm_col(col) for col in df.columns})
    return df


def _choose_column(columns: Iterable[str], candidates: list[str], name: str) -> str:
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            return candidate
    raise SystemExit(f"Could not infer {name} column. Available columns: {cols}")


def _infer_audio_cols(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    preferred = [
        col
        for col in cols
        if "cough" in col and any(token in col for token in ["path", "file", "wav", "audio"])
    ]
    if preferred:
        return preferred
    fallback = [
        col
        for col in cols
        if any(token in col for token in ["path", "file", "wav"])
        and not any(skip in col for skip in ["speech", "exhale", "exhalation", "breath"])
    ]
    if fallback:
        return fallback
    raise SystemExit(f"Could not infer cough audio path column. Available columns: {cols}")


def _infer_split_cols(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if "split" in col or col in {"randomised", "standard", "matched", "longitudinal"}]


def _label_to_coughkd(raw: object) -> str | None:
    value = _norm(raw).lower()
    if value in {"", "nan", "none", "unknown", "missing", "na"}:
        return None
    if any(token in value for token in ["positive", "detected", "pos", "covid", "sars"]):
        if any(token in value for token in ["negative", "not detected", "not_detected", "no covid"]):
            return "healthy"
        return "covid_positive"
    if any(token in value for token in ["negative", "not_detected", "not detected", "neg", "control"]):
        return "healthy"
    if value in {"1", "true", "yes"}:
        return "covid_positive"
    if value in {"0", "false", "no"}:
        return "healthy"
    return None


def _build_audio_index(dataset_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ("*.wav", "*.flac", "*.ogg", "*.webm", "*.mp3"):
        for path in dataset_dir.rglob(ext):
            index.setdefault(path.name.lower(), path)
    return index


def _resolve_audio(dataset_dir: Path, raw_value: object, audio_index: dict[str, Path] | None = None) -> Path | None:
    value = _norm(raw_value)
    if not value:
        return None
    raw = Path(value.replace("\\", "/"))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([dataset_dir / raw, dataset_dir / raw.name, dataset_dir / "audio" / raw, dataset_dir / "audio" / raw.name])
    if raw.suffix:
        stem = raw.with_suffix("").name
    else:
        stem = raw.name
    for ext in [".wav", ".flac", ".ogg", ".webm", ".mp3"]:
        candidates.extend([dataset_dir / f"{stem}{ext}", dataset_dir / "audio" / f"{stem}{ext}"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if audio_index is not None:
        found = audio_index.get(raw.name.lower())
        if found is not None:
            return found
    return candidates[0] if candidates else None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _symptom_summary(row: dict[str, object]) -> str:
    parts = []
    for key, value in row.items():
        if "symptom" not in key and key not in {"cough", "fever", "fatigue", "shortness_of_breath"}:
            continue
        text = _norm(value)
        if text and text.lower() not in {"0", "false", "no", "nan", "none"}:
            parts.append(f"{key}={text}")
    return ";".join(parts[:20])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-metadata", type=Path, required=True)
    parser.add_argument("--participant-metadata", type=Path, required=True)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ukcovid_open")
    parser.add_argument("--id-col", default="")
    parser.add_argument("--label-col", default="")
    parser.add_argument("--audio-cols", nargs="*", default=[])
    parser.add_argument("--split-col", default="", help="Optional split column from train_test_splits.csv.")
    parser.add_argument("--keep-split-values", nargs="*", default=[], help="Optional split values to keep, e.g. test.")
    parser.add_argument("--allow-missing-audio", action="store_true", help="Write rows even when audio files are not downloaded yet.")
    parser.add_argument("--recursive-audio-search", action="store_true", help="Index audio files recursively under dataset-dir before resolving paths.")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    audio_df = _load_csv(args.audio_metadata)
    participant_df = _load_csv(args.participant_metadata)
    id_col = _norm_col(args.id_col) if args.id_col else _choose_column(audio_df.columns, ID_CANDIDATES, "participant id")
    if id_col not in participant_df.columns:
        participant_id_col = _choose_column(participant_df.columns, ID_CANDIDATES, "participant id")
    else:
        participant_id_col = id_col
    label_col = _norm_col(args.label_col) if args.label_col else _choose_column(participant_df.columns, LABEL_CANDIDATES, "label")
    audio_cols = [_norm_col(col) for col in args.audio_cols] if args.audio_cols else _infer_audio_cols(audio_df.columns)

    split_df = None
    split_col = _norm_col(args.split_col) if args.split_col else ""
    if args.splits:
        split_df = _load_csv(args.splits)
        split_id_col = id_col if id_col in split_df.columns else _choose_column(split_df.columns, ID_CANDIDATES, "split participant id")
        if split_col and split_col not in split_df.columns:
            raise SystemExit(f"split column not found: {split_col}; available={list(split_df.columns)}")
        if not split_col:
            split_cols = _infer_split_cols(split_df.columns)
            split_col = split_cols[0] if split_cols else ""
        keep_cols = [split_id_col] + ([split_col] if split_col else [])
        split_df = split_df[keep_cols].rename(columns={split_id_col: id_col})

    merged = audio_df.merge(participant_df, left_on=id_col, right_on=participant_id_col, suffixes=("_audio", ""), how="inner")
    if split_df is not None:
        merged = merged.merge(split_df, on=id_col, how="left")
    if args.keep_split_values and split_col:
        keep = {value.lower() for value in args.keep_split_values}
        merged = merged[merged[split_col].astype(str).str.lower().isin(keep)]

    rows: list[dict[str, str]] = []
    skipped = {"unknown_label": 0, "missing_audio_value": 0, "missing_audio_file": 0}
    age_col = next((col for col in AGE_CANDIDATES if col in merged.columns), "")
    sex_col = next((col for col in SEX_CANDIDATES if col in merged.columns), "")
    audio_index = _build_audio_index(args.dataset_dir) if args.recursive_audio_search else None

    for item in merged.to_dict(orient="records"):
        label = _label_to_coughkd(item.get(label_col, ""))
        if label is None:
            skipped["unknown_label"] += 1
            continue
        for audio_col in audio_cols:
            raw_audio = item.get(audio_col, "")
            if not _norm(raw_audio):
                skipped["missing_audio_value"] += 1
                continue
            audio = _resolve_audio(args.dataset_dir, raw_audio, audio_index=audio_index)
            if audio is None:
                skipped["missing_audio_file"] += 1
                continue
            if not audio.is_file() and not args.allow_missing_audio:
                skipped["missing_audio_file"] += 1
                continue
            pid = _norm(item[id_col])
            stem = Path(_norm(raw_audio)).stem or audio.stem
            split_value = _norm(item.get(split_col, "test")) if split_col else "test"
            rows.append(
                {
                    "recording_id": f"{args.dataset_name}_{stem}",
                    "subject_id": pid,
                    "dataset": args.dataset_name,
                    "path": _rel(audio, args.root),
                    "label": label,
                    "split": split_value or "test",
                    "age": _norm(item.get(age_col, "")) if age_col else "",
                    "sex": _norm(item.get(sex_col, "")) if sex_col else "",
                    "country": "United Kingdom",
                    "device": "",
                    "symptoms": _symptom_summary(item),
                    "quality_score": "",
                    "cough_detected": "",
                    "source_status": _norm(item.get(label_col, "")),
                }
            )
            if args.max_rows and len(rows) >= args.max_rows:
                break
        if args.max_rows and len(rows) >= args.max_rows:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "subject_id",
        "dataset",
        "path",
        "label",
        "split",
        "age",
        "sex",
        "country",
        "device",
        "symptoms",
        "quality_score",
        "cough_detected",
        "source_status",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = pd.Series([row["label"] for row in rows]).value_counts().to_dict() if rows else {}
    split_counts = pd.Series([row["split"] for row in rows]).value_counts().head(20).to_dict() if rows else {}
    print(
        {
            "written": len(rows),
            "labels": counts,
            "splits": split_counts,
            "audio_cols": audio_cols,
            "id_col": id_col,
            "label_col": label_col,
            "split_col": split_col,
            "skipped": skipped,
            "out": str(args.out),
        }
    )


if __name__ == "__main__":
    main()
