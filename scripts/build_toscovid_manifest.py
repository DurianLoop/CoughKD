"""Build a CoughKD manifest for the Buenos Aires Tos COVID-19 dataset.

Dataset page/API:
https://data.buenosaires.gob.ar/dataset/tos-covid-19

The official metadata contains one participant/cough id per row and labels from
RT-PCR ("Detectable" vs "No Detectable"). Audio files are distributed in ZIP
archives as OGG files. This builder can run before audio download with
``--allow-missing-audio`` to create a metadata-only manifest and readiness
report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(r"D:\CoughKD")
FIELDNAMES = [
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
    "source_status",
    "source_subset",
    "source_batch",
]


def _norm_label(value: object) -> str | None:
    raw = str(value).strip().lower()
    if raw == "detectable":
        return "covid_positive"
    if raw in {"no detectable", "no detectale", "no detectado", "negative"}:
        return "healthy"
    return None


def _norm_split(value: object, fallback: str) -> str:
    raw = str(value).strip().lower()
    if raw in {"train", "training"}:
        return "train"
    if raw in {"test", "testing", "valid", "validation", "val"}:
        return "test"
    return fallback


def _norm_sex(value: object) -> str:
    raw = str(value).strip().upper()
    if raw == "F":
        return "female"
    if raw == "M":
        return "male"
    return ""


def _norm_age_bin(value: object) -> str:
    raw = str(value).strip()
    mapping = {
        "1": "0-17",
        "2": "18-39",
        "3": "40-59",
        "4": "60+",
    }
    return mapping.get(raw, raw if raw and raw.lower() != "nan" else "")


def _id_keys(value: object) -> set[str]:
    raw = str(value).strip()
    keys = {raw, raw.lower()}
    if raw.isdigit():
        intval = int(raw)
        keys.update({str(intval), f"{intval:03d}", f"id {intval:03d}", f"id {intval}"})
    return {key for key in keys if key}


def _build_audio_index(audio_roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in audio_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".ogg", ".wav", ".flac", ".mp3", ".webm"}:
                continue
            keys: set[str] = set()
            for parent in path.parents:
                name = parent.name.strip().lower()
                match = re.fullmatch(r"id\s*0*(\d+)", name)
                if match:
                    keys.update(_id_keys(match.group(1)))
                    keys.add(name)
                    break
                if parent == root:
                    break
            if not keys:
                stem = path.stem
                keys = {stem, stem.lower()}
                numbers = re.findall(r"\d+", stem)
                keys.update(numbers)
            for key in keys:
                index.setdefault(key, []).append(path)
    for paths in index.values():
        paths.sort()
    return index


def _read_csv(path: Path, batch: str) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    required = {"id_unico", "genero", "rango_etario", "resultado_hisopado", "subset"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}; available={list(df.columns)}")
    df["source_batch"] = batch
    return df


def _relative_or_abs(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, action="append", required=True, help="Official Tos CSV. Repeat for 2021/2022.")
    parser.add_argument("--batch-name", action="append", default=[], help="Batch label matching each --metadata.")
    parser.add_argument("--audio-root", type=Path, action="append", default=[], help="Directory containing extracted OGG files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "manifests" / "toscovid_external.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / "toscovid_metadata_audit" / "TOSCOVID_METADATA_AUDIT.md")
    parser.add_argument("--fallback-split", default="test")
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Keep only rows whose official subset is test.")
    args = parser.parse_args()

    if args.batch_name and len(args.batch_name) != len(args.metadata):
        raise SystemExit("--batch-name must be omitted or repeated once per --metadata")
    batches = args.batch_name or [f"batch{i + 1}" for i in range(len(args.metadata))]

    frames = [_read_csv(path, batch) for path, batch in zip(args.metadata, batches)]
    df = pd.concat(frames, ignore_index=True)
    audio_index = _build_audio_index(args.audio_root)

    rows: list[dict[str, str]] = []
    skipped = Counter()
    for item in df.to_dict(orient="records"):
        split = _norm_split(item.get("subset", ""), args.fallback_split)
        if args.test_only and split != "test":
            skipped["non_test_split"] += 1
            continue

        label = _norm_label(item.get("resultado_hisopado", ""))
        if label is None:
            skipped["unknown_label"] += 1
            continue

        batch = str(item["source_batch"])
        raw_id = str(item["id_unico"]).strip()
        keys = _id_keys(raw_id)
        audios = next((audio_index[key] for key in keys if key in audio_index), [])
        if not audios and not args.allow_missing_audio:
            skipped["missing_audio"] += 1
            continue

        subject_id = f"toscovid_{batch}_{int(raw_id):04d}" if raw_id.isdigit() else f"toscovid_{batch}_{raw_id}"
        if not audios:
            audios = [None]
        for audio_idx, audio in enumerate(audios):
            path_value = _relative_or_abs(audio, args.root) if audio else ""
            recording_id = f"{subject_id}_{audio_idx:03d}" if audio else subject_id
            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "dataset": "toscovid",
                    "path": path_value,
                    "label": label,
                    "split": split,
                    "age": _norm_age_bin(item.get("rango_etario", "")),
                    "sex": _norm_sex(item.get("genero", "")),
                    "country": "Argentina",
                    "device": "mobile_phone",
                    "symptoms": "",
                    "quality_score": "",
                    "source_status": str(item.get("resultado_hisopado", "")),
                    "source_subset": str(item.get("subset", "")),
                    "source_batch": batch,
                }
            )

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    batch_counts = Counter(row["source_batch"] for row in rows)
    audio_found = sum(1 for row in rows if row["path"])
    summary = {
        "metadata_rows": int(len(df)),
        "written_rows": len(rows),
        "audio_found": audio_found,
        "audio_missing_in_written": len(rows) - audio_found,
        "labels": dict(label_counts),
        "splits": dict(split_counts),
        "batches": dict(batch_counts),
        "skipped": dict(skipped),
        "manifest": str(out),
    }

    report = args.report if args.report.is_absolute() else ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Tos COVID-19 Metadata Audit",
                "",
                f"- Metadata rows: {summary['metadata_rows']}",
                f"- Manifest rows: {summary['written_rows']}",
                f"- Audio found: {summary['audio_found']}",
                f"- Audio missing in written rows: {summary['audio_missing_in_written']}",
                f"- Labels: `{json.dumps(summary['labels'], ensure_ascii=False)}`",
                f"- Splits: `{json.dumps(summary['splits'], ensure_ascii=False)}`",
                f"- Batches: `{json.dumps(summary['batches'], ensure_ascii=False)}`",
                f"- Skipped: `{json.dumps(summary['skipped'], ensure_ascii=False)}`",
                "",
                "## Interpretation",
                "",
                "This dataset is a promising second external COVID-cough target because its labels are tied to RT-PCR results and it provides official train/test subsets. It should be used as an external stress test, not as evidence of clinical diagnostic validity.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (report.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
