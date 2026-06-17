from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
TOS_ROOT = WORKSPACE / "external" / "tos_covid19"
CSV_2022 = TOS_ROOT / "tos-covid-2022.csv"
CSV_2021 = TOS_ROOT / "tos-covid-19.csv"
AUDIO_2021 = TOS_ROOT / "audio_2021"
OUT = ROOT / "runs" / "toscovid2022_local_candidate_audit"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        try:
            return str(path.resolve().relative_to(WORKSPACE.resolve()))
        except ValueError:
            return str(path)


def _read_2022() -> pd.DataFrame:
    if not CSV_2022.is_file():
        raise SystemExit(f"missing {CSV_2022}")
    return pd.read_csv(CSV_2022, sep=";", encoding="utf-8-sig").fillna("")


def _audio_keys(audio_root: Path) -> set[str]:
    keys: set[str] = set()
    if not audio_root.exists():
        return keys
    for path in audio_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".ogg", ".wav", ".flac", ".mp3", ".webm"}:
            continue
        keys.add(path.stem.lower())
        for parent in path.parents:
            match = re.fullmatch(r"id\s*0*(\d+)", parent.name.strip().lower())
            if match:
                value = str(int(match.group(1)))
                keys.update({value, f"{int(value):03d}", f"id {int(value):03d}", f"id {int(value)}"})
                break
            if parent == audio_root:
                break
    return keys


def _audio_file_count(audio_root: Path) -> int:
    if not audio_root.exists():
        return 0
    return sum(
        1
        for path in audio_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ogg", ".wav", ".flac", ".mp3", ".webm"}
    )


def _value_counts(series: pd.Series, limit: int = 20) -> dict[str, int]:
    counts = Counter(str(value).strip() for value in series.tolist())
    return dict(counts.most_common(limit))


def build_payload() -> dict[str, Any]:
    df = _read_2022()
    ids = [str(value).strip().lower() for value in df.get("ID", pd.Series(dtype=str)).tolist()]
    id_set = {value for value in ids if value}
    audio_keys = _audio_keys(AUDIO_2021)
    audio_file_count = _audio_file_count(AUDIO_2021)
    overlap = sorted(id_set & audio_keys)
    audio_dirs = sorted(path.name for path in TOS_ROOT.glob("audio*") if path.is_dir())

    csv_2021_rows = None
    if CSV_2021.is_file():
        csv_2021_rows = int(pd.read_csv(CSV_2021).shape[0])

    verdict = "METADATA_ONLY_NOT_LOCAL_AUDIO_TARGET"
    if overlap:
        verdict = "POSSIBLE_AUDIO_OVERLAP_REQUIRES_MANUAL_REVIEW"

    return {
        "verdict": verdict,
        "csv_2022": _rel(CSV_2022),
        "csv_2022_rows": int(df.shape[0]),
        "csv_2022_columns": [str(col) for col in df.columns],
        "csv_2022_unique_ids": len(id_set),
        "result_counts": _value_counts(df["Resultado"]) if "Resultado" in df else {},
        "sample_type_counts": _value_counts(df["Muestra"]) if "Muestra" in df else {},
        "process_counts": _value_counts(df["Proceso"]) if "Proceso" in df else {},
        "csv_2021": _rel(CSV_2021),
        "csv_2021_rows": csv_2021_rows,
        "local_audio_dirs": audio_dirs,
        "audio_2021_file_count": audio_file_count,
        "audio_2021_key_count": len(audio_keys),
        "id_overlap_with_audio_2021_keys": len(overlap),
        "id_overlap_examples": overlap[:10],
        "claim_role": "raw_metadata_only_candidate",
        "decision": (
            "Do not count TosCOVID 2022 as a local third audio target. The local workspace has "
            "the 2022 testing metadata table but no matching 2022 audio directory or audio-path manifest."
        ),
    }


def write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid2022_local_candidate_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    display_columns = [col.encode("ascii", errors="replace").decode("ascii") for col in payload["csv_2022_columns"]]
    lines = [
        "# TosCOVID 2022 Local Candidate Audit",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- 2022 rows: `{payload['csv_2022_rows']}`",
        f"- 2022 unique IDs: `{payload['csv_2022_unique_ids']}`",
        f"- Local audio dirs under `external/tos_covid19`: `{json.dumps(payload['local_audio_dirs'])}`",
        f"- ID overlap with local 2021 audio keys: `{payload['id_overlap_with_audio_2021_keys']}`",
        "",
        "## Fields",
        "",
        "`" + "`, `".join(display_columns) + "`",
        "",
        "## Counts",
        "",
        f"- Resultado: `{json.dumps(payload['result_counts'], ensure_ascii=True)}`",
        f"- Muestra: `{json.dumps(payload['sample_type_counts'], ensure_ascii=True)}`",
        f"- Proceso: `{json.dumps(payload['process_counts'], ensure_ascii=True)}`",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "This keeps the local-only evidence boundary honest: TosCOVID 2022 can be revisited only if matching audio is added locally or explicitly approved for acquisition.",
        "",
    ]
    (OUT / "TOSCOVID2022_LOCAL_CANDIDATE_AUDIT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    write_report(payload)
    print(json.dumps({"verdict": payload["verdict"], "rows": payload["csv_2022_rows"]}, indent=2))
    print(OUT / "TOSCOVID2022_LOCAL_CANDIDATE_AUDIT.md")


if __name__ == "__main__":
    main()
