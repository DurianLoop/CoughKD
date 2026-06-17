"""Check whether UKCOVID split-zip extraction is ready on Windows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = Path(r"D:\CoughKD\external\ukcovid_open")
DEFAULT_INTEGRITY_JSON = (
    ROOT / "runs" / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_integrity.json"
)
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_7z_paths() -> list[Path]:
    candidates: list[Path] = []
    on_path = shutil.which("7z") or shutil.which("7z.exe")
    if on_path:
        candidates.append(Path(on_path))
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "7-Zip" / "7z.exe")
    candidates.extend(
        [
            ROOT.parents[0] / "tools" / "7zip" / "7z.exe",
            ROOT.parents[0] / "tools" / "7-Zip" / "7z.exe",
        ]
    )
    seen: set[str] = set()
    unique = []
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _find_7z(explicit: Path | None) -> tuple[Path | None, list[dict[str, object]]]:
    candidates = [explicit] if explicit else _candidate_7z_paths()
    checked = []
    found: Path | None = None
    for path in candidates:
        if path is None:
            continue
        exists = path.is_file()
        checked.append({"path": str(path), "exists": exists})
        if exists and found is None:
            found = path
    return found, checked


def _cmd_quote(path: Path) -> str:
    return f'"{path}"'


def _markdown_table(rows: list[dict[str, object]]) -> str:
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
    parser.add_argument("--archive-integrity-json", type=Path, default=DEFAULT_INTEGRITY_JSON)
    parser.add_argument("--sevenzip", type=Path, default=None)
    args = parser.parse_args()

    integrity = _read_json(args.archive_integrity_json)
    archive_verdict = str(integrity.get("verdict", "missing"))
    archive_ready = archive_verdict == "READY_FOR_EXTRACTION"
    main_zip = args.dataset_dir / "covid_data.zip"
    main_zip_exists = main_zip.is_file()
    sevenzip, checked_paths = _find_7z(args.sevenzip)
    tool_found = sevenzip is not None

    if not archive_ready:
        verdict = "ARCHIVE_NOT_READY"
    elif not main_zip_exists:
        verdict = "MAIN_SPLIT_ZIP_MISSING"
    elif not tool_found:
        verdict = "EXTRACTION_TOOL_MISSING"
    else:
        verdict = "READY_TO_EXTRACT"

    extract_command = ""
    if sevenzip:
        extract_command = (
            f"{_cmd_quote(sevenzip)} x {_cmd_quote(main_zip)} "
            f"-o{_cmd_quote(args.dataset_dir)} -y"
        )

    payload = {
        "verdict": verdict,
        "dataset_dir": str(args.dataset_dir),
        "archive_integrity_json": str(args.archive_integrity_json),
        "archive_integrity_verdict": archive_verdict,
        "main_split_zip": str(main_zip),
        "main_split_zip_exists": main_zip_exists,
        "sevenzip_found": tool_found,
        "sevenzip_path": str(sevenzip) if sevenzip else "",
        "checked_paths": checked_paths,
        "extract_command": extract_command,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ukcovid_extraction_preflight.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# UKCOVID Extraction Preflight",
        "",
        f"- Verdict: `{verdict}`",
        f"- Dataset dir: `{args.dataset_dir}`",
        f"- Archive integrity verdict: `{archive_verdict}`",
        f"- Main split zip exists: `{main_zip_exists}`",
        f"- 7-Zip found: `{tool_found}`",
        f"- 7-Zip path: `{sevenzip if sevenzip else 'not found'}`",
        "",
        "## Checked 7-Zip Paths",
        "",
        _markdown_table(checked_paths),
        "",
        "## Next Step",
        "",
    ]
    if verdict == "READY_TO_EXTRACT":
        lines.extend(
            [
                "Run this command to extract the verified split archive:",
                "",
                "```cmd",
                extract_command,
                "```",
                "",
                "Then run:",
                "",
                "```cmd",
                "cd /d D:\\CoughKD\\AAAI",
                "scripts\\run_ukcovid_semantic_router_after_audio_windows.cmd",
                "```",
            ]
        )
    elif verdict == "EXTRACTION_TOOL_MISSING":
        lines.extend(
            [
                "Install 7-Zip or pass an explicit executable path with `--sevenzip`.",
                "Do not use Windows `Expand-Archive` as the primary path for this split zip.",
            ]
        )
    elif verdict == "MAIN_SPLIT_ZIP_MISSING":
        lines.append("The archive verifier is ready, but `covid_data.zip` is missing from the dataset directory.")
    else:
        lines.append("Download/verify the split archive first; extraction is intentionally blocked until checksum verification passes.")
    lines.append("")
    (OUT / "UKCOVID_EXTRACTION_PREFLIGHT.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "verdict": verdict,
                "archive_integrity_verdict": archive_verdict,
                "main_split_zip_exists": main_zip_exists,
                "sevenzip_found": tool_found,
                "sevenzip_path": str(sevenzip) if sevenzip else "",
            },
            indent=2,
        )
    )
    print(OUT / "UKCOVID_EXTRACTION_PREFLIGHT.md")


if __name__ == "__main__":
    main()
