"""One-command onboarding for a new external or auxiliary target manifest.

Given a CoughKD-style manifest, this script runs the bounded evidence pipeline:
overlap audit -> external evaluation -> multi-target summaries -> paper tables
-> submission-readiness audit. It does not train new models or search new KD
variants.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], dry_run: bool) -> None:
    print("[run] " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "manifests" / "coswara_cough.csv")
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-hash-overlap", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--subject-aggregate", action="store_true", help="Also aggregate segmented target predictions by subject_id.")
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    source_manifest = args.source_manifest if args.source_manifest.is_absolute() else ROOT / args.source_manifest
    if not manifest.is_file():
        raise SystemExit(f"missing manifest: {manifest}")
    if not source_manifest.is_file():
        raise SystemExit(f"missing source manifest: {source_manifest}")

    if not args.skip_overlap:
        overlap_out = ROOT / "runs" / "overlap_audit" / f"coswara_vs_{args.target_tag}"
        overlap_cmd = [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "audit_manifest_overlap.py"),
            "--source-manifest",
            str(source_manifest),
            "--target-manifest",
            str(manifest),
            "--out",
            str(overlap_out),
        ]
        if args.enable_hash_overlap:
            overlap_cmd.append("--enable-hash")
        _run(overlap_cmd, args.dry_run)

    eval_cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "evaluate_external_model_set.py"),
        "--manifest",
        str(manifest),
        "--target-tag",
        args.target_tag,
        "--root",
        str(args.root),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.skip_existing:
        eval_cmd.append("--skip-existing")
    if args.dry_run:
        eval_cmd.append("--dry-run")
    _run(eval_cmd, False)

    if args.subject_aggregate:
        _run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "aggregate_external_predictions_by_subject.py"),
                "--manifest",
                str(manifest),
                "--target-tag",
                args.target_tag,
            ],
            args.dry_run,
        )

    for script in [
        "summarize_coughkd_guard_multitarget.py",
        "build_multitarget_shift_audit_table.py",
        "build_shift_audit_paper_tables.py",
        "audit_submission_readiness.py",
    ]:
        _run([sys.executable, "-B", str(ROOT / "scripts" / script)], args.dry_run)

    print("Done. Review:")
    print(ROOT / "runs" / "coughkd_guard_multitarget" / "COUGHKD_GUARD_MULTITARGET_AUDIT.md")
    print(ROOT / "runs" / "kd_failure_analysis" / "SHIFT_AUDIT_MULTITARGET_TABLE.md")
    print(ROOT / "runs" / "submission_readiness" / "SUBMISSION_READINESS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
