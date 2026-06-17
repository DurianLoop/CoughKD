"""Evaluate the fixed CoughKD model set on an arbitrary external manifest.

This is the next bounded step for KD-Triage: no new KD loss, no retraining, only
external evaluation of already trained compact students on a new target domain.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "evaluate_external_checkpoint.py"


CHECKPOINTS = {
    "ce": {
        7: ROOT / "runs/stage1_panns_response_seed7/checkpoints/ce_student_best.pt",
        11: ROOT / "runs/stage1_panns_response_seed11/checkpoints/ce_student_best.pt",
        23: ROOT / "runs/stage1_panns_response_seed23/checkpoints/ce_student_best.pt",
    },
    "kd": {
        7: ROOT / "runs/stage1_panns_response_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/stage1_panns_response_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/stage1_panns_response_seed23/checkpoints/student_best.pt",
    },
    "source_only": {
        7: ROOT / "runs/stage3b_source_only_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/stage3c_source_only_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/stage3c_source_only_seed23/checkpoints/student_best.pt",
    },
    "tcd_very_strong": {
        7: ROOT / "runs/stage3b_tcd_very_strong_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/stage3c_tcd_very_strong_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/stage3c_tcd_very_strong_seed23/checkpoints/student_best.pt",
    },
    "tcd_conf035": {
        7: ROOT / "runs/stage3b_tcd_conf035_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/stage3c_tcd_conf035_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/stage3c_tcd_conf035_seed23/checkpoints/student_best.pt",
    },
    "candidate_a": {
        7: ROOT / "runs/candidate_a_shortcut_suppressed_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/candidate_a_shortcut_suppressed_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/candidate_a_shortcut_suppressed_seed23/checkpoints/student_best.pt",
    },
    "candidate_b": {
        7: ROOT / "runs/candidate_b_disagreement_gated_seed7/checkpoints/student_best.pt",
        11: ROOT / "runs/candidate_b_disagreement_gated_seed11/checkpoints/student_best.pt",
        23: ROOT / "runs/candidate_b_disagreement_gated_seed23/checkpoints/student_best.pt",
    },
    "candidate_c": {
        7: ROOT / "runs/candidate_c_probe_adv_seed7/student_domain_adv.pt",
        11: ROOT / "runs/candidate_c_probe_adv_seed11/student_domain_adv.pt",
        23: ROOT / "runs/candidate_c_probe_adv_seed23/student_domain_adv.pt",
    },
    "candidate_f_artifact_env_irm_ramp": {
        7: ROOT / "runs/candidate_f_artifact_env_irm_ramp_seed7/checkpoints/student_best.pt",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-tag", required=True, help="Short tag used in output names, e.g. dicova_test.")
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--methods", nargs="*", default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[7, 11, 23], choices=[7, 11, 23])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"missing manifest: {args.manifest}")

    commands: list[list[str]] = []
    for method in args.methods:
        for seed in args.seeds:
            if seed not in CHECKPOINTS[method]:
                print(f"[skip] unsupported seed {method} seed{seed}")
                continue
            checkpoint = CHECKPOINTS[method][seed]
            if not checkpoint.is_file():
                print(f"[skip] missing checkpoint {method} seed{seed}: {checkpoint}")
                continue
            out = ROOT / "runs" / f"external_{args.target_tag}_{method}_seed{seed}"
            metrics = out / "metrics.json"
            if args.skip_existing and metrics.is_file() and metrics.stat().st_size > 0:
                print(f"[skip] existing {metrics}")
                continue
            commands.append(
                [
                    sys.executable,
                    "-B",
                    str(EVAL_SCRIPT),
                    "--manifest",
                    str(args.manifest),
                    "--root",
                    str(args.root),
                    "--checkpoint",
                    str(checkpoint),
                    "--out",
                    str(out),
                    "--device",
                    args.device,
                    "--batch-size",
                    str(args.batch_size),
                ]
            )

    for command in commands:
        print("[run] " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=str(ROOT), check=True)

    print({"commands": len(commands), "dry_run": args.dry_run})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
