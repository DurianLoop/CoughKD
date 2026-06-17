from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_external_model_set import CHECKPOINTS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "toscovid_calib10_stable_probe_approval"
MANIFEST = ROOT / "manifests" / "toscovid2021_train_calib10_external.csv"
METHODS = ["source_only", "ce", "kd"]
SEEDS = [7, 11, 23]


def _rows() -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            checkpoint = CHECKPOINTS[method][seed]
            out_dir = ROOT / "runs" / f"external_toscovid2021_train_calib10_{method}_seed{seed}"
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "checkpoint_exists": checkpoint.is_file(),
                    "output_dir": str(out_dir.relative_to(ROOT)),
                    "prediction_exists": (out_dir / "predictions.csv").is_file(),
                }
            )
    return rows


def _table(rows: list[dict[str, Any]]) -> str:
    cols = list(rows[0]) if rows else []
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(MANIFEST).fillna("")
    rows = _rows()
    command = (
        "D:\\conda\\envs\\CoughKD\\python.exe -B scripts\\evaluate_external_model_set.py "
        "--manifest manifests\\toscovid2021_train_calib10_external.csv "
        "--target-tag toscovid2021_train_calib10 --root D:\\CoughKD "
        "--device auto --batch-size 16 --skip-existing --methods source_only ce kd"
    )
    manual_driver = "scripts\\run_toscovid_official_calibration_probe_windows.cmd calib10 stable"
    approved_driver = "scripts\\run_toscovid_official_calibration_probe_windows.cmd calib10 stable yes"
    dryrun_driver = "scripts\\run_toscovid_official_calibration_probe_windows.cmd calib10 stable yes dryrun"
    payload = {
        "verdict": "READY_FOR_USER_APPROVAL",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "methods": METHODS,
        "seeds": SEEDS,
        "prediction_jobs": rows,
        "all_checkpoints_exist": all(row["checkpoint_exists"] for row in rows),
        "any_predictions_exist": any(row["prediction_exists"] for row in rows),
        "command": command,
        "manual_driver": manual_driver,
        "approved_noninteractive_driver": approved_driver,
        "dryrun_driver": dryrun_driver,
        "approval_boundary": "Runs inference only on 500 existing TosCOVID calib10 audio rows for 9 method/seed jobs; no training and no data download.",
        "decision_rule": "If the official-test delta is below 1 pp, stop this official-split enhancement route; if it clears 1 pp, promote to a larger stable subset; if it clears 3 pp, consider full-router escalation.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_calib10_stable_probe_approval.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TosCOVID Calib10 Stable Probe Approval Packet",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Manifest: `{payload['manifest']}`",
        f"- Rows: `{payload['rows']}`",
        f"- Subjects: `{payload['subjects']}`",
        f"- Label counts: `{json.dumps(payload['label_counts'], ensure_ascii=False)}`",
        f"- All checkpoints exist: `{payload['all_checkpoints_exist']}`",
        f"- Any predictions already exist: `{payload['any_predictions_exist']}`",
        f"- Approval boundary: {payload['approval_boundary']}",
        f"- Decision rule: {payload['decision_rule']}",
        "",
        "## Command",
        "",
        "Manual command, pauses for confirmation:",
        "",
        "```cmd",
        manual_driver,
        "```",
        "",
        "Non-interactive command after explicit approval:",
        "",
        "```cmd",
        approved_driver,
        "```",
        "",
        "Driver dry-run command:",
        "",
        "```cmd",
        dryrun_driver,
        "```",
        "",
        "Equivalent direct command:",
        "",
        "```cmd",
        command,
        "```",
        "",
        "## Prediction Jobs",
        "",
        _table(rows),
        "",
    ]
    (OUT / "TOSCOVID_CALIB10_STABLE_PROBE_APPROVAL.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(OUT / "TOSCOVID_CALIB10_STABLE_PROBE_APPROVAL.md")


if __name__ == "__main__":
    main()
