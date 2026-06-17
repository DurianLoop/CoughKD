from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "transfer_gap_diagnostics"
SEEDS = (7, 11, 23)
CLASSES = ["covid_positive", "covid_recovered", "exposed", "healthy", "respiratory_illness"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            probs = [float(row[f"prob_{label}"]) for label in CLASSES if f"prob_{label}" in row]
            if len(probs) != len(CLASSES):
                continue
            rows[row["recording_id"]] = {
                "true_label": row.get("true_label", ""),
                "pred_label": row.get("pred_label", ""),
                "probs": probs,
            }
    return rows


def _entropy(probs: list[float]) -> float:
    return -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)


def _margin(probs: list[float]) -> float:
    ordered = sorted(probs, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _hist(rows: dict[str, dict[str, Any]]) -> list[float]:
    total = max(1, len(rows))
    sums = [0.0 for _ in CLASSES]
    for row in rows.values():
        for idx, prob in enumerate(row["probs"]):
            sums[idx] += prob
    return [value / total for value in sums]


def _hard_hist(rows: dict[str, dict[str, Any]]) -> list[float]:
    total = max(1, len(rows))
    counts = [0.0 for _ in CLASSES]
    for row in rows.values():
        if row["pred_label"] in CLASSES:
            counts[CLASSES.index(row["pred_label"])] += 1.0
    return [value / total for value in counts]


def _l1(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(_fmt(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _external_path(target: str, method: str, seed: int) -> Path:
    if target == "coughvid":
        if seed == 7:
            suffix = "baseline" if method in {"ce", "kd"} else f"seed{seed}"
            return RUNS / f"external_coughvid_test_{method}_{suffix}" / "metrics.json"
        return RUNS / f"external_coughvid_test_{method}_seed{seed}" / "metrics.json"
    if target == "toscovid2021_test":
        return RUNS / f"external_toscovid2021_test_{method}_seed{seed}" / "metrics.json"
    raise ValueError(target)


def _external_predictions_path(target: str, method: str, seed: int) -> Path:
    return _external_path(target, method, seed).with_name("predictions.csv")


def _source_gap(seed: int) -> dict[str, Any]:
    pred_dir = RUNS / f"stage1_panns_response_seed{seed}" / "predictions"
    teacher = _load_predictions(pred_dir / "teacher_test_predictions.csv")
    kd = _load_predictions(pred_dir / "student_test_predictions.csv")
    ce = _load_predictions(pred_dir / "ce_student_test_predictions.csv")
    common = sorted(set(teacher) & set(kd) & set(ce))

    def compare(student_rows: dict[str, dict[str, Any]]) -> dict[str, float]:
        kls = []
        agreements = []
        teacher_correct = []
        high_conf_gap = []
        for rid in common:
            t = teacher[rid]
            s = student_rows[rid]
            kls.append(_kl(t["probs"], s["probs"]))
            agreements.append(1.0 if t["pred_label"] == s["pred_label"] else 0.0)
            teacher_correct.append(1.0 if t["pred_label"] == t["true_label"] else 0.0)
            if max(t["probs"]) >= 0.45:
                high_conf_gap.append(_kl(t["probs"], s["probs"]))
        return {
            "kl_teacher_to_student": _mean(kls),
            "teacher_student_agreement": _mean(agreements),
            "teacher_accuracy_proxy": _mean(teacher_correct),
            "high_conf_teacher_kl": _mean(high_conf_gap),
        }

    teacher_entropies = [_entropy(teacher[rid]["probs"]) for rid in common]
    teacher_margins = [_margin(teacher[rid]["probs"]) for rid in common]
    return {
        "seed": seed,
        "n": len(common),
        "teacher_entropy": _mean(teacher_entropies),
        "teacher_margin": _mean(teacher_margins),
        "teacher_soft_hist": _hist(teacher),
        "teacher_hard_hist": _hard_hist(teacher),
        "kd": compare(kd),
        "ce": compare(ce),
    }


def _target_shift(target: str, method: str, seed: int, source_soft_hist: list[float]) -> dict[str, Any]:
    metrics_path = _external_path(target, method, seed)
    pred_path = _external_predictions_path(target, method, seed)
    metrics = _load_json(metrics_path)
    rows = _load_predictions(pred_path)
    entropies = [_entropy(row["probs"]) for row in rows.values()]
    margins = [_margin(row["probs"]) for row in rows.values()]
    soft_hist = _hist(rows)
    hard_hist = _hard_hist(rows)
    return {
        "target": target,
        "method": method,
        "seed": seed,
        "macro_auroc": float(metrics.get("macro_ovr_auroc", 0.0)),
        "covid_auroc": float(metrics.get("covid_positive", 0.0)),
        "macro_auprc": float(metrics.get("macro_ovr_auprc", 0.0)),
        "target_entropy": _mean(entropies),
        "target_margin": _mean(margins),
        "soft_l1_to_source_teacher": _l1(soft_hist, source_soft_hist),
        "hard_hist": hard_hist,
        "soft_hist": soft_hist,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows = [_source_gap(seed) for seed in SEEDS]
    source_by_seed = {row["seed"]: row for row in source_rows}
    target_rows = []
    for seed in SEEDS:
        source_hist = source_by_seed[seed]["teacher_soft_hist"]
        for target in ("coughvid", "toscovid2021_test"):
            for method in ("ce", "kd"):
                try:
                    target_rows.append(_target_shift(target, method, seed, source_hist))
                except FileNotFoundError:
                    continue

    source_table = []
    for row in source_rows:
        source_table.append(
            [
                row["seed"],
                row["n"],
                row["teacher_entropy"],
                row["teacher_margin"],
                row["kd"]["kl_teacher_to_student"],
                row["kd"]["teacher_student_agreement"],
                row["ce"]["kl_teacher_to_student"],
                row["ce"]["teacher_student_agreement"],
            ]
        )

    target_table = []
    for row in target_rows:
        target_table.append(
            [
                row["target"],
                row["method"],
                row["seed"],
                row["macro_auroc"],
                row["covid_auroc"],
                row["target_entropy"],
                row["target_margin"],
                row["soft_l1_to_source_teacher"],
            ]
        )

    payload = {"source_gap": source_rows, "target_shift": target_rows}
    (OUT / "transfer_gap_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Transfer-Gap Diagnostics",
        "",
        "This report uses existing predictions only. It does not train a new method and does not use target labels for any selection rule.",
        "",
        "## Source Teacher-Student Gap",
        "",
        _table(
            [
                "Seed",
                "n",
                "Teacher entropy",
                "Teacher margin",
                "KD KL",
                "KD agree",
                "CE KL",
                "CE agree",
            ],
            source_table,
        ),
        "",
        "## Target Prediction Shift",
        "",
        _table(
            [
                "Target",
                "Method",
                "Seed",
                "Macro AUROC",
                "COVID AUROC",
                "Target entropy",
                "Target margin",
                "L1 to source teacher",
            ],
            target_table,
        ),
        "",
        "## Readout",
        "",
        "- Existing source-test predictions are enough for a first transfer-gap audit, but not enough to train a source-sample weighting rule.",
        "- Source-train teacher/student logits are still missing; Candidate E should cache them before implementing a new KD loss.",
        "- A plain teacher-entropy or confidence weighting rule would collide with existing adaptive/uncertainty KD literature.",
        "- A claimable candidate must connect transfer-gap diagnostics to COUGHVID/Tos external gains without target-label tuning.",
        "",
    ]
    (OUT / "TRANSFER_GAP_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "TRANSFER_GAP_DIAGNOSTICS.md")


if __name__ == "__main__":
    main()
