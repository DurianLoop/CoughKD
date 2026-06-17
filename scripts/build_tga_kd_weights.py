from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = _clip(q) * (len(values) - 1)
    left = int(pos)
    right = min(left + 1, len(values) - 1)
    frac = pos - left
    return values[left] * (1.0 - frac) + values[right] * frac


def _read_gap(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_active_weights(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    weights: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            weights[row["recording_id"]] = float(row["kd_weight"])
    return weights


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-csv", type=Path, default=ROOT / "runs/source_transfer_gap_seed7/source_transfer_gap.csv")
    parser.add_argument("--active-weights", type=Path, default=ROOT / "runs/active_cough_kd_weights_power4/active_cough_kd_weights.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tga_kd_weights_seed7/tga_kd_weights.csv")
    parser.add_argument(
        "--primary-variant",
        default="tga_coverage_weight",
        choices=["tga_reliability_weight", "tga_coverage_weight", "tga_conservative_weight"],
    )
    parser.add_argument("--floor", type=float, default=0.15)
    parser.add_argument("--ceil", type=float, default=1.0)
    args = parser.parse_args()

    rows = _read_gap(args.gap_csv)
    active = _read_active_weights(args.active_weights)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["true_label"]].append(row)

    class_reliability = {
        label: sum(float(row["teacher_correct"]) for row in label_rows) / len(label_rows)
        for label, label_rows in by_label.items()
    }
    max_rel = max(class_reliability.values()) if class_reliability else 1.0
    min_rel = min(class_reliability.values()) if class_reliability else 0.0

    out_rows = []
    for row in rows:
        entropy = float(row["teacher_entropy"])
        margin = float(row["teacher_margin"])
        kd_kl = float(row["kd_kl_from_teacher"])
        teacher_correct = float(row["teacher_correct"])
        label = row["true_label"]
        rel = class_reliability[label]
        rel_norm = (rel - min_rel) / max(1e-12, max_rel - min_rel)
        margin_score = _clip((margin - 0.02) / 0.28)
        entropy_score = _clip((1.58 - entropy) / 0.30)
        kl_score = 1.0 - _clip(kd_kl / 0.25)

        reliability = 0.50 * teacher_correct + 0.25 * margin_score + 0.25 * entropy_score
        coverage = 0.45 * teacher_correct + 0.25 * margin_score + 0.15 * entropy_score + 0.15 * rel_norm
        conservative = teacher_correct * (0.55 * margin_score + 0.25 * entropy_score + 0.20 * kl_score)

        def scale(value: float) -> float:
            return args.floor + (args.ceil - args.floor) * _clip(value)

        out_row = {
            "recording_id": row["recording_id"],
            "true_label": label,
            "teacher_correct": row["teacher_correct"],
            "teacher_entropy": row["teacher_entropy"],
            "teacher_margin": row["teacher_margin"],
            "kd_kl_from_teacher": row["kd_kl_from_teacher"],
            "class_reliability": f"{rel:.8f}",
            "tga_reliability_weight": f"{scale(reliability):.8f}",
            "tga_coverage_weight": f"{scale(coverage):.8f}",
            "tga_conservative_weight": f"{scale(conservative):.8f}",
            "active_weight": f"{active.get(row['recording_id'], 1.0):.8f}",
        }
        out_row["kd_weight"] = out_row[args.primary_variant]
        out_rows.append(out_row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    variants = ["tga_reliability_weight", "tga_coverage_weight", "tga_conservative_weight", "active_weight"]
    summary: dict[str, Any] = {
        "num_records": len(out_rows),
        "primary_variant": args.primary_variant,
        "class_reliability": class_reliability,
        "variants": {},
    }
    active_values = [float(row["active_weight"]) for row in out_rows]
    for variant in variants:
        values = [float(row[variant]) for row in out_rows]
        by_label_summary = {}
        for label in sorted(by_label):
            label_values = [float(row[variant]) for row in out_rows if row["true_label"] == label]
            by_label_summary[label] = _summary(label_values)
        summary["variants"][variant] = {
            **_summary(values),
            "pearson_with_active": _pearson(values, active_values) if variant != "active_weight" else 1.0,
            "by_label": by_label_summary,
        }

    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# TGA-KD Weight Audit", "", f"- Records: `{len(out_rows)}`", f"- Primary variant: `{args.primary_variant}`", ""]
    lines.append("## Class Reliability")
    lines.append("")
    lines.append("| Label | Teacher correctness proxy |")
    lines.append("|---|---:|")
    for label, value in sorted(class_reliability.items()):
        lines.append(f"| {label} | {value:.6f} |")
    lines.append("")
    lines.append("## Variant Summary")
    lines.append("")
    lines.append("| Variant | Mean | p10 | p50 | p90 | Min | Max | Pearson vs active |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        item = summary["variants"][variant]
        lines.append(
            f"| {variant} | {item['mean']:.6f} | {item['p10']:.6f} | {item['p50']:.6f} | "
            f"{item['p90']:.6f} | {item['min']:.6f} | {item['max']:.6f} | {item['pearson_with_active']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- These are diagnostic candidate weights, not a validated method.",
            "- If a variant is too close to Candidate D active weights or collapses most samples to the floor, do not train it.",
            "- A training gate should use only one variant, selected before seeing external labels.",
            "",
        ]
    )
    report_path = args.out.with_suffix(".md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
