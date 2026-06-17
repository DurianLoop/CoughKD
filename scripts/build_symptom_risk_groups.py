from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESP_TOKENS = {"asthma", "bd", "cld", "cold", "cough", "others_resp", "pneumonia", "st"}
SYSTEMIC_TOKENS = {"diarrhoea", "fever", "ftg", "ht", "loss_of_smell", "mp"}
PREEXISTING_TOKENS = {"diabetes", "ihd", "others_preexist", "smoker"}


def _tokens(raw: object) -> set[str]:
    text = "" if pd.isna(raw) else str(raw).strip().lower()
    if not text:
        return set()
    return {part.strip() for part in re.split(r"[;,|]", text) if part.strip()}


def _group(tokens: set[str]) -> str:
    if not tokens:
        return "none"
    has_resp = bool(tokens & RESP_TOKENS)
    has_systemic = bool(tokens & SYSTEMIC_TOKENS)
    has_preexisting = bool(tokens & PREEXISTING_TOKENS)
    if has_resp and has_systemic:
        return "resp_systemic"
    if has_resp and has_preexisting:
        return "resp_preexisting"
    if has_resp:
        return "resp_only"
    if has_systemic:
        return "systemic_only"
    if has_preexisting:
        return "preexisting_only"
    return "other_symptom"


def _fallback_group(group: str) -> str:
    if group == "resp_preexisting":
        return "resp_only"
    if group == "other_symptom":
        return "preexisting_only"
    return group


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-fraction", type=float, default=0.03)
    parser.add_argument("--merge-small", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/symptom_risk_groups")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.split:
        work = df[df["split"] == args.split].copy()
    else:
        work = df.copy()
    work["symptom_group"] = [_group(_tokens(value)) for value in work["symptoms"].fillna("")]
    counts = Counter(work["symptom_group"])
    total = len(work)
    kept = {group for group, count in counts.items() if count / max(1, total) >= args.min_fraction}
    if args.merge_small:
        work["symptom_group_pruned"] = [
            group if group in kept else _fallback_group(group) for group in work["symptom_group"]
        ]
    else:
        work["symptom_group_pruned"] = [group if group in kept else "other_small" for group in work["symptom_group"]]
    pruned = sorted(work["symptom_group_pruned"].unique())
    group_to_id = {group: idx for idx, group in enumerate(pruned)}
    work["artifact_env"] = [group_to_id[group] for group in work["symptom_group_pruned"]]

    args.out.mkdir(parents=True, exist_ok=True)
    assignment_path = args.out / "source_symptom_risk_groups.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recording_id", "artifact_env", "symptom_group"])
        writer.writeheader()
        for row in work.itertuples(index=False):
            writer.writerow(
                {
                    "recording_id": row.recording_id,
                    "artifact_env": int(row.artifact_env),
                    "symptom_group": row.symptom_group_pruned,
                }
            )

    rows = []
    label_groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in work.itertuples(index=False):
        label_groups[row.symptom_group_pruned][str(row.label)] += 1
    labels = sorted(work["label"].dropna().astype(str).unique())
    for group in pruned:
        n = int(sum(label_groups[group].values()))
        item = {
            "group": group,
            "env_id": int(group_to_id[group]),
            "n": n,
            "fraction": float(n / max(1, total)),
        }
        for label in labels:
            item[f"p_{label}"] = float(label_groups[group][label] / max(1, n))
        rows.append(item)
    table = pd.DataFrame(rows).sort_values("env_id")
    table.to_csv(args.out / "source_symptom_risk_group_table.csv", index=False)
    min_fraction = float(table["fraction"].min()) if not table.empty else 0.0
    max_label_prop = float(
        max(
            table[[col for col in table.columns if col.startswith("p_")]].max(axis=1).max(),
            0.0,
        )
    ) if not table.empty else 0.0
    summary = {
        "manifest": str(args.manifest),
        "split": args.split,
        "n_rows": int(total),
        "groups": table.to_dict(orient="records"),
        "group_to_id": group_to_id,
        "merge_small": bool(args.merge_small),
        "min_group_fraction": min_fraction,
        "max_group_label_prop": max_label_prop,
        "hard_gate_pass": bool(min_fraction >= args.min_fraction and max_label_prop < 0.90 and len(table) >= 3),
        "assignment_path": str(assignment_path),
    }
    (args.out / "source_symptom_risk_group_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Source Symptom Risk Groups",
        "",
        f"Hard gate pass: `{summary['hard_gate_pass']}`",
        "",
        "## Group Table",
        "",
        _to_md(table),
        "",
        "## Mapping",
        "",
        "```json",
        json.dumps(group_to_id, indent=2),
        "```",
        "",
    ]
    report = args.out / "SOURCE_SYMPTOM_RISK_GROUPS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
