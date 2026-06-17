from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_tos_env1_separability import _assign_envs, _read_csv


METRIC_KEY = "macro_ovr_auroc"


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(path: Path, key: str = METRIC_KEY) -> float:
    data = _load_metrics(path)
    if key not in data:
        raise KeyError(f"{key} missing from {path}")
    return float(data[key])


def _env_distribution(source_rows: list[dict[str, str]], target_rows: list[dict[str, str]], seed: int) -> dict[str, Any]:
    source_envs, target_envs = _assign_envs(source_rows, target_rows, seed)
    source_counts = Counter(int(value) for value in source_envs.values())
    target_counts = Counter(int(value) for value in target_envs.values())
    minority_env = min(source_counts, key=lambda env: source_counts[env])
    source_total = sum(source_counts.values())
    target_total = sum(target_counts.values())
    source_minority_prop = source_counts[minority_env] / max(1, source_total)
    target_minority_prop = target_counts[minority_env] / max(1, target_total)
    return {
        "minority_env": int(minority_env),
        "source_counts": {str(key): int(value) for key, value in sorted(source_counts.items())},
        "target_counts": {str(key): int(value) for key, value in sorted(target_counts.items())},
        "source_minority_prop": source_minority_prop,
        "target_minority_prop": target_minority_prop,
        "minority_prop_delta": target_minority_prop - source_minority_prop,
    }


def _target_spec() -> dict[str, dict[str, Any]]:
    return {
        "coughvid": {
            "display": "COUGHVID",
            "features": ROOT / "runs/target_failure_slice_audit_seed7/coughvid_test_artifact_features.csv",
            "metrics": {
                "source_only": ROOT / "runs/external_coughvid_test_stage3b_source_only_seed7/metrics.json",
                "candidate_f": ROOT / "runs/external_coughvid_test_candidate_f_artifact_env_irm_ramp_seed7/metrics.json",
            },
        },
        "tos": {
            "display": "Tos COVID-19",
            "features": ROOT / "runs/artifact_environment_audit_seed7/tos_artifact_features.csv",
            "metrics": {
                "source_only": ROOT / "runs/external_toscovid2021_test_source_only_seed7/metrics.json",
                "candidate_f": ROOT / "runs/external_toscovid2021_test_candidate_f_artifact_env_irm_ramp_seed7/metrics.json",
            },
        },
    }


def _summarize_policy(rows: list[dict[str, Any]], policy_name: str, metric_key: str) -> dict[str, Any]:
    values = [float(row[f"{policy_name}_{metric_key}"]) for row in rows]
    return {
        "mean": sum(values) / max(1, len(values)),
        "worst": min(values) if values else 0.0,
        "values": {row["target"]: float(row[f"{policy_name}_{metric_key}"]) for row in rows},
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Guarded Transfer Replay",
        "",
        "This is a retrospective no-label gate sanity check, not a final paper claim.",
        "",
        "## Rule",
        "",
        f"- Source-fitted artifact KMeans: `k={summary['config']['k']}`",
        f"- Reject artifact-IRM when target source-minority-env proportion exceeds source by more than `{summary['config']['minority_prop_tolerance']}`.",
        "- If rejected, deploy source-only; otherwise deploy Candidate F artifact-IRM.",
        "",
        "## Target Replay",
        "",
        "| Target | Source minority prop | Target minority prop | Delta | Decision | Source-only AUROC | Candidate F AUROC | Guarded AUROC |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["targets"]:
        lines.append(
            f"| {row['target']} | {row['source_minority_prop']:.6f} | {row['target_minority_prop']:.6f} | "
            f"{row['minority_prop_delta']:.6f} | {row['guard_decision']} | {row['source_only_macro_ovr_auroc']:.6f} | "
            f"{row['candidate_f_macro_ovr_auroc']:.6f} | {row['guarded_macro_ovr_auroc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(summary["aggregate"], indent=2),
            "```",
            "",
            "## Decision Hint",
            "",
            summary["decision_hint"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7/source_artifact_features.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/guarded_transfer_replay_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--minority-prop-tolerance", type=float, default=0.10)
    parser.add_argument("--metric-key", default=METRIC_KEY)
    args = parser.parse_args()

    if args.k != 2:
        raise ValueError("This replay uses the k=2 source artifact environment already used by Candidate F.")

    source_rows = _read_csv(args.source_features)
    target_rows: list[dict[str, Any]] = []
    for target_name, spec in _target_spec().items():
        features = _read_csv(spec["features"])
        env = _env_distribution(source_rows, features, args.seed)
        reject = float(env["minority_prop_delta"]) > float(args.minority_prop_tolerance)
        source_score = _metric(spec["metrics"]["source_only"], args.metric_key)
        candidate_score = _metric(spec["metrics"]["candidate_f"], args.metric_key)
        guarded_score = source_score if reject else candidate_score
        target_rows.append(
            {
                "target": spec["display"],
                **env,
                "guard_decision": "reject_candidate_f_use_source_only" if reject else "allow_candidate_f",
                f"source_only_{args.metric_key}": source_score,
                f"candidate_f_{args.metric_key}": candidate_score,
                f"guarded_{args.metric_key}": guarded_score,
                "guarded_minus_source_only": guarded_score - source_score,
                "guarded_minus_candidate_f": guarded_score - candidate_score,
            }
        )

    aggregate = {
        "source_only": _summarize_policy(target_rows, "source_only", args.metric_key),
        "candidate_f": _summarize_policy(target_rows, "candidate_f", args.metric_key),
        "guarded": _summarize_policy(target_rows, "guarded", args.metric_key),
    }
    guarded_mean_gain_vs_source = aggregate["guarded"]["mean"] - aggregate["source_only"]["mean"]
    guarded_worst_gain_vs_candidate = aggregate["guarded"]["worst"] - aggregate["candidate_f"]["worst"]
    if guarded_mean_gain_vs_source >= 0.03 and guarded_worst_gain_vs_candidate >= 0.03:
        decision_hint = "Guarded replay clears a rough 3-point gate; design a real no-label guarded-KD method next."
    else:
        decision_hint = (
            "Guarded replay avoids Candidate F negative transfer on Tos and preserves its COUGHVID gain, "
            "but the aggregate gain is too small for a standalone 3-5 point ICASSP method claim."
        )
    summary = {
        "config": {
            "source_features": str(args.source_features),
            "seed": args.seed,
            "k": args.k,
            "minority_prop_tolerance": args.minority_prop_tolerance,
            "metric_key": args.metric_key,
        },
        "targets": target_rows,
        "aggregate": {
            **aggregate,
            "guarded_mean_gain_vs_source_only": guarded_mean_gain_vs_source,
            "guarded_worst_gain_vs_candidate_f": guarded_worst_gain_vs_candidate,
        },
        "decision_hint": decision_hint,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "guarded_transfer_replay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "GUARDED_TRANSFER_REPLAY.md", summary)
    print(str(args.out / "GUARDED_TRANSFER_REPLAY.md"), flush=True)


if __name__ == "__main__":
    main()
