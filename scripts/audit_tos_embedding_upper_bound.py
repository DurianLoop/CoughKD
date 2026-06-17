from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_tos_env1_separability import _assign_envs, _read_csv
from coughkd.config import RunConfig
from coughkd.metrics import average_precision, roc_auc
from coughkd.torch_models import CoughManifestDataset, DepthwiseStudent, pad_collate


def _require_torch() -> Any:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader, np


def _load_model(checkpoint: Path, device: Any) -> tuple[Any, dict[str, int]]:
    torch, _, _ = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(key): int(value) for key, value in payload["label_to_index"].items()}
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, label_to_index


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_embeddings(
    *,
    model_name: str,
    checkpoint: Path,
    manifest: Path,
    root: Path,
    out_path: Path,
    batch_size: int,
    device_name: str,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))
    torch, DataLoader, _ = _require_torch()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, label_to_index = _load_model(checkpoint, device)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    dataset = CoughManifestDataset(
        manifest=manifest,
        root=root,
        config=RunConfig(experiment_name="tos_embedding_upper_bound"),
        label_to_index=label_to_index,
        splits={"train", "val", "test", "external", "adapt"},
        max_duration_sec=max_duration_sec,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            out = model(features)
            embeddings = out["embedding"].detach().cpu().tolist()
            probs = torch.softmax(out["logits"].detach().cpu(), dim=1).tolist()
            for rid, label_idx, embedding, prob in zip(batch["recording_id"], batch["label"].tolist(), embeddings, probs):
                row: dict[str, Any] = {
                    "recording_id": rid,
                    "true_label": index_to_label[int(label_idx)],
                    "model": model_name,
                    "embedding": [float(value) for value in embedding],
                }
                row.update({f"prob_{index_to_label[idx]}": float(value) for idx, value in enumerate(prob)})
                rows.append(row)
            if batch_idx == 1 or batch_idx % 20 == 0:
                print(f"[extract] {model_name} batch {batch_idx}/{len(loader)}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _binary_label(label: str) -> int:
    return 1 if label == "covid_positive" else 0


def _probe(rows: list[dict[str, Any]], seed: int, folds: int) -> dict[str, Any]:
    _, _, np = _require_torch()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = [_binary_label(str(row["true_label"])) for row in rows]
    counts = Counter(labels)
    if len(counts) != 2 or min(counts.values()) < 2:
        return {"enabled": False, "reason": f"insufficient binary labels: {dict(counts)}", "n": len(rows)}
    x = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    n_splits = min(folds, min(counts.values()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(rows), dtype=np.float32)
    for train_idx, test_idx in cv.split(x, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
        clf.fit(x[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    return {
        "enabled": True,
        "folds": int(n_splits),
        "n": int(len(rows)),
        "positive": int(counts[1]),
        "negative": int(counts[0]),
        "auroc": roc_auc(y.tolist(), scores.tolist()),
        "auprc": average_precision(y.tolist(), scores.tolist()),
        "mean_positive_score": float(scores[y == 1].mean()),
        "mean_negative_score": float(scores[y == 0].mean()),
    }


def _score_auc(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    labels = [_binary_label(str(row["true_label"])) for row in rows]
    scores = [float(row[key]) for row in rows]
    return {
        "n": len(rows),
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "mean_positive_score": sum(score for label, score in zip(labels, scores) if label) / max(1, sum(labels)),
        "mean_negative_score": sum(score for label, score in zip(labels, scores) if not label) / max(1, len(labels) - sum(labels)),
    }


def _summarize_model(rows: list[dict[str, Any]], tos_envs: dict[str, int], seed: int, folds: int) -> dict[str, Any]:
    for row in rows:
        row["artifact_env"] = int(tos_envs[row["recording_id"]])
        row["covid_vs_healthy_margin"] = float(row.get("prob_covid_positive", 0.0)) - float(row.get("prob_healthy", 0.0))
    summary: dict[str, Any] = {}
    for env in sorted({int(row["artifact_env"]) for row in rows}):
        env_rows = [row for row in rows if int(row["artifact_env"]) == env]
        summary[str(env)] = {
            "n": len(env_rows),
            "label_counts": dict(Counter(str(row["true_label"]) for row in env_rows)),
            "embedding_cv_probe": _probe(env_rows, seed=seed, folds=folds),
            "native_score_auc": {
                "prob_covid_positive": _score_auc(env_rows, "prob_covid_positive"),
                "covid_vs_healthy_margin": _score_auc(env_rows, "covid_vs_healthy_margin"),
            },
        }
    summary["all"] = {
        "n": len(rows),
        "label_counts": dict(Counter(str(row["true_label"]) for row in rows)),
        "embedding_cv_probe": _probe(rows, seed=seed, folds=folds),
        "native_score_auc": {
            "prob_covid_positive": _score_auc(rows, "prob_covid_positive"),
            "covid_vs_healthy_margin": _score_auc(rows, "covid_vs_healthy_margin"),
        },
    }
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Tos Embedding Upper-Bound Audit",
        "",
        "## Env Distributions",
        "",
        "```json",
        json.dumps(summary["env_distributions"], indent=2),
        "```",
        "",
    ]
    for model, item in summary["models"].items():
        lines.extend([f"## {model}", ""])
        for env in ("0", "1", "all"):
            env_item = item.get(env)
            if not env_item:
                continue
            lines.extend(
                [
                    f"### Env {env}",
                    "",
                    f"- n: `{env_item['n']}`",
                    f"- label counts: `{env_item['label_counts']}`",
                    f"- embedding CV probe: `{env_item['embedding_cv_probe']}`",
                    f"- native score AUCs: `{env_item['native_score_auc']}`",
                    "",
                ]
            )
    lines.extend(["## Decision Hint", "", summary["decision_hint"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/toscovid2021_test_external.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--source-features", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7/source_artifact_features.csv")
    parser.add_argument("--tos-features", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7/tos_artifact_features.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tos_embedding_upper_bound_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--checkpoint", action="append", default=[])
    args = parser.parse_args()

    checkpoints = {
        "source_ce_student": ROOT / "runs/stage1_panns_response_seed7/checkpoints/ce_student_best.pt",
        "source_kd_student": ROOT / "runs/stage1_panns_response_seed7/checkpoints/student_best.pt",
        "candidate_f_artifact_irm": ROOT / "runs/candidate_f_artifact_env_irm_ramp_seed7/checkpoints/student_best.pt",
    }
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError(f"--checkpoint must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)

    source_features = _read_csv(args.source_features)
    tos_features = _read_csv(args.tos_features)
    _, tos_envs = _assign_envs(source_features, tos_features, args.seed)
    manifest_ids = {row["recording_id"] for row in _manifest_rows(args.manifest)}
    tos_envs = {rid: env for rid, env in tos_envs.items() if rid in manifest_ids}

    args.out.mkdir(parents=True, exist_ok=True)
    model_summaries: dict[str, Any] = {}
    for model_name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            model_summaries[model_name] = {"error": f"checkpoint not found: {checkpoint}"}
            continue
        rows = _extract_embeddings(
            model_name=model_name,
            checkpoint=checkpoint,
            manifest=args.manifest,
            root=args.root,
            out_path=args.out / "embeddings" / f"{model_name}.json",
            batch_size=args.batch_size,
            device_name=args.device,
            max_duration_sec=args.max_duration_sec,
        )
        rows = [row for row in rows if row["recording_id"] in tos_envs]
        model_summaries[model_name] = _summarize_model(rows, tos_envs, seed=args.seed, folds=args.folds)

    env1_probes = [
        item.get("1", {}).get("embedding_cv_probe", {}).get("auroc", 0.0)
        for item in model_summaries.values()
        if isinstance(item, dict)
    ]
    best_env1 = max([float(value) for value in env1_probes] or [0.0])
    if best_env1 >= 0.60:
        decision_hint = "Tos env1 is separable in at least one source-trained embedding; a target-aware clustering or adaptation candidate may be worth a small gate."
    else:
        decision_hint = "Tos env1 is not reliably separable even in source-trained embeddings; target-unlabeled adaptation is unlikely to yield a defensible Tos claim without new data or labels."
    summary = {
        "env_distributions": dict(Counter(str(value) for value in tos_envs.values())),
        "models": model_summaries,
        "decision_hint": decision_hint,
    }
    (args.out / "tos_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "TOS_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
    print(str(args.out / "TOS_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
