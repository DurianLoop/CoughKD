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

from audit_tos_embedding_upper_bound import _probe
from audit_tos_env1_separability import _assign_envs, _read_csv
from coughkd.config import RunConfig
from coughkd.torch_models import CoughManifestDataset, PannsCnn14Teacher, pad_collate


def _require_torch() -> Any:
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label_to_index(rows: list[dict[str, str]]) -> dict[str, int]:
    labels = sorted({row["label"] for row in rows if row.get("label")})
    return {label: idx for idx, label in enumerate(labels)}


def _extract_panns_embeddings(
    *,
    manifest: Path,
    root: Path,
    checkpoint: Path,
    repo: Path,
    out_path: Path,
    batch_size: int,
    device_name: str,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))

    torch, DataLoader = _require_torch()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    manifest_rows = _manifest_rows(manifest)
    label_to_index = _label_to_index(manifest_rows)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    dataset = CoughManifestDataset(
        manifest=manifest,
        root=root,
        config=RunConfig(experiment_name="tos_foundation_embedding_upper_bound"),
        label_to_index=label_to_index,
        splits={"train", "val", "test", "external", "adapt"},
        max_duration_sec=max_duration_sec,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    model = PannsCnn14Teacher(num_classes=len(label_to_index), checkpoint=checkpoint, repo=repo).to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            out = model.forward_batch(batch, device)
            embeddings = out["embedding"].detach().cpu().tolist()
            for rid, label_idx, embedding in zip(batch["recording_id"], batch["label"].tolist(), embeddings):
                rows.append(
                    {
                        "recording_id": rid,
                        "true_label": index_to_label[int(label_idx)],
                        "model": "panns_cnn14_16k_audioset_frozen",
                        "embedding": [float(value) for value in embedding],
                    }
                )
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[extract] panns frozen batch {batch_idx}/{len(loader)}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _summarize(rows: list[dict[str, Any]], tos_envs: dict[str, int], seed: int, folds: int) -> dict[str, Any]:
    kept = [row for row in rows if row["recording_id"] in tos_envs]
    for row in kept:
        row["artifact_env"] = int(tos_envs[row["recording_id"]])
    summary: dict[str, Any] = {}
    for env in sorted({int(row["artifact_env"]) for row in kept}):
        env_rows = [row for row in kept if int(row["artifact_env"]) == env]
        summary[str(env)] = {
            "n": len(env_rows),
            "label_counts": dict(Counter(str(row["true_label"]) for row in env_rows)),
            "embedding_cv_probe": _probe(env_rows, seed=seed, folds=folds),
        }
    summary["all"] = {
        "n": len(kept),
        "label_counts": dict(Counter(str(row["true_label"]) for row in kept)),
        "embedding_cv_probe": _probe(kept, seed=seed, folds=folds),
    }
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Tos Foundation Embedding Upper-Bound Audit",
        "",
        "## Assets",
        "",
        f"- PANNs checkpoint: `{summary['assets']['panns_checkpoint']}`",
        f"- PANNs repo: `{summary['assets']['panns_repo']}`",
        "- BEATs/AST/PaSST were not run because local assets/wrappers are unavailable.",
        "",
        "## Env Distributions",
        "",
        "```json",
        json.dumps(summary["env_distributions"], indent=2),
        "```",
        "",
        "## PANNs CNN14 16 kHz AudioSet Frozen Embedding",
        "",
    ]
    for env in ("0", "1", "all"):
        item = summary["panns_cnn14_16k_audioset_frozen"].get(env)
        if not item:
            continue
        lines.extend(
            [
                f"### Env {env}",
                "",
                f"- n: `{item['n']}`",
                f"- label counts: `{item['label_counts']}`",
                f"- embedding CV probe: `{item['embedding_cv_probe']}`",
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
    parser.add_argument("--panns-checkpoint", type=Path, default=ROOT / "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--panns-repo", type=Path, default=ROOT / "external/teacher_repos/audioset_tagging_cnn_zip")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tos_foundation_embedding_upper_bound_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    args = parser.parse_args()

    source_features = _read_csv(args.source_features)
    tos_features = _read_csv(args.tos_features)
    _, tos_envs = _assign_envs(source_features, tos_features, args.seed)
    manifest_ids = {row["recording_id"] for row in _manifest_rows(args.manifest)}
    tos_envs = {rid: env for rid, env in tos_envs.items() if rid in manifest_ids}

    args.out.mkdir(parents=True, exist_ok=True)
    panns_rows = _extract_panns_embeddings(
        manifest=args.manifest,
        root=args.root,
        checkpoint=args.panns_checkpoint,
        repo=args.panns_repo,
        out_path=args.out / "embeddings" / "panns_cnn14_16k_audioset_frozen.json",
        batch_size=args.batch_size,
        device_name=args.device,
        max_duration_sec=args.max_duration_sec,
    )
    panns_summary = _summarize(panns_rows, tos_envs=tos_envs, seed=args.seed, folds=args.folds)
    env1_auc = float(panns_summary.get("1", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    all_auc = float(panns_summary.get("all", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    if env1_auc >= 0.60 and all_auc > 0.564254:
        decision_hint = "PANNs frozen embeddings clear the Tos gate; a lightweight foundation-KD candidate is worth designing."
    else:
        decision_hint = "PANNs frozen embeddings do not clear the Tos gate; do not launch a PANNs-only positive-claim KD experiment."
    summary = {
        "assets": {
            "panns_checkpoint": str(args.panns_checkpoint),
            "panns_repo": str(args.panns_repo),
        },
        "env_distributions": dict(Counter(str(value) for value in tos_envs.values())),
        "panns_cnn14_16k_audioset_frozen": panns_summary,
        "gate": {
            "required_env1_auroc": 0.60,
            "required_all_tos_auroc_gt": 0.564254,
            "observed_env1_auroc": env1_auc,
            "observed_all_tos_auroc": all_auc,
        },
        "decision_hint": decision_hint,
    }
    (args.out / "tos_foundation_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "TOS_FOUNDATION_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
    print(str(args.out / "TOS_FOUNDATION_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
