from __future__ import annotations

import argparse
import csv
import importlib.util
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


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _module_status() -> dict[str, bool]:
    modules = ["torch", "torchaudio", "librosa", "numpy", "timm", "omegaconf", "hydra", "soundfile"]
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def _repo_status(repo: Path) -> dict[str, Any]:
    return {
        "repo_path": str(repo),
        "repo_exists": repo.is_dir(),
        "extractor_exists": (repo / "src/benchmark/model_util.py").is_file(),
        "readme_exists": (repo / "README.md").is_file(),
    }


def _load_extractor(repo: Path) -> Any:
    module_path = repo / "src/benchmark/model_util.py"
    spec = importlib.util.spec_from_file_location("opera_model_util", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OPERA extractor from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    try:
        spec.loader.exec_module(module)
    finally:
        for value in (str(repo / "src"), str(repo)):
            try:
                sys.path.remove(value)
            except ValueError:
                pass
    if not hasattr(module, "extract_opera_feature"):
        raise RuntimeError("OPERA model_util.py does not expose extract_opera_feature")
    return module.extract_opera_feature


def _extract_opera_embeddings(
    *,
    manifest: Path,
    root: Path,
    repo: Path,
    out_path: Path,
    pretrain: str,
    input_sec: int,
    dim: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))
    import numpy as np

    rows = _manifest_rows(manifest)
    extractor = _load_extractor(repo)
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        paths = [str(root / row["path"]) for row in chunk]
        features = extractor(paths, pretrain=pretrain, input_sec=input_sec, dim=dim)
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row, embedding in zip(chunk, arr.tolist()):
            records.append(
                {
                    "recording_id": row["recording_id"],
                    "true_label": row["label"],
                    "model": f"opera_{pretrain}_frozen",
                    "embedding": [float(value) for value in embedding],
                }
            )
        batch_idx = start // batch_size + 1
        total = (len(rows) + batch_size - 1) // batch_size
        if batch_idx == 1 or batch_idx % 10 == 0:
            print(f"[extract] OPERA {pretrain} batch {batch_idx}/{total}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records), encoding="utf-8")
    return records


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
        "# OPERA Embedding Upper-Bound Audit",
        "",
        "## Preflight",
        "",
        "```json",
        json.dumps(summary["preflight"], indent=2),
        "```",
        "",
    ]
    if not summary.get("ran_gate"):
        lines.extend(["## Decision Hint", "", summary["decision_hint"], ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
            "## Gate",
            "",
            "```json",
            json.dumps(summary["gate"], indent=2),
            "```",
            "",
            "## OPERA Frozen Embedding",
            "",
        ]
    )
    for env in ("0", "1", "all"):
        item = summary["opera_frozen"].get(env)
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
    parser.add_argument("--repo", type=Path, default=ROOT / "external/teacher_repos/OPERA")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/opera_embedding_upper_bound_seed7")
    parser.add_argument("--pretrain", default="operaCT", choices=["operaCT", "operaCE", "operaGT"])
    parser.add_argument("--input-sec", type=int, default=8)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-gate", action="store_true", help="May trigger OPERA checkpoint downloads through the upstream extractor.")
    args = parser.parse_args()

    modules = _module_status()
    repo = _repo_status(args.repo)
    ready = all(modules.values()) and bool(repo["extractor_exists"])
    preflight = {
        "modules": modules,
        "repo": repo,
        "environment_ready": all(modules.values()),
        "asset_ready": bool(repo["extractor_exists"]),
        "run_gate": args.run_gate,
        "warning": "OPERA extractor may auto-download checkpoints; only use --run-gate after approval.",
        "source": "https://github.com/evelyn0414/OPERA",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.run_gate or not ready:
        reason = "run-gate not requested" if not args.run_gate else "OPERA repo/dependencies are not ready"
        summary = {
            "preflight": preflight,
            "ran_gate": False,
            "decision_hint": f"{reason}; clone OPERA, install missing dependencies, and rerun with --run-gate only after approval.",
        }
        (args.out / "opera_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_report(args.out / "OPERA_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
        print(str(args.out / "OPERA_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)
        return

    source_features = _read_csv(args.source_features)
    tos_features = _read_csv(args.tos_features)
    _, tos_envs = _assign_envs(source_features, tos_features, args.seed)
    manifest_ids = {row["recording_id"] for row in _manifest_rows(args.manifest)}
    tos_envs = {rid: env for rid, env in tos_envs.items() if rid in manifest_ids}
    rows = _extract_opera_embeddings(
        manifest=args.manifest,
        root=args.root,
        repo=args.repo,
        out_path=args.out / "embeddings" / f"{args.pretrain}_frozen.json",
        pretrain=args.pretrain,
        input_sec=args.input_sec,
        dim=args.dim,
        batch_size=args.batch_size,
    )
    opera_summary = _summarize(rows, tos_envs=tos_envs, seed=args.seed, folds=args.folds)
    env1_auc = float(opera_summary.get("1", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    all_auc = float(opera_summary.get("all", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    gate = {
        "required_env1_auroc": 0.60,
        "required_all_tos_auroc_gt": 0.564254,
        "observed_env1_auroc": env1_auc,
        "observed_all_tos_auroc": all_auc,
    }
    decision_hint = (
        "OPERA frozen embeddings clear the Tos gate; design a guarded foundation-KD candidate next."
        if env1_auc >= 0.60 and all_auc > 0.564254
        else "OPERA frozen embeddings do not clear the Tos gate; do not launch an OPERA-based positive KD experiment."
    )
    summary = {
        "preflight": preflight,
        "ran_gate": True,
        "env_distributions": dict(Counter(str(value) for value in tos_envs.values())),
        "opera_frozen": opera_summary,
        "gate": gate,
        "decision_hint": decision_hint,
    }
    (args.out / "opera_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "OPERA_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
    print(str(args.out / "OPERA_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
