from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.config import RunConfig
from coughkd.torch_models import CoughManifestDataset, DepthwiseStudent, pad_collate


def _require_torch() -> Any:
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader


def _read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def _sample_manifest(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    parts = []
    for _, group in df.groupby("label", sort=False):
        take = max(1, round(n * len(group) / len(df)))
        parts.append(group.sample(n=min(take, len(group)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _write_manifest(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _load_model(checkpoint: Path, device: Any) -> tuple[Any, dict[str, int]]:
    torch, _ = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(k): int(v) for k, v in payload["label_to_index"].items()}
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, label_to_index


def _extract(model_name: str, checkpoint: Path, manifest: Path, root: Path, out_path: Path, batch_size: int, device_name: str) -> pd.DataFrame:
    if out_path.is_file():
        print(f"[cache] {out_path}", flush=True)
        return pd.read_pickle(out_path)
    torch, DataLoader = _require_torch()
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, label_to_index = _load_model(checkpoint, device)
    dataset = CoughManifestDataset(
        manifest=manifest,
        root=root,
        config=RunConfig(experiment_name="domain_probe"),
        label_to_index=label_to_index,
        splits={"train", "val", "test", "external", "adapt"},
        max_duration_sec=4.0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            out = model(features)
            emb = out["embedding"].detach().cpu().numpy()
            probs = torch.softmax(out["logits"].detach().cpu(), dim=1).numpy()
            for rid, z, p in zip(batch["recording_id"], emb, probs):
                row: dict[str, Any] = {"recording_id": rid, "model": model_name}
                row.update({f"emb_{idx:04d}": float(value) for idx, value in enumerate(z)})
                row.update({f"prob_{idx}": float(value) for idx, value in enumerate(p)})
                records.append(row)
            if batch_idx == 1 or batch_idx % 20 == 0:
                print(f"[extract] {model_name} batch {batch_idx}/{len(loader)}", flush=True)
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    return df


def _probe(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y.astype(str))
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_enc,
        test_size=0.3,
        random_state=seed,
        stratify=y_enc,
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    if len(enc.classes_) == 2:
        auc = roc_auc_score(y_test, prob[:, 1])
    else:
        auc = roc_auc_score(y_test, prob, average="macro", multi_class="ovr")
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "classes": [str(item) for item in enc.classes_],
        "n": int(len(y_enc)),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        vals = []
        for value in row:
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coswara-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--coughvid-manifest", type=Path, default=ROOT / "manifests/coughvid_external.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage1_panns_response_seed7")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/domain_probe_coswara_coughvid_students")
    parser.add_argument("--samples-per-dataset", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extra-checkpoint", action="append", default=[], help="Optional NAME=PATH checkpoint to include.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    coswara = _sample_manifest(_read_manifest(args.coswara_manifest), args.samples_per_dataset, args.seed)
    coughvid = _sample_manifest(_read_manifest(args.coughvid_manifest), args.samples_per_dataset, args.seed)
    combined = pd.concat([coswara, coughvid], ignore_index=True).sample(frac=1.0, random_state=args.seed)
    combined_path = args.out / "combined_probe_manifest.csv"
    _write_manifest(combined_path, combined)

    checkpoints = {
        "ce_student": args.run_dir / "checkpoints/ce_student_best.pt",
        "kd_student": args.run_dir / "checkpoints/student_best.pt",
    }
    for item in args.extra_checkpoint:
        if "=" not in item:
            raise ValueError(f"--extra-checkpoint must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        checkpoints[name] = Path(path)
    embeddings = {}
    for model_name, checkpoint in checkpoints.items():
        embeddings[model_name] = _extract(
            model_name,
            checkpoint,
            combined_path,
            args.root,
            args.out / "embeddings" / f"{model_name}.pkl",
            args.batch_size,
            args.device,
        )

    emb_cols = [col for col in next(iter(embeddings.values())).columns if col.startswith("emb_")]
    rows = []
    for model_name, emb_df in embeddings.items():
        merged = combined.merge(emb_df, on="recording_id", how="inner")
        x = merged[emb_cols].to_numpy(dtype=np.float32)
        targets = {
            "dataset_domain": merged["dataset"].astype(str).to_numpy(),
            "task_label": merged["label"].astype(str).to_numpy(),
        }
        for target_name, y in targets.items():
            rows.append({"model": model_name, "target": target_name, **_probe(x, y, args.seed)})
    result = pd.DataFrame(rows)
    result.to_csv(args.out / "domain_probe_results.csv", index=False)
    pivot = result.pivot_table(index="target", columns="model", values="auc", aggfunc="first")
    if {"ce_student", "kd_student"}.issubset(pivot.columns):
        pivot["kd_minus_ce"] = pivot["kd_student"] - pivot["ce_student"]
    report = [
        "# Coswara vs COUGHVID Student Domain Probe",
        "",
        f"- Samples per dataset target: `{args.samples_per_dataset}`",
        f"- Combined manifest: `{combined_path}`",
        f"- Dataset counts: `{combined['dataset'].value_counts().to_dict()}`",
        f"- Label counts: `{combined['label'].value_counts().to_dict()}`",
        "",
        "## Probe Results",
        "",
        _markdown_table(result[["model", "target", "auc", "balanced_accuracy", "accuracy", "n"]]),
        "",
        "## KD Minus CE",
        "",
        _markdown_table(pivot.reset_index()),
        "",
    ]
    (args.out / "DOMAIN_PROBE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
