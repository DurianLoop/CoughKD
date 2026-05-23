from __future__ import annotations

import argparse
import csv
import json
import math
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
from coughkd.torch_models import (  # noqa: E402
    CoughManifestDataset,
    DepthwiseStudent,
    _forward_model,
    pad_collate,
)


def _require_torch() -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for embedding extraction") from exc
    return torch, DataLoader


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch, _ = _require_torch()
    print(f"[checkpoint] loading {path}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model(model_name: str, checkpoint: Path, num_classes: int, panns_checkpoint: Path, panns_repo: Path) -> Any:
    torch, _ = _require_torch()
    payload = _load_checkpoint(checkpoint)
    if model_name == "panns_teacher":
        print("[model] constructing PANNs architecture without official preload", flush=True)
        model = _RunCheckpointPannsTeacher(num_classes=num_classes, repo=panns_repo)
    elif model_name in {"ce_student", "kd_student"}:
        model = DepthwiseStudent(num_classes=num_classes)
    else:
        raise ValueError(f"unknown model_name: {model_name}")
    print(f"[model] loading state dict for {model_name}", flush=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    print(f"[model] ready {model_name}", flush=True)
    return model


def _import_panns_models(repo: Path) -> Any:
    import importlib

    pytorch_dir = repo.resolve() / "pytorch"
    if not pytorch_dir.is_dir():
        raise RuntimeError(f"PANNs source directory not found: {pytorch_dir}")
    sys.path.insert(0, str(pytorch_dir))
    try:
        print(f"[panns] importing models from {pytorch_dir}", flush=True)
        return importlib.import_module("models")
    finally:
        try:
            sys.path.remove(str(pytorch_dir))
        except ValueError:
            pass


class _RunCheckpointPannsTeacher:  # lightweight wrapper; avoids loading official checkpoint twice.
    def __init__(self, num_classes: int, repo: Path):
        torch, _ = _require_torch()
        from torch import nn
        from torch.nn import functional as F

        class _Wrapper(nn.Module):
            def __init__(self, classes: int, repo_path: Path):
                super().__init__()
                panns_models = _import_panns_models(repo_path)
                panns_models.init_layer = lambda layer: None
                panns_models.init_bn = lambda bn: None
                print("[panns] instantiating Cnn14_16k", flush=True)
                self.backbone = panns_models.Cnn14_16k(
                    sample_rate=16000,
                    window_size=512,
                    hop_size=160,
                    mel_bins=64,
                    fmin=50,
                    fmax=8000,
                    classes_num=527,
                )
                print("[panns] Cnn14_16k instantiated", flush=True)
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.classifier = nn.Linear(2048, classes)

            def forward_batch(self, batch: dict[str, Any], device: Any) -> dict[str, Any]:
                waveform = batch["waveform"].to(device, non_blocking=True)
                with torch.no_grad():
                    backbone_out = self.backbone(waveform)
                    embedding = backbone_out["embedding"]
                logits = self.classifier(embedding)
                return {
                    "logits": logits,
                    "features": embedding,
                    "embedding": F.normalize(embedding, dim=1),
                    "attention": torch.empty((waveform.shape[0], 1), device=device),
                }

        self.module = _Wrapper(num_classes, repo)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


def _extract_embeddings(
    model_name: str,
    model: Any,
    manifest: Path,
    root: Path,
    label_to_index: dict[str, int],
    device: str,
    batch_size: int,
    max_duration_sec: float,
    max_records: int | None,
    out_path: Path,
) -> pd.DataFrame:
    torch, DataLoader = _require_torch()
    dataset = CoughManifestDataset(
        manifest=manifest,
        root=root,
        config=RunConfig(experiment_name="shortcut_probe"),
        label_to_index=label_to_index,
        splits={"train", "val", "test"},
        max_records=max_records,
        max_duration_sec=max_duration_sec,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    model.to(device)
    model.eval()
    records: list[dict[str, Any]] = []
    print(f"[extract] {model_name}: records={len(dataset)} batch_size={batch_size} device={device}", flush=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            out = _forward_model(model, batch, torch.device(device))
            embeddings = out["embedding"].detach().cpu().numpy()
            logits = out["logits"].detach().cpu()
            probs = torch.softmax(logits, dim=1).numpy()
            for rid, emb, prob in zip(batch["recording_id"], embeddings, probs):
                row: dict[str, Any] = {"recording_id": rid, "model": model_name}
                row.update({f"emb_{idx:04d}": float(value) for idx, value in enumerate(emb)})
                row.update({f"prob_{idx}": float(value) for idx, value in enumerate(prob)})
                records.append(row)
            if batch_idx == 1 or batch_idx % 20 == 0:
                print(f"[extract] {model_name}: batch {batch_idx}/{math.ceil(len(dataset) / batch_size)}", flush=True)
    df = pd.DataFrame.from_records(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    return df


def _load_or_extract(
    model_name: str,
    checkpoint: Path,
    manifest: Path,
    root: Path,
    label_to_index: dict[str, int],
    device: str,
    batch_size: int,
    max_duration_sec: float,
    max_records: int | None,
    panns_checkpoint: Path,
    panns_repo: Path,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    out_path = cache_dir / f"{model_name}_embeddings.pkl"
    if out_path.is_file() and not refresh:
        print(f"[cache] loading {out_path}", flush=True)
        return pd.read_pickle(out_path)
    print(f"[model] building {model_name}", flush=True)
    model = _build_model(model_name, checkpoint, len(label_to_index), panns_checkpoint, panns_repo)
    return _extract_embeddings(
        model_name=model_name,
        model=model,
        manifest=manifest,
        root=root,
        label_to_index=label_to_index,
        device=device,
        batch_size=batch_size,
        max_duration_sec=max_duration_sec,
        max_records=max_records,
        out_path=out_path,
    )


def _audio_quality_proxy(path: Path) -> dict[str, float]:
    from coughkd.audio import read_wav_mono

    samples, sample_rate = read_wav_mono(path)
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return {"duration_sec": 0.0, "rms": 0.0, "silence_ratio": 1.0, "clipping_ratio": 0.0}
    abs_arr = np.abs(arr)
    return {
        "duration_sec": float(arr.size / max(1, sample_rate)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "silence_ratio": float(np.mean(abs_arr < 0.01)),
        "clipping_ratio": float(np.mean(abs_arr > 0.98)),
    }


def _build_metadata(manifest: Path, root: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    df["recording_type"] = df["path"].str.extract(r"(cough-heavy|cough-shallow)")
    df["country_group"] = np.where(df["country"].astype(str).str.lower().eq("india"), "india", "non_india")
    df["sex_binary"] = df["sex"].where(df["sex"].isin(["male", "female"]))
    ages = pd.to_numeric(df["age"], errors="coerce")
    df["age_bin"] = pd.cut(ages, bins=[0, 30, 45, 200], labels=["age_le_30", "age_31_45", "age_gt_45"])
    df["symptom_present"] = np.where(df["symptoms"].notna() & df["symptoms"].astype(str).str.len().gt(0), "symptom", "no_symptom")
    parts = df["path"].str.split("/", expand=True)
    df["collection_date"] = parts[1] if parts.shape[1] > 1 else ""
    df["collection_period"] = df["collection_date"].astype(str).str.slice(0, 6)

    quality_rows = []
    for row in df.itertuples(index=False):
        quality_rows.append(_audio_quality_proxy(root / row.path))
    qdf = pd.DataFrame(quality_rows)
    df = pd.concat([df.reset_index(drop=True), qdf], axis=1)
    for col in ["duration_sec", "rms", "silence_ratio"]:
        try:
            df[f"{col}_bin"] = pd.qcut(df[col].rank(method="first"), q=3, labels=[f"{col}_low", f"{col}_mid", f"{col}_high"])
        except ValueError:
            df[f"{col}_bin"] = None
    df["clipping_bin"] = np.where(df["clipping_ratio"] > 0, "has_clipping", "no_clipping")
    return df


def _probe_auc(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any] | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    enc = LabelEncoder()
    y_train_enc = enc.fit_transform(y_train)
    known = np.isin(y_test, enc.classes_)
    if known.sum() < 10:
        return None
    y_test_known = y_test[known]
    x_test_known = x_test[known]
    y_test_enc = enc.transform(y_test_known)
    if len(enc.classes_) < 2 or min(np.bincount(y_train_enc)) < 5 or len(set(y_test_enc)) < 2:
        return None
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(x_train, y_train_enc)
    pred = clf.predict(x_test_known)
    probs = clf.predict_proba(x_test_known)
    if len(enc.classes_) == 2:
        auc = roc_auc_score(y_test_enc, probs[:, 1])
    else:
        present = np.unique(y_test_enc)
        if len(present) < 2:
            return None
        if len(present) != len(enc.classes_):
            probs = probs[:, present]
            probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            remap = {old: new for new, old in enumerate(present)}
            y_test_enc = np.asarray([remap[item] for item in y_test_enc])
        auc = roc_auc_score(y_test_enc, probs, multi_class="ovr", average="macro")
    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_test_enc, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test_enc, pred)),
        "classes": [str(item) for item in enc.classes_],
        "n_train": int(len(y_train_enc)),
        "n_test": int(len(y_test_enc)),
    }


def _run_probes(embeddings: dict[str, pd.DataFrame], metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    emb_cols = [col for col in next(iter(embeddings.values())).columns if col.startswith("emb_")]
    targets = {
        "task_label": "label",
        "recording_type": "recording_type",
        "sex_binary": "sex_binary",
        "country_group": "country_group",
        "age_bin": "age_bin",
        "symptom_present": "symptom_present",
        "collection_period": "collection_period",
        "duration_bin": "duration_sec_bin",
        "rms_bin": "rms_bin",
        "silence_bin": "silence_ratio_bin",
        "clipping_bin": "clipping_bin",
    }
    rows = []
    skipped: dict[str, str] = {}
    for model_name, emb_df in embeddings.items():
        merged = metadata.merge(emb_df, on="recording_id", how="inner")
        train_df = merged[merged["split"].eq("train")]
        test_df = merged[merged["split"].eq("test")]
        x_train = train_df[emb_cols].to_numpy(dtype=np.float32)
        x_test = test_df[emb_cols].to_numpy(dtype=np.float32)
        for target_name, col in targets.items():
            usable_train = train_df[col].notna()
            usable_test = test_df[col].notna()
            if usable_train.sum() < 50 or usable_test.sum() < 30:
                skipped[f"{model_name}:{target_name}"] = "too few labeled examples"
                continue
            report = _probe_auc(
                x_train[usable_train.to_numpy()],
                train_df.loc[usable_train, col].astype(str).to_numpy(),
                x_test[usable_test.to_numpy()],
                test_df.loc[usable_test, col].astype(str).to_numpy(),
            )
            if report is None:
                skipped[f"{model_name}:{target_name}"] = "insufficient class support"
                continue
            rows.append({"model": model_name, "target": target_name, **report})
    return pd.DataFrame(rows), skipped


def _write_markdown(path: Path, probe_df: pd.DataFrame, skipped: dict[str, Any], metadata: pd.DataFrame) -> None:
    def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
        if df.empty:
            return ""
        cols = list(df.columns)
        lines_local = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---" for _ in cols]) + " |"]
        for row in df.itertuples(index=False):
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(format(value, floatfmt))
                else:
                    values.append(str(value))
            lines_local.append("| " + " | ".join(values) + " |")
        return "\n".join(lines_local)

    lines = [
        "# Coswara Shortcut Probe Feasibility",
        "",
        "## Scope",
        "",
        "This is an internal Coswara-only feasibility analysis. It can test whether teacher/student embeddings encode proxy shortcut variables inside Coswara, but it cannot measure true cross-dataset domain transfer because only Coswara is present.",
        "",
        "## Available Proxy Targets",
        "",
        f"- Records: `{len(metadata)}`",
        f"- Splits: `{metadata['split'].value_counts().to_dict()}`",
        f"- Labels: `{metadata['label'].value_counts().to_dict()}`",
        f"- Sex: `{metadata['sex_binary'].value_counts(dropna=False).to_dict()}`",
        f"- Country group: `{metadata['country_group'].value_counts(dropna=False).to_dict()}`",
        f"- Recording type: `{metadata['recording_type'].value_counts(dropna=False).to_dict()}`",
        f"- Symptom present: `{metadata['symptom_present'].value_counts(dropna=False).to_dict()}`",
        "",
        "## Probe Results",
        "",
    ]
    if probe_df.empty:
        lines.append("No valid probes were trained.")
    else:
        display = probe_df[["model", "target", "auc", "balanced_accuracy", "accuracy", "n_train", "n_test"]].copy()
        display = display.sort_values(["target", "model"])
        lines.append(_markdown_table(display))
        lines.extend(
            [
                "",
                "## KD Shortcut Transfer Check",
                "",
                "Positive `KD minus CE` means the KD student embedding makes that proxy easier to predict than the CE-only student embedding. This is not causal proof, but it is evidence that vanilla KD may preserve or amplify that proxy information.",
                "",
            ]
        )
        pivot = probe_df.pivot_table(index="target", columns="model", values="auc", aggfunc="first")
        if {"ce_student", "kd_student"}.issubset(set(pivot.columns)):
            pivot["kd_minus_ce"] = pivot["kd_student"] - pivot["ce_student"]
            pivot["teacher_minus_ce"] = pivot.get("panns_teacher", np.nan) - pivot["ce_student"]
            lines.append(_markdown_table(pivot.reset_index()))
    lines.extend(["", "## Skipped Probes", "", "```json", json.dumps(skipped, indent=2, ensure_ascii=False), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage1_panns_response_seed7")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/shortcut_probe_coswara")
    parser.add_argument("--panns-checkpoint", type=Path, default=ROOT / "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--panns-repo", type=Path, default=ROOT / "external/teacher_repos/audioset_tagging_cnn_zip")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--models", default="panns_teacher,ce_student,kd_student")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    label_to_index = {str(k): int(v) for k, v in config["label_to_index"].items()}
    checkpoints = {
        "panns_teacher": args.run_dir / "checkpoints/teacher_best.pt",
        "ce_student": args.run_dir / "checkpoints/ce_student_best.pt",
        "kd_student": args.run_dir / "checkpoints/student_best.pt",
    }
    requested_models = {item.strip() for item in args.models.split(",") if item.strip()}
    embeddings = {}
    for model_name, checkpoint in checkpoints.items():
        if model_name not in requested_models:
            continue
        embeddings[model_name] = _load_or_extract(
            model_name=model_name,
            checkpoint=checkpoint,
            manifest=args.manifest,
            root=args.root,
            label_to_index=label_to_index,
            device=args.device,
            batch_size=args.batch_size,
            max_duration_sec=args.max_duration_sec,
            max_records=args.max_records,
            panns_checkpoint=args.panns_checkpoint,
            panns_repo=args.panns_repo,
            cache_dir=args.out / "embeddings",
            refresh=args.refresh,
        )
        print(f"[done] embeddings for {model_name}: {len(embeddings[model_name])}", flush=True)
    metadata = _build_metadata(args.manifest, args.root)
    metadata.to_csv(args.out / "metadata_with_quality_proxy.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    probe_df, skipped = _run_probes(embeddings, metadata)
    probe_df.to_csv(args.out / "probe_results.csv", index=False)
    payload = {"skipped": skipped, "records": len(metadata)}
    (args.out / "probe_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.out / "PROBE_REPORT.md", probe_df, skipped, metadata)


if __name__ == "__main__":
    main()
