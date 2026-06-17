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
from coughkd.audio import read_wav_mono, resample_linear


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_modules() -> dict[str, bool]:
    import importlib.util

    modules = ["torch", "transformers", "librosa", "numpy"]
    return {module: importlib.util.find_spec(module) is not None for module in modules}


def _asset_present(model_path: Path) -> bool:
    required_any = ["config.json"]
    if not model_path.is_dir():
        return False
    return all((model_path / item).is_file() for item in required_any)


def _log_mel_image(audio_path: Path, root: Path, sample_rate: int, max_duration_sec: float) -> Any:
    import librosa
    import numpy as np

    samples, original_rate = read_wav_mono(root / audio_path)
    samples = resample_linear(samples, original_rate, sample_rate)
    if max_duration_sec > 0:
        samples = samples[: int(max_duration_sec * sample_rate)]
    if not samples:
        samples = [0.0]
    y = np.asarray(samples, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        fmin=20,
        fmax=sample_rate / 2,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    db = (db - float(db.mean())) / max(1e-6, float(db.std()))
    return db.astype(np.float32)


def _load_model(model_path: Path, device: Any) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True).to(device)
    model.eval()
    return processor, model


def _embedding_from_output(output: Any) -> Any:
    import torch
    from torch.nn import functional as F

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        embedding = output.pooler_output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
        embedding = hidden.mean(dim=1) if hidden.ndim == 3 else hidden
    elif isinstance(output, (tuple, list)) and output:
        first = output[0]
        embedding = first.mean(dim=1) if getattr(first, "ndim", 0) == 3 else first
    else:
        raise RuntimeError("Could not find a tensor embedding in HeAR model output.")
    if not torch.is_tensor(embedding):
        raise RuntimeError("HeAR model output embedding is not a tensor.")
    return F.normalize(embedding, dim=1)


def _processor_call(processor: Any, images: list[Any], device: Any) -> dict[str, Any]:
    try:
        batch = processor(images=images, return_tensors="pt")
    except TypeError:
        batch = processor(images, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items() if hasattr(value, "to")}


def _extract_hear_embeddings(
    *,
    manifest: Path,
    root: Path,
    model_path: Path,
    out_path: Path,
    batch_size: int,
    device_name: str,
    sample_rate: int,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if out_path.is_file():
        return json.loads(out_path.read_text(encoding="utf-8"))

    import torch

    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor, model = _load_model(model_path, device)
    rows = _manifest_rows(manifest)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            images = [_log_mel_image(Path(row["path"]), root, sample_rate, max_duration_sec) for row in chunk]
            batch = _processor_call(processor, images, device)
            output = model(**batch)
            embeddings = _embedding_from_output(output).detach().cpu().tolist()
            for row, embedding in zip(chunk, embeddings):
                records.append(
                    {
                        "recording_id": row["recording_id"],
                        "true_label": row["label"],
                        "model": "hear_pytorch_frozen",
                        "embedding": [float(value) for value in embedding],
                    }
                )
            batch_idx = start // batch_size + 1
            total = (len(rows) + batch_size - 1) // batch_size
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(f"[extract] hear pytorch batch {batch_idx}/{total}", flush=True)
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
        "# HeAR PyTorch Embedding Upper-Bound Audit",
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
            "## HeAR PyTorch Frozen Embedding",
            "",
        ]
    )
    for env in ("0", "1", "all"):
        item = summary["hear_pytorch_frozen"].get(env)
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
    parser.add_argument("--model-path", type=Path, default=ROOT / "pretrained/teachers/hear_pytorch")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/hear_pytorch_embedding_upper_bound_seed7")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    modules = _require_modules()
    preflight = {
        "model_path": str(args.model_path),
        "model_path_exists": args.model_path.exists(),
        "asset_present": _asset_present(args.model_path),
        "required_modules": modules,
        "environment_ready": all(modules.values()),
        "preflight_only": args.preflight_only,
        "expected_source": "https://huggingface.co/google/hear-pytorch",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.preflight_only or not preflight["asset_present"] or not preflight["environment_ready"]:
        reason = "preflight only requested" if args.preflight_only else "HeAR PyTorch local asset or environment is not ready"
        summary = {
            "preflight": preflight,
            "ran_gate": False,
            "decision_hint": f"{reason}; accept the Hugging Face terms and place/download google/hear-pytorch under the model path before running the gate.",
        }
        (args.out / "hear_pytorch_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_report(args.out / "HEAR_PYTORCH_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
        print(str(args.out / "HEAR_PYTORCH_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)
        return

    source_features = _read_csv(args.source_features)
    tos_features = _read_csv(args.tos_features)
    _, tos_envs = _assign_envs(source_features, tos_features, args.seed)
    manifest_ids = {row["recording_id"] for row in _manifest_rows(args.manifest)}
    tos_envs = {rid: env for rid, env in tos_envs.items() if rid in manifest_ids}
    hear_rows = _extract_hear_embeddings(
        manifest=args.manifest,
        root=args.root,
        model_path=args.model_path,
        out_path=args.out / "embeddings" / "hear_pytorch_frozen.json",
        batch_size=args.batch_size,
        device_name=args.device,
        sample_rate=args.sample_rate,
        max_duration_sec=args.max_duration_sec,
    )
    hear_summary = _summarize(hear_rows, tos_envs=tos_envs, seed=args.seed, folds=args.folds)
    env1_auc = float(hear_summary.get("1", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    all_auc = float(hear_summary.get("all", {}).get("embedding_cv_probe", {}).get("auroc", 0.0))
    gate = {
        "required_env1_auroc": 0.60,
        "required_all_tos_auroc_gt": 0.564254,
        "observed_env1_auroc": env1_auc,
        "observed_all_tos_auroc": all_auc,
    }
    if env1_auc >= 0.60 and all_auc > 0.564254:
        decision_hint = "HeAR PyTorch frozen embeddings clear the Tos gate; design a guarded foundation-KD candidate next."
    else:
        decision_hint = "HeAR PyTorch frozen embeddings do not clear the Tos gate; do not launch a HeAR-based positive KD experiment."
    summary = {
        "preflight": preflight,
        "ran_gate": True,
        "env_distributions": dict(Counter(str(value) for value in tos_envs.values())),
        "hear_pytorch_frozen": hear_summary,
        "gate": gate,
        "decision_hint": decision_hint,
    }
    (args.out / "hear_pytorch_embedding_upper_bound_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "HEAR_PYTORCH_EMBEDDING_UPPER_BOUND_AUDIT.md", summary)
    print(str(args.out / "HEAR_PYTORCH_EMBEDDING_UPPER_BOUND_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
