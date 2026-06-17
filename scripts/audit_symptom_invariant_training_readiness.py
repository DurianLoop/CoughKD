from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from audit_prediction_ensemble_frontier import LABELS, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


SOURCE_SYMPTOM_ALIASES = {
    "asthma",
    "cold",
    "cough",
    "fever",
    "ftg",
    "loss_of_smell",
    "mp",
    "pneumonia",
    "sore_throat",
}


def _split_symptoms(raw: object) -> list[str]:
    text = "" if pd.isna(raw) else str(raw).strip()
    if not text:
        return []
    tokens = []
    for part in re.split(r"[;,|]", text):
        token = part.strip()
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip().lower()
            if value in {"true", "1", "yes"}:
                tokens.append(key)
            elif value in {"false", "0", "no"}:
                tokens.append(f"not_{key}")
            else:
                tokens.append(f"{key}={value}")
        else:
            tokens.append(token.lower())
    return sorted(set(tokens))


def _symptom_matrix(df: pd.DataFrame, min_count: int) -> tuple[np.ndarray, list[str], list[list[str]]]:
    token_lists = [_split_symptoms(value) for value in df["symptoms"].fillna("")]
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocab = sorted(token for token, count in counts.items() if count >= min_count)
    if not vocab:
        return np.zeros((len(df), 0), dtype=float), [], token_lists
    mlb = MultiLabelBinarizer(classes=vocab)
    x = mlb.fit_transform(token_lists).astype(float)
    return x, vocab, token_lists


def _fit_predict_symptom_only(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    min_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    combined = pd.concat([train_df, eval_df], ignore_index=True)
    x_all, vocab, _ = _symptom_matrix(combined, min_count=min_count)
    x_train = x_all[: len(train_df)]
    x_eval = x_all[len(train_df) :]
    y_train = train_df["label"].reset_index(drop=True)
    y_eval = eval_df["label"].reset_index(drop=True)
    if x_train.shape[1] == 0 or y_train.nunique() < 2 or y_eval.nunique() < 2:
        return {"enabled": False, "reason": "insufficient_symptom_or_label_diversity"}, pd.DataFrame()
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_eval)
    scores = np.zeros((len(eval_df), len(LABELS)), dtype=float)
    for local_i, label in enumerate(model.classes_):
        if label in LABELS:
            scores[:, LABELS.index(label)] = pred[:, local_i]
    row_sum = scores.sum(axis=1, keepdims=True)
    scores = np.divide(scores, np.clip(row_sum, 1e-12, None))
    auc = _macro_ovr_auc(y_eval, scores)
    out = eval_df[["recording_id", "label"]].copy()
    for i, col in enumerate([f"symptom_prob_{label}" for label in LABELS]):
        out[col] = scores[:, i]
    return {
        "enabled": True,
        "n_train": int(len(train_df)),
        "n_eval": int(len(eval_df)),
        "n_features": int(len(vocab)),
        "features": vocab,
        "macro_ovr_auroc": float(auc),
    }, out


def _label_by_token(df: pd.DataFrame, token_lists: list[list[str]], top_tokens: list[str]) -> pd.DataFrame:
    rows = []
    labels = sorted(df["label"].dropna().astype(str).unique())
    for token in top_tokens:
        mask = np.asarray([token in tokens for tokens in token_lists], dtype=bool)
        if not mask.any():
            continue
        item = {"token": token, "n": int(mask.sum()), "coverage": float(mask.mean())}
        counts = df.loc[mask, "label"].value_counts()
        for label in labels:
            item[f"p_{label}"] = float(counts.get(label, 0) / max(1, int(mask.sum())))
        rows.append(item)
    return pd.DataFrame(rows)


def _dataset_summary(name: str, df: pd.DataFrame, min_count: int) -> tuple[dict[str, object], pd.DataFrame]:
    x, vocab, token_lists = _symptom_matrix(df, min_count=min_count)
    counts = Counter(token for tokens in token_lists for token in tokens)
    top = [token for token, _count in counts.most_common(20)]
    nonempty = np.asarray([bool(tokens) for tokens in token_lists])
    table = _label_by_token(df, token_lists, top)
    return {
        "name": name,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "symptom_nonempty_rate": float(nonempty.mean()) if len(nonempty) else 0.0,
        "n_tokens_total": int(len(counts)),
        "n_tokens_min_count": int(len(vocab)),
        "top_tokens": [{"token": token, "count": int(count)} for token, count in counts.most_common(20)],
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
    }, table


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
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--coughvid-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/symptom_invariant_training_readiness")
    args = parser.parse_args()

    source = pd.read_csv(args.source_manifest)
    coughvid = pd.read_csv(args.coughvid_manifest)
    source_train = source[source["split"] == "train"].reset_index(drop=True)
    source_test = source[source["split"] == "test"].reset_index(drop=True)
    coughvid_adapt = coughvid[coughvid["split"] == "adapt"].reset_index(drop=True)
    coughvid_test = coughvid[coughvid["split"] == "test"].reset_index(drop=True)

    source_summary, source_table = _dataset_summary("Coswara source", source, args.min_count)
    coughvid_summary, coughvid_table = _dataset_summary("COUGHVID", coughvid, args.min_count)

    source_symptom_model, source_symptom_predictions = _fit_predict_symptom_only(source_train, source_test, args.min_count)
    coughvid_symptom_model, coughvid_symptom_predictions = _fit_predict_symptom_only(
        coughvid_adapt,
        coughvid_test,
        args.min_count,
    )

    source_vocab = {item["token"] for item in source_summary["top_tokens"]}
    coughvid_vocab = {item["token"] for item in coughvid_summary["top_tokens"]}
    exact_overlap = sorted(source_vocab.intersection(coughvid_vocab))
    has_structured_target_resp = any(
        item["token"] in {"respiratory_condition", "not_respiratory_condition"}
        for item in coughvid_summary["top_tokens"]
    )
    has_source_resp_alias = any(item["token"] in SOURCE_SYMPTOM_ALIASES for item in source_summary["top_tokens"])

    decision_flags = {
        "source_symptoms_available": source_summary["symptom_nonempty_rate"] >= 0.40,
        "target_symptoms_available": coughvid_summary["symptom_nonempty_rate"] >= 0.90,
        "source_symptom_only_predictive": bool(
            source_symptom_model.get("enabled") and source_symptom_model.get("macro_ovr_auroc", 0.0) >= 0.55
        ),
        "target_symptom_only_predictive": bool(
            coughvid_symptom_model.get("enabled") and coughvid_symptom_model.get("macro_ovr_auroc", 0.0) >= 0.55
        ),
        "exact_token_overlap": bool(exact_overlap),
        "semantic_resp_alignment_possible": bool(has_structured_target_resp and has_source_resp_alias),
    }
    hard_gate_pass = (
        decision_flags["source_symptoms_available"]
        and decision_flags["target_symptoms_available"]
        and decision_flags["semantic_resp_alignment_possible"]
        and decision_flags["source_symptom_only_predictive"]
        and decision_flags["target_symptom_only_predictive"]
    )

    args.out.mkdir(parents=True, exist_ok=True)
    source_table.to_csv(args.out / "source_symptom_label_table.csv", index=False)
    coughvid_table.to_csv(args.out / "coughvid_symptom_label_table.csv", index=False)
    if not source_symptom_predictions.empty:
        source_symptom_predictions.to_csv(args.out / "source_symptom_only_predictions.csv", index=False)
    if not coughvid_symptom_predictions.empty:
        coughvid_symptom_predictions.to_csv(args.out / "coughvid_symptom_only_predictions.csv", index=False)

    payload = {
        "source": source_summary,
        "coughvid": coughvid_summary,
        "source_symptom_only_model": source_symptom_model,
        "coughvid_symptom_only_model": coughvid_symptom_model,
        "exact_top_token_overlap": exact_overlap,
        "decision_flags": decision_flags,
        "hard_gate_pass": bool(hard_gate_pass),
        "candidate_if_pass": "Symptom-confound risk equalization KD, using source symptoms as risk groups and target symptoms only for audit/calibration gates.",
    }
    (args.out / "symptom_invariant_training_readiness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Symptom-Invariant Training Readiness",
        "",
        "This audit checks whether a symptom-confound training mechanism is worth a real training run.",
        "",
        "## Decision",
        "",
        f"Hard gate pass: `{hard_gate_pass}`",
        "",
        "```json",
        json.dumps(decision_flags, indent=2),
        "```",
        "",
        "## Symptom-Only Models",
        "",
        "Source train -> source test:",
        "",
        "```json",
        json.dumps(source_symptom_model, indent=2),
        "```",
        "",
        "COUGHVID adapt -> COUGHVID test:",
        "",
        "```json",
        json.dumps(coughvid_symptom_model, indent=2),
        "```",
        "",
        "## Top Source Symptom Tokens",
        "",
        _to_md(pd.DataFrame(source_summary["top_tokens"])),
        "",
        "## Top COUGHVID Symptom Tokens",
        "",
        _to_md(pd.DataFrame(coughvid_summary["top_tokens"])),
        "",
        "## Source Token Label Table",
        "",
        _to_md(source_table.head(12)),
        "",
        "## COUGHVID Token Label Table",
        "",
        _to_md(coughvid_table.head(12)),
        "",
    ]
    report = args.out / "SYMPTOM_INVARIANT_TRAINING_READINESS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
