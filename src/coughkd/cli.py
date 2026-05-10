"""Command-line entry points for COUGHKD foundation tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio import preprocess_manifest_smoke
from .augment import add_uniform_noise, specaugment, time_shift
from .benchmark import benchmark_report
from .baselines import baseline_smoke_report
from .cache import save_prediction_cache
from .config import RunConfig, save_run_metadata, set_seed
from .datasets import build_coswara_manifest, build_manifest_from_directory, build_manifest_from_metadata, dataset_smoke_report
from .grid import ablation_grid, aggregate_results, run_smoke_grid
from .manifest import (
    check_external_selection_guard,
    filter_manifest_rows,
    read_manifest,
    subject_disjoint_split,
    validate_manifest,
    write_manifest,
    write_validation_report,
)
from .metrics import (
    binary_classification_report,
    bootstrap_auc_ci,
    external_drop_report,
    multiclass_ovr_auroc,
    write_report,
)
from .losses import coughkd_loss
from .models import SmokeStudent, SmokeTeacher
from .paper_tables import generate_paper_tables
from .reporting import assert_no_unsupported_clinical_claims, subgroup_report, write_subgroup_report
from .segmentation import aggregate_scores, merge_intervals, sliding_windows
from .smoke import make_smoke_data
from .torch_models import run_prelong_check, run_torch_manifest_smoke, run_torch_smoke, run_torch_training


def cmd_make_smoke_data(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    out_dir = Path(args.out)
    manifest = make_smoke_data(out_dir, seed=args.seed)
    save_run_metadata(out_dir, RunConfig(experiment_name="make_smoke_data", seed=args.seed), sys.argv)
    print(f"Wrote smoke manifest: {manifest}")
    return 0


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    issues, summary = validate_manifest(Path(args.manifest), Path(args.root))
    write_validation_report(issues, summary, Path(args.out))
    error_count = sum(1 for issue in issues if issue.severity == "error")
    print(json.dumps(summary, indent=2))
    return 1 if error_count else 0


def cmd_split_manifest(args: argparse.Namespace) -> int:
    rows = read_manifest(Path(args.manifest))
    split_rows = subject_disjoint_split(rows, seed=args.seed)
    out_dir = Path(args.out)
    out_manifest = out_dir / "manifest_split.csv"
    write_manifest(split_rows, out_manifest)
    issues, summary = validate_manifest(out_manifest, Path(args.root))
    write_validation_report(issues, summary, out_dir)
    if any(issue.severity == "error" for issue in issues):
        print(json.dumps(summary, indent=2))
        return 1
    print(f"Wrote split manifest: {out_manifest}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_filter_manifest(args: argparse.Namespace) -> int:
    rows = read_manifest(Path(args.manifest))
    drop_labels = {item.strip() for item in args.drop_labels.split(",") if item.strip()}
    kept, filter_report = filter_manifest_rows(
        rows,
        root=Path(args.root),
        min_duration_sec=args.min_duration_sec,
        drop_labels=drop_labels,
    )
    out_dir = Path(args.out)
    out_manifest = out_dir / "manifest_filtered.csv"
    write_manifest(kept, out_manifest)
    issues, summary = validate_manifest(out_manifest, Path(args.root))
    write_validation_report(issues, summary, out_dir)
    payload = {"filter": filter_report, "validation": summary, "errors": [issue.__dict__ for issue in issues if issue.severity == "error"]}
    (out_dir / "filter_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if payload["errors"] else 0


def cmd_preprocess_smoke(args: argparse.Namespace) -> int:
    config = RunConfig(experiment_name="preprocess_smoke", seed=args.seed)
    set_seed(config.seed)
    out_dir = Path(args.out)
    save_run_metadata(out_dir, config, sys.argv)
    report = preprocess_manifest_smoke(Path(args.manifest), Path(args.root), out_dir, config)
    print(json.dumps(report, indent=2))
    return 0 if report["num_valid"] == report["num_recordings"] else 1


def cmd_metrics_smoke(args: argparse.Namespace) -> int:
    labels = [0, 0, 1, 1, 0, 1]
    scores = [0.05, 0.30, 0.55, 0.90, 0.40, 0.80]
    report = binary_classification_report(labels, scores)
    report.update({f"auroc_ci_{key}": value for key, value in bootstrap_auc_ci(labels, scores, n_bootstrap=50).items()})
    report.update({f"drop_{key}": value for key, value in external_drop_report(0.90, 0.82).items()})
    multiclass = multiclass_ovr_auroc(
        ["a", "b", "c", "a"],
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1]],
        ["a", "b", "c"],
    )
    report.update({f"multiclass_{key}": value for key, value in multiclass.items()})
    write_report(report, Path(args.out))
    print(json.dumps(report, indent=2))
    if abs(report["auroc"] - 1.0) > 1e-12:
        return 1
    return 0


def cmd_check_selection_guard(args: argparse.Namespace) -> int:
    rows = read_manifest(Path(args.manifest))
    allowed = {item.strip() for item in args.selection_splits.split(",") if item.strip()}
    issues = check_external_selection_guard(rows, allowed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"selection_splits": sorted(allowed), "issues": [issue.__dict__ for issue in issues]}
    (out_dir / "selection_guard.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if issues else 0


def cmd_aggregation_smoke(args: argparse.Namespace) -> int:
    windows = sliding_windows(duration_sec=2.2, window_sec=1.0, hop_sec=0.5)
    merged = merge_intervals([(0.0, 0.4), (0.45, 0.8), (1.5, 1.8)], gap_sec=0.1)
    scores = [0.2, 0.8, 0.6, 0.4]
    qualities = [0.1, 0.9, 0.8, 0.2]
    report = {
        "num_windows": len(windows),
        "windows": [window.__dict__ for window in windows],
        "merged": merged,
        "mean": aggregate_scores(scores, method="mean"),
        "max": aggregate_scores(scores, method="max"),
        "topk": aggregate_scores(scores, method="topk", top_k=2),
        "quality_topk": aggregate_scores(scores, qualities=qualities, method="quality_topk", top_k=2),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregation_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["num_windows"] == 4 and len(report["merged"]) == 2 else 1


def cmd_augment_smoke(args: argparse.Namespace) -> int:
    samples = [0.0, 0.1, -0.1, 0.2]
    noisy = add_uniform_noise(samples, amplitude=0.01, seed=7)
    shifted = time_shift(samples, shift=1)
    features = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    masked = specaugment(features, time_mask=1, freq_mask=1, seed=7)
    report = {"noisy": noisy, "shifted": shifted, "masked": masked}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "augment_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if shifted == [0.2, 0.0, 0.1, -0.1] else 1


def cmd_model_smoke(args: argparse.Namespace) -> int:
    features = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]]
    teacher = SmokeTeacher()
    student = SmokeStudent()
    teacher_out = teacher.forward(features)
    student_out = student.forward(features)
    loss = coughkd_loss(
        label=1,
        teacher_logits=teacher_out.logits,
        student_logits=student_out.logits,
        teacher_features=teacher_out.features,
        student_features=student_out.features,
        teacher_attention=teacher_out.attention,
        student_attention=student_out.attention,
        teacher_batch_embeddings=[teacher_out.embedding, teacher_out.embedding],
        student_batch_embeddings=[student_out.embedding, student_out.embedding],
    )
    out_dir = Path(args.out)
    teacher_cache = save_prediction_cache(
        out_dir,
        recording_id="smoke_recording",
        model_id=teacher.name,
        config=RunConfig(experiment_name="model_smoke"),
        logits=teacher_out.logits,
        embedding=teacher_out.embedding,
    )
    report = {
        "teacher": {"name": teacher.name, "params": teacher.parameter_count(), "logits": teacher_out.logits},
        "student": {"name": student.name, "params": student.parameter_count(), "logits": student_out.logits},
        "loss": loss,
        "teacher_prediction_cache": str(teacher_cache),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if loss["total"] > 0 and teacher.parameter_count() >= student.parameter_count() else 1


def cmd_grid_dry_run(args: argparse.Namespace) -> int:
    grid = ablation_grid()
    preview = grid[: args.limit]
    payload = {"num_total": len(grid), "preview": preview}
    print(json.dumps(payload, indent=2))
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "grid_dry_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def cmd_grid_smoke(args: argparse.Namespace) -> int:
    summary = run_smoke_grid(Path(args.out), force=args.force, limit=args.limit)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failures"] else 0


def cmd_aggregate_results(args: argparse.Namespace) -> int:
    summary = aggregate_results(Path(args.runs_dir), Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0 if summary["num_runs"] >= args.min_runs else 1


def cmd_benchmark_smoke(args: argparse.Namespace) -> int:
    model = SmokeStudent()
    features = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.0]]
    report = benchmark_report(model, features, Path(args.out))
    print(json.dumps(report, indent=2))
    return 0 if report["parameter_count"] > 0 and report["latency"]["repeats"] > 0 else 1


def cmd_subgroup_smoke(args: argparse.Namespace) -> int:
    labels = [0, 1, 0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    metadata = [
        {"sex": "f", "device": "phone"},
        {"sex": "f", "device": "phone"},
        {"sex": "f", "device": "phone"},
        {"sex": "f", "device": "phone"},
        {"sex": "m", "device": "laptop"},
        {"sex": "m", "device": "laptop"},
        {"sex": "m", "device": "laptop"},
        {"sex": "m", "device": "laptop"},
    ]
    report = subgroup_report(labels, scores, metadata, fields=["sex", "device"], min_n=4)
    write_subgroup_report(report, Path(args.out))
    assert_no_unsupported_clinical_claims("screening research model for respiratory audio")
    print(json.dumps(report, indent=2))
    return 0


def cmd_paper_tables_smoke(args: argparse.Namespace) -> int:
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    required = [item.strip() for item in args.required_run_ids.split(",") if item.strip()]
    audit = generate_paper_tables(runs_dir, out_dir, required)
    print(json.dumps(audit, indent=2))
    return 0


def cmd_baseline_smoke(args: argparse.Namespace) -> int:
    report = baseline_smoke_report(Path(args.out))
    print(json.dumps(report, indent=2))
    return 0 if len(report["baselines"]) >= 4 else 1


def cmd_build_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root)
    out_manifest = Path(args.out)
    if args.metadata:
        report = build_manifest_from_metadata(
            root=root,
            metadata_csv=Path(args.metadata),
            out_manifest=out_manifest,
            dataset=args.dataset,
            path_column=args.path_column,
            subject_column=args.subject_column,
            label_column=args.label_column,
            split_column=args.split_column,
            default_split=args.default_split,
        )
    else:
        report = build_manifest_from_directory(
            root=root,
            out_manifest=out_manifest,
            dataset=args.dataset,
            label_from_parent=args.label_from_parent,
            default_label=args.default_label,
            default_split=args.default_split,
        )
    print(json.dumps(report.__dict__, indent=2))
    return 1 if report.validation_errors else 0


def cmd_build_coswara_manifest(args: argparse.Namespace) -> int:
    recording_types = [item.strip() for item in args.recording_types.split(",") if item.strip()]
    report = build_coswara_manifest(
        root=Path(args.root),
        metadata_csv=Path(args.metadata),
        out_manifest=Path(args.out),
        extracted_dir=args.extracted_dir,
        recording_types=recording_types,
        default_split=args.default_split,
    )
    print(json.dumps(report.__dict__, indent=2))
    return 1 if report.validation_errors else 0


def cmd_dataset_smoke(args: argparse.Namespace) -> int:
    report = dataset_smoke_report(Path(args.out))
    print(json.dumps(report, indent=2))
    return 1 if report["build"]["validation_errors"] else 0


def cmd_torch_smoke(args: argparse.Namespace) -> int:
    try:
        report = run_torch_smoke(Path(args.out), device=args.device, batch_size=args.batch_size)
    except RuntimeError as exc:
        print(json.dumps({"status": "skipped", "reason": str(exc)}, indent=2))
        return 77 if args.allow_missing_torch else 1
    print(json.dumps(report, indent=2))
    return 0 if report["student_params"] < report["teacher_params"] and report["initial_loss"] > 0 else 1


def cmd_torch_manifest_smoke(args: argparse.Namespace) -> int:
    try:
        report = run_torch_manifest_smoke(
            manifest=Path(args.manifest),
            root=Path(args.root),
            out_dir=Path(args.out),
            device=args.device,
            batch_size=args.batch_size,
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "skipped", "reason": str(exc)}, indent=2))
        return 77 if args.allow_missing_torch else 1
    print(json.dumps(report, indent=2))
    ok = report["dataset_records"] > 0 and report["student_params"] < report["teacher_params"] and report["initial_loss"] > 0
    return 0 if ok else 1


def cmd_torch_train(args: argparse.Namespace) -> int:
    try:
        report = run_torch_training(
            manifest=Path(args.manifest),
            root=Path(args.root),
            out_dir=Path(args.out),
            device=args.device,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            num_workers=args.num_workers,
            max_records_per_split=args.max_records_per_split,
            max_duration_sec=args.max_duration_sec,
            teacher_kind=args.teacher_kind,
            teacher_checkpoint=Path(args.teacher_checkpoint) if args.teacher_checkpoint else None,
            teacher_repo=Path(args.teacher_repo) if args.teacher_repo else None,
            kd_temperature=args.kd_temperature,
            kd_response_weight=args.kd_response_weight,
            kd_feature_weight=args.kd_feature_weight,
            kd_embedding_weight=args.kd_embedding_weight,
            kd_relation_weight=args.kd_relation_weight,
            label_smoothing=args.label_smoothing,
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "metrics": {
                    "teacher_test": report["teacher_test"],
                    "ce_student_test": report["ce_student_test"],
                    "student_test": report["student_test"],
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_prelong_check(args: argparse.Namespace) -> int:
    try:
        report = run_prelong_check(
            manifest=Path(args.manifest),
            root=Path(args.root),
            out_dir=Path(args.out),
            device=args.device,
            expected_python=args.expected_python,
        )
    except RuntimeError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coughkd")
    sub = parser.add_subparsers(required=True)

    make_smoke = sub.add_parser("make-smoke-data")
    make_smoke.add_argument("--out", required=True)
    make_smoke.add_argument("--seed", type=int, default=7)
    make_smoke.set_defaults(func=cmd_make_smoke_data)

    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--root", default=".")
    validate.add_argument("--out", required=True)
    validate.set_defaults(func=cmd_validate_manifest)

    split = sub.add_parser("split-manifest")
    split.add_argument("--manifest", required=True)
    split.add_argument("--root", default=".")
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=7)
    split.set_defaults(func=cmd_split_manifest)

    filter_manifest = sub.add_parser("filter-manifest")
    filter_manifest.add_argument("--manifest", required=True)
    filter_manifest.add_argument("--root", default=".")
    filter_manifest.add_argument("--out", required=True)
    filter_manifest.add_argument("--min-duration-sec", type=float)
    filter_manifest.add_argument("--drop-labels", default="")
    filter_manifest.set_defaults(func=cmd_filter_manifest)

    preprocess = sub.add_parser("preprocess-smoke")
    preprocess.add_argument("--manifest", required=True)
    preprocess.add_argument("--root", default=".")
    preprocess.add_argument("--out", required=True)
    preprocess.add_argument("--seed", type=int, default=7)
    preprocess.set_defaults(func=cmd_preprocess_smoke)

    metrics = sub.add_parser("metrics-smoke")
    metrics.add_argument("--out", required=True)
    metrics.set_defaults(func=cmd_metrics_smoke)

    guard = sub.add_parser("check-selection-guard")
    guard.add_argument("--manifest", required=True)
    guard.add_argument("--selection-splits", default="train,val")
    guard.add_argument("--out", required=True)
    guard.set_defaults(func=cmd_check_selection_guard)

    aggregation = sub.add_parser("aggregation-smoke")
    aggregation.add_argument("--out", required=True)
    aggregation.set_defaults(func=cmd_aggregation_smoke)

    augment = sub.add_parser("augment-smoke")
    augment.add_argument("--out", required=True)
    augment.set_defaults(func=cmd_augment_smoke)

    model = sub.add_parser("model-smoke")
    model.add_argument("--out", required=True)
    model.set_defaults(func=cmd_model_smoke)

    grid_dry = sub.add_parser("grid-dry-run")
    grid_dry.add_argument("--out")
    grid_dry.add_argument("--limit", type=int, default=5)
    grid_dry.set_defaults(func=cmd_grid_dry_run)

    grid_smoke = sub.add_parser("grid-smoke")
    grid_smoke.add_argument("--out", required=True)
    grid_smoke.add_argument("--limit", type=int, default=2)
    grid_smoke.add_argument("--force", action="store_true")
    grid_smoke.set_defaults(func=cmd_grid_smoke)

    aggregate = sub.add_parser("aggregate-results")
    aggregate.add_argument("--runs-dir", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--min-runs", type=int, default=1)
    aggregate.set_defaults(func=cmd_aggregate_results)

    benchmark = sub.add_parser("benchmark-smoke")
    benchmark.add_argument("--out", required=True)
    benchmark.set_defaults(func=cmd_benchmark_smoke)

    subgroup = sub.add_parser("subgroup-smoke")
    subgroup.add_argument("--out", required=True)
    subgroup.set_defaults(func=cmd_subgroup_smoke)

    tables = sub.add_parser("paper-tables-smoke")
    tables.add_argument("--runs-dir", required=True)
    tables.add_argument("--out", required=True)
    tables.add_argument("--required-run-ids", required=True)
    tables.set_defaults(func=cmd_paper_tables_smoke)

    baseline = sub.add_parser("baseline-smoke")
    baseline.add_argument("--out", required=True)
    baseline.set_defaults(func=cmd_baseline_smoke)

    build_manifest = sub.add_parser("build-manifest")
    build_manifest.add_argument("--root", required=True)
    build_manifest.add_argument("--out", required=True)
    build_manifest.add_argument("--dataset", required=True)
    build_manifest.add_argument("--metadata")
    build_manifest.add_argument("--path-column")
    build_manifest.add_argument("--subject-column")
    build_manifest.add_argument("--label-column")
    build_manifest.add_argument("--split-column")
    build_manifest.add_argument("--default-label", default="unknown")
    build_manifest.add_argument("--default-split", default="unassigned")
    build_manifest.add_argument("--label-from-parent", action="store_true", default=True)
    build_manifest.set_defaults(func=cmd_build_manifest)

    coswara_manifest = sub.add_parser("build-coswara-manifest")
    coswara_manifest.add_argument("--root", required=True)
    coswara_manifest.add_argument("--metadata", default="combined_data.csv")
    coswara_manifest.add_argument("--out", required=True)
    coswara_manifest.add_argument("--extracted-dir", default="Extracted_data")
    coswara_manifest.add_argument("--recording-types", default="cough-heavy,cough-shallow")
    coswara_manifest.add_argument("--default-split", default="unassigned")
    coswara_manifest.set_defaults(func=cmd_build_coswara_manifest)

    dataset_smoke = sub.add_parser("dataset-smoke")
    dataset_smoke.add_argument("--out", required=True)
    dataset_smoke.set_defaults(func=cmd_dataset_smoke)

    torch_smoke = sub.add_parser("torch-smoke")
    torch_smoke.add_argument("--out", required=True)
    torch_smoke.add_argument("--device", default="auto")
    torch_smoke.add_argument("--batch-size", type=int, default=4)
    torch_smoke.add_argument("--allow-missing-torch", action="store_true")
    torch_smoke.set_defaults(func=cmd_torch_smoke)

    torch_manifest = sub.add_parser("torch-manifest-smoke")
    torch_manifest.add_argument("--manifest", required=True)
    torch_manifest.add_argument("--root", required=True)
    torch_manifest.add_argument("--out", required=True)
    torch_manifest.add_argument("--device", default="auto")
    torch_manifest.add_argument("--batch-size", type=int, default=4)
    torch_manifest.add_argument("--allow-missing-torch", action="store_true")
    torch_manifest.set_defaults(func=cmd_torch_manifest_smoke)

    torch_train = sub.add_parser("torch-train")
    torch_train.add_argument("--manifest", required=True)
    torch_train.add_argument("--root", required=True)
    torch_train.add_argument("--out", required=True)
    torch_train.add_argument("--device", default="auto")
    torch_train.add_argument("--teacher-epochs", type=int, default=8)
    torch_train.add_argument("--student-epochs", type=int, default=8)
    torch_train.add_argument("--batch-size", type=int, default=64)
    torch_train.add_argument("--lr", type=float, default=1e-3)
    torch_train.add_argument("--seed", type=int, default=7)
    torch_train.add_argument("--num-workers", type=int, default=0)
    torch_train.add_argument("--max-records-per-split", type=int)
    torch_train.add_argument("--max-duration-sec", type=float, default=4.0)
    torch_train.add_argument("--teacher-kind", default="compact", choices=["compact", "panns_cnn14_16k"])
    torch_train.add_argument("--teacher-checkpoint")
    torch_train.add_argument("--teacher-repo")
    torch_train.add_argument("--kd-temperature", type=float, default=2.0)
    torch_train.add_argument("--kd-response-weight", type=float, default=0.7)
    torch_train.add_argument("--kd-feature-weight", type=float, default=0.1)
    torch_train.add_argument("--kd-embedding-weight", type=float, default=0.0)
    torch_train.add_argument("--kd-relation-weight", type=float, default=0.0)
    torch_train.add_argument("--label-smoothing", type=float, default=0.0)
    torch_train.set_defaults(func=cmd_torch_train)

    prelong = sub.add_parser("prelong-check")
    prelong.add_argument("--manifest", required=True)
    prelong.add_argument("--root", required=True)
    prelong.add_argument("--out", required=True)
    prelong.add_argument("--device", default="auto")
    prelong.add_argument("--expected-python")
    prelong.set_defaults(func=cmd_prelong_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
