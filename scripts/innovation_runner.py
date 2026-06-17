from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


SEEDS = (7, 11, 23)
SOURCE_MANIFEST = "runs/coswara_cough_filtered_split/manifest_split.csv"
TARGET_MANIFEST = "manifests/coughvid_adapt_test.csv"
DATA_ROOT = r"D:\CoughKD"
PANNS_REPO = "external/teacher_repos/audioset_tagging_cnn_zip"
TEACHER_CKPT = "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth"


@dataclass(frozen=True)
class Task:
    name: str
    done_file: Path
    command: list[str]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display: str
    kind: str
    checkpoint_template: str | None
    external_template: str
    probe_template: str | None
    probe_model: str | None
    stage: str


METHODS: list[MethodSpec] = [
    MethodSpec("ce", "CE-only student", "baseline", None, "runs/external_coughvid_test_ce_{tag}", "runs/domain_probe_stage1_panns_seed{seed}", "ce_student", "baseline"),
    MethodSpec("kd", "Vanilla KD student", "baseline", None, "runs/external_coughvid_test_kd_{tag}", "runs/domain_probe_stage1_panns_seed{seed}", "kd_student", "baseline"),
    MethodSpec("source_only", "Source-only continuation", "control", "runs/stage3c_source_only_seed{seed}/checkpoints/student_best.pt", "runs/external_coughvid_test_stage3c_source_only_seed{seed}", "runs/domain_probe_stage3c_tcd_seed{seed}", "source_only", "current"),
    MethodSpec("tcd_very_strong", "TCD very strong", "current", "runs/stage3c_tcd_very_strong_seed{seed}/checkpoints/student_best.pt", "runs/external_coughvid_test_stage3c_tcd_very_strong_seed{seed}", "runs/domain_probe_stage3c_tcd_seed{seed}", "tcd_very_strong", "current"),
    MethodSpec("tcd_conf035", "TCD confidence 0.35", "current", "runs/stage3c_tcd_conf035_seed{seed}/checkpoints/student_best.pt", "runs/external_coughvid_test_stage3c_tcd_conf035_seed{seed}", "runs/domain_probe_stage3c_tcd_seed{seed}", "tcd_conf035", "current"),
    MethodSpec("candidate_a", "Candidate A shortcut-suppressed KD", "candidate", "runs/candidate_a_shortcut_suppressed_seed{seed}/checkpoints/student_best.pt", "runs/external_coughvid_test_candidate_a_seed{seed}", "runs/domain_probe_candidate_a_seed{seed}", "candidate_a", "candidate_a"),
    MethodSpec("candidate_b", "Candidate B disagreement-gated KD", "candidate", "runs/candidate_b_disagreement_gated_seed{seed}/checkpoints/student_best.pt", "runs/external_coughvid_test_candidate_b_seed{seed}", "runs/domain_probe_candidate_b_seed{seed}", "candidate_b", "candidate_b"),
    MethodSpec("candidate_c", "Candidate C probe-adversarial student", "candidate", "runs/candidate_c_probe_adv_seed{seed}/student_domain_adv.pt", "runs/external_coughvid_test_candidate_c_seed{seed}", "runs/domain_probe_candidate_c_seed{seed}", "candidate_c", "candidate_c"),
]


def _rel(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _seed_tag(seed: int) -> str:
    return "baseline" if seed == 7 else f"seed{seed}"


def _external_dir(method: MethodSpec, seed: int) -> Path:
    return _rel(method.external_template.format(seed=seed, tag=_seed_tag(seed)))


def _probe_dir(method: MethodSpec, seed: int) -> Path | None:
    if method.probe_template is None:
        return None
    if seed == 7 and method.name in {"ce", "kd"}:
        stage1 = ROOT / "runs/domain_probe_stage1_panns_seed7"
        if stage1.is_dir():
            return stage1
        legacy = ROOT / "runs/domain_probe_coswara_coughvid_students"
        if legacy.is_dir():
            return legacy
    if seed == 7 and method.name in {"source_only", "tcd_very_strong", "tcd_conf035"}:
        stage3c = ROOT / "runs/domain_probe_stage3c_tcd_seed7"
        if stage3c.is_dir():
            return stage3c
        legacy = ROOT / "runs/domain_probe_stage3b_tcd_seed7"
        if legacy.is_dir():
            return legacy
    return _rel(method.probe_template.format(seed=seed, tag=_seed_tag(seed)))


def _checkpoint(method: MethodSpec, seed: int) -> Path:
    if method.name == "ce":
        return ROOT / f"runs/stage1_panns_response_seed{seed}/checkpoints/ce_student_best.pt"
    if method.name == "kd":
        return ROOT / f"runs/stage1_panns_response_seed{seed}/checkpoints/student_best.pt"
    if method.name == "source_only" and seed == 7:
        legacy = ROOT / "runs/stage3b_source_only_seed7/checkpoints/student_best.pt"
        if legacy.is_file():
            return legacy
    if method.name == "tcd_very_strong" and seed == 7:
        legacy = ROOT / "runs/stage3b_tcd_very_strong_seed7/checkpoints/student_best.pt"
        if legacy.is_file():
            return legacy
    if method.name == "tcd_conf035" and seed == 7:
        legacy = ROOT / "runs/stage3b_tcd_conf035_seed7/checkpoints/student_best.pt"
        if legacy.is_file():
            return legacy
    if method.checkpoint_template is None:
        raise ValueError(f"method has no checkpoint template: {method.name}")
    return _rel(method.checkpoint_template.format(seed=seed, tag=_seed_tag(seed)))


def _completed(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _cmd(*parts: str | Path) -> list[str]:
    return [str(part) for part in parts]


def _stage1_task(seed: int) -> Task:
    out = ROOT / f"runs/stage1_panns_response_seed{seed}/metrics.json"
    return Task(
        name=f"stage1_panns_seed{seed}",
        done_file=out,
        command=_cmd(
            sys.executable,
            "-B",
            "-m",
            "coughkd.cli",
            "torch-train",
            "--manifest",
            SOURCE_MANIFEST,
            "--root",
            DATA_ROOT,
            "--out",
            f"runs/stage1_panns_response_seed{seed}",
            "--device",
            "auto",
            "--teacher-kind",
            "panns_cnn14_16k",
            "--teacher-checkpoint",
            TEACHER_CKPT,
            "--teacher-repo",
            PANNS_REPO,
            "--teacher-epochs",
            "8",
            "--student-epochs",
            "8",
            "--batch-size",
            "16",
            "--seed",
            str(seed),
            "--kd-temperature",
            "2.0",
            "--kd-response-weight",
            "0.7",
            "--kd-feature-weight",
            "0.1",
            "--kd-embedding-weight",
            "0.0",
            "--kd-relation-weight",
            "0.0",
        ),
    )


def _external_task(method: MethodSpec, seed: int) -> Task:
    out = _external_dir(method, seed) / "metrics.json"
    return Task(
        name=f"external_{method.name}_seed{seed}",
        done_file=out,
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/evaluate_external_checkpoint.py",
            "--manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--checkpoint",
            _checkpoint(method, seed),
            "--out",
            _external_dir(method, seed),
            "--device",
            "auto",
            "--batch-size",
            "16",
        ),
    )


def _baseline_probe_task(seed: int) -> Task:
    out = ROOT / f"runs/domain_probe_stage1_panns_seed{seed}/domain_probe_results.csv"
    return Task(
        name=f"probe_stage1_seed{seed}",
        done_file=out,
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/domain_probe_students.py",
            "--coswara-manifest",
            SOURCE_MANIFEST,
            "--coughvid-manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--run-dir",
            f"runs/stage1_panns_response_seed{seed}",
            "--out",
            f"runs/domain_probe_stage1_panns_seed{seed}",
            "--samples-per-dataset",
            "1500",
            "--seed",
            str(seed),
            "--batch-size",
            "16",
            "--device",
            "auto",
        ),
    )


def _stage3_train_task(method: MethodSpec, seed: int) -> Task:
    checkpoint = _checkpoint(method, seed)
    target_weight = {"source_only": "0.0", "tcd_very_strong": "1.0", "tcd_conf035": "0.60"}[method.name]
    threshold = {"source_only": "0.0", "tcd_very_strong": "0.0", "tcd_conf035": "0.35"}[method.name]
    return Task(
        name=f"train_{method.name}_seed{seed}",
        done_file=checkpoint,
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/train_target_consistency_student.py",
            "--source-manifest",
            SOURCE_MANIFEST,
            "--target-manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--init-student-checkpoint",
            "runs/stage1_panns_response_seed7/checkpoints/student_best.pt",
            "--init-teacher-checkpoint",
            "runs/stage1_panns_response_seed7/checkpoints/teacher_best.pt",
            "--teacher-kind",
            "panns_cnn14_16k",
            "--teacher-checkpoint",
            TEACHER_CKPT,
            "--teacher-repo",
            PANNS_REPO,
            "--out",
            f"runs/stage3c_{method.name}_seed{seed}",
            "--device",
            "auto",
            "--epochs",
            "6",
            "--batch-size",
            "16",
            "--seed",
            str(seed),
            "--lr",
            "5e-5",
            "--target-weight",
            target_weight,
            "--confidence-threshold",
            threshold,
        ),
    )


def _stage3_probe_task(seed: int) -> Task:
    out = ROOT / f"runs/domain_probe_stage3c_tcd_seed{seed}/domain_probe_results.csv"
    return Task(
        name=f"probe_stage3c_seed{seed}",
        done_file=out,
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/domain_probe_students.py",
            "--coswara-manifest",
            SOURCE_MANIFEST,
            "--coughvid-manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--run-dir",
            "runs/stage1_panns_response_seed7",
            "--out",
            f"runs/domain_probe_stage3c_tcd_seed{seed}",
            "--samples-per-dataset",
            "1500",
            "--seed",
            str(seed),
            "--batch-size",
            "16",
            "--device",
            "auto",
            "--extra-checkpoint",
            f"source_only=runs/stage3c_source_only_seed{seed}/checkpoints/student_best.pt",
            "--extra-checkpoint",
            f"tcd_very_strong=runs/stage3c_tcd_very_strong_seed{seed}/checkpoints/student_best.pt",
            "--extra-checkpoint",
            f"tcd_conf035=runs/stage3c_tcd_conf035_seed{seed}/checkpoints/student_best.pt",
        ),
    )


def _candidate_a_tasks(seed: int) -> list[Task]:
    weights = ROOT / f"runs/shortcut_weights/candidate_a_seed{seed}/shortcut_weights.csv"
    train = ROOT / f"runs/candidate_a_shortcut_suppressed_seed{seed}/checkpoints/student_best.pt"
    return [
        Task(
            name=f"weights_candidate_a_seed{seed}",
            done_file=weights,
            command=_cmd(
                sys.executable,
                "-B",
                "scripts/build_shortcut_weights.py",
                "--source-manifest",
                SOURCE_MANIFEST,
                "--target-manifest",
                TARGET_MANIFEST,
                "--root",
                DATA_ROOT,
                "--out",
                f"runs/shortcut_weights/candidate_a_seed{seed}",
                "--source-split",
                "train",
                "--target-split",
                "adapt",
                "--stability-checkpoint",
                f"runs/stage1_panns_response_seed{seed}/checkpoints/student_best.pt",
                "--device",
                "auto",
                "--seed",
                str(seed),
                "--quality-power",
                "0.25",
                "--domain-power",
                "0.15",
                "--stability-power",
                "0.25",
                "--floor",
                "0.50",
            ),
        ),
        Task(
            name=f"train_candidate_a_seed{seed}",
            done_file=train,
            command=_cmd(
                sys.executable,
                "-B",
                "scripts/train_shortcut_suppressed_student.py",
                "--source-manifest",
                SOURCE_MANIFEST,
                "--root",
                DATA_ROOT,
                "--out",
                f"runs/candidate_a_shortcut_suppressed_seed{seed}",
                "--device",
                "auto",
                "--teacher-kind",
                "panns_cnn14_16k",
                "--teacher-checkpoint",
                TEACHER_CKPT,
                "--teacher-repo",
                PANNS_REPO,
                "--init-teacher-checkpoint",
                f"runs/stage1_panns_response_seed{seed}/checkpoints/teacher_best.pt",
                "--epochs",
                "8",
                "--batch-size",
                "16",
                "--seed",
                str(seed),
                "--kd-temperature",
                "2.0",
                "--kd-response-weight",
                "0.7",
                "--sample-weights",
                f"runs/shortcut_weights/candidate_a_seed{seed}/shortcut_weights.csv",
            ),
        ),
    ]


def _candidate_b_task(seed: int) -> Task:
    return Task(
        name=f"train_candidate_b_seed{seed}",
        done_file=ROOT / f"runs/candidate_b_disagreement_gated_seed{seed}/checkpoints/student_best.pt",
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/train_disagreement_gated_student.py",
            "--source-manifest",
            SOURCE_MANIFEST,
            "--root",
            DATA_ROOT,
            "--out",
            f"runs/candidate_b_disagreement_gated_seed{seed}",
            "--device",
            "auto",
            "--epochs",
            "8",
            "--batch-size",
            "16",
            "--seed",
            str(seed),
            "--teacher-kind",
            "panns_cnn14_16k",
            "--teacher-checkpoint",
            TEACHER_CKPT,
            "--teacher-repo",
            PANNS_REPO,
            "--init-teacher-checkpoint",
            f"runs/stage1_panns_response_seed{seed}/checkpoints/teacher_best.pt",
        ),
    )


def _candidate_c_task(seed: int) -> Task:
    return Task(
        name=f"train_candidate_c_seed{seed}",
        done_file=ROOT / f"runs/candidate_c_probe_adv_seed{seed}/student_domain_adv.pt",
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/train_domain_adversarial_student.py",
            "--source-manifest",
            SOURCE_MANIFEST,
            "--target-manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--init-checkpoint",
            f"runs/stage1_panns_response_seed{seed}/checkpoints/student_best.pt",
            "--out",
            f"runs/candidate_c_probe_adv_seed{seed}",
            "--device",
            "auto",
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--lr",
            "1e-4",
            "--domain-weight",
            "0.2",
            "--grl-weight",
            "1.0",
            "--max-target-records",
            "3000",
        ),
    )


def _candidate_probe_task(method: MethodSpec, seed: int) -> Task:
    out = _probe_dir(method, seed)
    assert out is not None
    checkpoint = _checkpoint(method, seed)
    assert method.probe_model is not None
    return Task(
        name=f"probe_{method.name}_seed{seed}",
        done_file=out / "domain_probe_results.csv",
        command=_cmd(
            sys.executable,
            "-B",
            "scripts/domain_probe_students.py",
            "--coswara-manifest",
            SOURCE_MANIFEST,
            "--coughvid-manifest",
            TARGET_MANIFEST,
            "--root",
            DATA_ROOT,
            "--run-dir",
            f"runs/stage1_panns_response_seed{seed}",
            "--out",
            out,
            "--samples-per-dataset",
            "1500",
            "--seed",
            str(seed),
            "--batch-size",
            "16",
            "--device",
            "auto",
            "--extra-checkpoint",
            f"{method.probe_model}={checkpoint}",
        ),
    )


def _tasks_for_stage(stage: str) -> list[Task]:
    tasks: list[Task] = []
    if stage == "baseline":
        for seed in SEEDS:
            tasks.append(_stage1_task(seed))
            tasks.append(_external_task(METHODS[0], seed))
            tasks.append(_external_task(METHODS[1], seed))
            tasks.append(_baseline_probe_task(seed))
    elif stage == "current":
        current_methods = [m for m in METHODS if m.stage == "current"]
        for seed in SEEDS:
            for method in current_methods:
                if seed != 7:
                    tasks.append(_stage3_train_task(method, seed))
                tasks.append(_external_task(method, seed))
            if seed != 7:
                tasks.append(_stage3_probe_task(seed))
    elif stage == "candidate_a":
        method = _method("candidate_a")
        for seed in SEEDS:
            tasks.extend(_candidate_a_tasks(seed))
            tasks.append(_external_task(method, seed))
            tasks.append(_candidate_probe_task(method, seed))
    elif stage == "candidate_b":
        method = _method("candidate_b")
        for seed in SEEDS:
            tasks.append(_candidate_b_task(seed))
            tasks.append(_external_task(method, seed))
            tasks.append(_candidate_probe_task(method, seed))
    elif stage == "candidate_c":
        method = _method("candidate_c")
        for seed in SEEDS:
            tasks.append(_candidate_c_task(seed))
            tasks.append(_external_task(method, seed))
            tasks.append(_candidate_probe_task(method, seed))
    else:
        raise ValueError(f"unknown stage: {stage}")
    return tasks


def _method(name: str) -> MethodSpec:
    for method in METHODS:
        if method.name == name:
            return method
    raise KeyError(name)


def _run_task(task: Task, args: argparse.Namespace) -> str:
    if _completed(task.done_file) and not args.force:
        print(f"[skip] {task.name}: {task.done_file}", flush=True)
        return "skipped"
    print(f"[run] {task.name}", flush=True)
    print(" ".join(str(item) for item in task.command), flush=True)
    if args.smoke or args.dry_run:
        return "planned"
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(task.command, cwd=ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"task failed ({result.returncode}): {task.name}")
    return "ran"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metric(path: Path, key: str) -> float | None:
    data = _load_json(path)
    value = data.get(key) if isinstance(data, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def _stage1_metric(seed: int, model: str, key: str) -> float | None:
    data = _load_json(ROOT / f"runs/stage1_panns_response_seed{seed}/metrics.json")
    if not data:
        return None
    block = {"teacher": "teacher_test", "ce": "ce_student_test", "kd": "student_test"}[model]
    value = data.get(block, {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _probe_metric(method: MethodSpec, seed: int, target: str) -> float | None:
    if method.probe_model is None:
        return None
    probe_dir = _probe_dir(method, seed)
    if probe_dir is None:
        return None
    csv_path = probe_dir / "domain_probe_results.csv"
    if not csv_path.is_file():
        return None
    import csv

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == method.probe_model and row.get("target") == target:
                try:
                    return float(row["auc"])
                except Exception:
                    return None
    return None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def _collect_method(method: MethodSpec) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        external = _external_dir(method, seed) / "metrics.json"
        rows.append(
            {
                "seed": seed,
                "external_macro_auroc": _metric(external, "macro_ovr_auroc"),
                "external_covid_auroc": _metric(external, "covid_positive"),
                "external_macro_auprc": _metric(external, "macro_ovr_auprc"),
                "probe_domain_auc": _probe_metric(method, seed, "dataset_domain"),
                "probe_task_auc": _probe_metric(method, seed, "task_label"),
                "source_macro_auroc": _stage1_metric(seed, method.name, "macro_ovr_auroc") if method.name in {"ce", "kd"} else None,
            }
        )
    summary: dict[str, Any] = {"method": method.name, "display": method.display, "kind": method.kind, "stage": method.stage, "seeds": rows}
    for key in ("external_macro_auroc", "external_covid_auroc", "external_macro_auprc", "probe_domain_auc", "probe_task_auc", "source_macro_auroc"):
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        mean, std = _mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_n"] = len(values)
    return summary


def _baseline_reference(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        item
        for item in summaries
        if item["method"] in {"ce", "kd", "source_only"} and isinstance(item.get("external_macro_auroc_mean"), (int, float))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["external_macro_auroc_mean"]))


def _passes_gate(method: dict[str, Any], baseline: dict[str, Any] | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if baseline is None:
        return False, ["baseline reference missing"]
    macro = method.get("external_macro_auroc_mean")
    covid = method.get("external_covid_auroc_mean")
    base_macro = baseline.get("external_macro_auroc_mean")
    base_covid = baseline.get("external_covid_auroc_mean")
    if not all(isinstance(x, (int, float)) for x in (macro, covid, base_macro, base_covid)):
        return False, ["external metrics incomplete"]
    macro_delta = float(macro) - float(base_macro)
    covid_delta = float(covid) - float(base_covid)
    external_ok = macro_delta >= 0.003 or covid_delta >= 0.005
    reasons.append(f"external delta macro={macro_delta:+.6f}, covid={covid_delta:+.6f}")

    positive_seeds = 0
    for row, base_row in zip(method.get("seeds", []), baseline.get("seeds", [])):
        value = row.get("external_macro_auroc")
        base = base_row.get("external_macro_auroc")
        if isinstance(value, (int, float)) and isinstance(base, (int, float)) and float(value) > float(base):
            positive_seeds += 1
    stability_ok = positive_seeds >= 2 and method.get("external_macro_auroc_n", 0) >= 3
    reasons.append(f"positive seeds={positive_seeds}/3")

    domain = method.get("probe_domain_auc_mean")
    task = method.get("probe_task_auc_mean")
    base_domain = baseline.get("probe_domain_auc_mean")
    base_task = baseline.get("probe_task_auc_mean")
    probe_ok = all(isinstance(x, (int, float)) for x in (domain, task, base_domain, base_task)) and float(domain) <= float(base_domain) and float(task) >= float(base_task)
    if all(isinstance(x, (int, float)) for x in (domain, task, base_domain, base_task)):
        reasons.append(f"probe delta domain={float(domain)-float(base_domain):+.6f}, task={float(task)-float(base_task):+.6f}")
    else:
        reasons.append("probe metrics incomplete")
    return bool(external_ok and stability_ok and probe_ok), reasons


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "n/a"


def _write_report(path: Path, summaries: list[dict[str, Any]], statuses: dict[str, str], verdicts: dict[str, dict[str, Any]]) -> None:
    baseline = _baseline_reference(summaries)
    passed = [name for name, verdict in verdicts.items() if verdict.get("passed")]
    final = "METHOD_GATE_PASSED" if passed else "PIVOT_TO_ANALYSIS_PAPER"
    lines = [
        "# 2026-05-29 Autonomous Innovation Loop",
        "",
        "## Verdict",
        "",
        f"- Final decision: `{final}`",
        f"- Baseline reference: `{baseline['method'] if baseline else 'missing'}`",
        f"- Passing candidates: `{passed}`",
        "",
        "## Task Status",
        "",
        "| Task | Status |",
        "|---|---|",
    ]
    for name, status in statuses.items():
        lines.append(f"| `{name}` | `{status}` |")
    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| Method | Kind | Ext macro AUROC | Ext COVID AUROC | Ext macro AUPRC | Domain probe AUC | Task probe AUC | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["method"],
                    item["kind"],
                    f"{_fmt(item.get('external_macro_auroc_mean'))} +/- {_fmt(item.get('external_macro_auroc_std'))}",
                    f"{_fmt(item.get('external_covid_auroc_mean'))} +/- {_fmt(item.get('external_covid_auroc_std'))}",
                    _fmt(item.get("external_macro_auprc_mean")),
                    _fmt(item.get("probe_domain_auc_mean")),
                    _fmt(item.get("probe_task_auc_mean")),
                    str(item.get("external_macro_auroc_n", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Innovation Gate", ""])
    for name, verdict in verdicts.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Passed: `{verdict.get('passed')}`")
        for reason in verdict.get("reasons", []):
            lines.append(f"- {reason}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- If no candidate passes, the controlled conclusion is that current KD variants do not yet provide enough external-generalization and representation evidence for a strong AAAI method claim.",
            "- In that case the safer paper route is an analysis-oriented study on when KD helps or fails for ultra-light cough audio models under dataset shift.",
            "- If a candidate passes, the next stage should expand to DiCOVA, calibration, latency, and teacher/student efficiency tables rather than adding more arbitrary combinations.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "runs/innovation_loop_summary.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "summaries": summaries, "statuses": statuses, "verdicts": verdicts}, indent=2),
        encoding="utf-8",
    )


def _run_stage(stage: str, args: argparse.Namespace, statuses: dict[str, str]) -> None:
    print(f"\n===== stage: {stage} =====", flush=True)
    for task in _tasks_for_stage(stage):
        statuses[task.name] = _run_task(task, args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Validate existing state and print planned missing tasks without launching long training.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even if their done file exists.")
    parser.add_argument("--stop-after-current", action="store_true", help="Only run baseline and current-method audit.")
    args = parser.parse_args()

    statuses: dict[str, str] = {}
    for stage in ("baseline", "current"):
        _run_stage(stage, args, statuses)

    summaries = [_collect_method(method) for method in METHODS]
    baseline = _baseline_reference(summaries)
    verdicts: dict[str, dict[str, Any]] = {}
    for name in ("tcd_very_strong", "tcd_conf035"):
        summary = next(item for item in summaries if item["method"] == name)
        passed, reasons = _passes_gate(summary, baseline)
        verdicts[name] = {"passed": passed, "reasons": reasons}

    if not args.stop_after_current and not any(v["passed"] for v in verdicts.values()):
        for stage in ("candidate_a", "candidate_b", "candidate_c"):
            _run_stage(stage, args, statuses)
            summaries = [_collect_method(method) for method in METHODS]
            baseline = _baseline_reference(summaries)
            method_name = stage
            summary = next(item for item in summaries if item["method"] == method_name)
            passed, reasons = _passes_gate(summary, baseline)
            verdicts[method_name] = {"passed": passed, "reasons": reasons}
            if passed:
                print(f"[gate] {method_name} passed. Stop method search.", flush=True)
                break

    summaries = [_collect_method(method) for method in METHODS]
    report_path = ROOT / "docs/progress/5.29_autonomous_innovation_loop.md"
    _write_report(report_path, summaries, statuses, verdicts)
    print(f"\n[report] {report_path}", flush=True)


if __name__ == "__main__":
    main()
