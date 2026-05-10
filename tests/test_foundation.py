import tempfile
import unittest
from shutil import copyfile
from pathlib import Path

from coughkd.datasets import build_coswara_manifest, build_manifest_from_metadata, dataset_smoke_report, normalize_label, normalize_split
from coughkd.manifest import (
    check_external_selection_guard,
    filter_manifest_rows,
    read_manifest,
    subject_disjoint_split,
    validate_manifest,
    write_manifest,
)
from coughkd.augment import add_uniform_noise, specaugment, time_shift
from coughkd.benchmark import benchmark_report, quantize_values
from coughkd.baselines import NearestCentroidBaseline, baseline_smoke_report
from coughkd.cache import feature_cache_path, prediction_cache_path, save_prediction_cache, stable_config_hash
from coughkd.config import RunConfig
from coughkd.grid import ablation_grid, aggregate_results, run_smoke_grid
from coughkd.losses import coughkd_loss, cross_entropy, kl_divergence, softmax
from coughkd.models import SmokeStudent, SmokeTeacher
from coughkd.paper_tables import generate_paper_tables
from coughkd.reporting import assert_no_unsupported_clinical_claims, subgroup_report
from coughkd.metrics import (
    average_precision,
    binary_classification_report,
    bootstrap_auc_ci,
    external_drop_report,
    multiclass_ovr_auroc,
    roc_auc,
)
from coughkd.segmentation import aggregate_scores, merge_intervals, sliding_windows
from coughkd.smoke import make_smoke_data
from coughkd.torch_models import require_torch, run_torch_smoke


class FoundationTests(unittest.TestCase):
    def test_metrics_are_deterministic(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)
        report = binary_classification_report(labels, scores)
        self.assertEqual(report["macro_f1"], 1.0)
        ci = bootstrap_auc_ci(labels, scores, n_bootstrap=20, seed=2)
        self.assertEqual(ci["mean"], 1.0)
        drop = external_drop_report(0.9, 0.8)
        self.assertAlmostEqual(drop["external_auroc_drop"], 0.1)
        multiclass = multiclass_ovr_auroc(
            ["a", "b", "c", "a"],
            [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6], [0.7, 0.2, 0.1]],
            ["a", "b", "c"],
        )
        self.assertEqual(multiclass["macro_ovr_auroc"], 1.0)

    def test_subject_split_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_smoke_data(root / "smoke")
            rows = read_manifest(manifest)
            split_rows = subject_disjoint_split(rows, seed=1)
            split_manifest = root / "split.csv"
            write_manifest(split_rows, split_manifest)
            issues, _ = validate_manifest(split_manifest, root)
            self.assertEqual([issue for issue in issues if issue.severity == "error"], [])

    def test_subject_leakage_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_smoke_data(root / "smoke")
            rows = read_manifest(manifest)
            rows[0]["split"] = "train"
            rows[1]["split"] = "test"
            leaked_manifest = root / "leaked.csv"
            write_manifest(rows, leaked_manifest)
            issues, _ = validate_manifest(leaked_manifest, root)
            messages = [issue.message for issue in issues]
            self.assertTrue(any("subject leakage" in message for message in messages))

    def test_manifest_report_contains_split_and_duration_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_smoke_data(root / "smoke")
            split_rows = subject_disjoint_split(read_manifest(manifest), seed=1)
            split_manifest = root / "split.csv"
            write_manifest(split_rows, split_manifest)
            issues, summary = validate_manifest(split_manifest, root)
            self.assertEqual([issue for issue in issues if issue.severity == "error"], [])
            self.assertIn("subjects_by_split", summary)
            self.assertIn("labels_by_split", summary)
            self.assertEqual(summary["duration_sec"]["count"], 12)
            self.assertIn("quality", summary)

    def test_selection_guard_rejects_external_selection(self):
        rows = [{"recording_id": "a", "split": "external"}]
        issues = check_external_selection_guard(rows, {"train", "val", "external"})
        self.assertEqual(len(issues), 1)

    def test_manifest_filter_drops_labels_and_short_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = make_smoke_data(root / "smoke")
            rows = read_manifest(manifest)
            kept, report = filter_manifest_rows(rows, root, min_duration_sec=1.0, drop_labels={"healthy"})
            self.assertEqual(len(kept), 0)
            self.assertEqual(report["dropped_records"], 12)
            self.assertEqual(report["drop_reasons"]["drop_label"], 6)
            self.assertEqual(report["drop_reasons"]["short_duration"], 6)

    def test_segmentation_and_aggregation(self):
        windows = sliding_windows(duration_sec=2.2, window_sec=1.0, hop_sec=0.5)
        self.assertEqual(len(windows), 4)
        merged = merge_intervals([(0.0, 0.3), (0.35, 0.7), (1.2, 1.5)], gap_sec=0.1)
        self.assertEqual(merged, [(0.0, 0.7), (1.2, 1.5)])
        scores = [0.2, 0.8, 0.6, 0.4]
        qualities = [0.1, 0.9, 0.8, 0.2]
        self.assertAlmostEqual(aggregate_scores(scores, method="mean"), 0.5)
        self.assertAlmostEqual(aggregate_scores(scores, method="max"), 0.8)
        self.assertAlmostEqual(aggregate_scores(scores, method="topk", top_k=2), 0.7)
        self.assertAlmostEqual(
            aggregate_scores(scores, qualities=qualities, method="quality_topk", top_k=2),
            (0.8 * 0.9 + 0.6 * 0.8) / 1.7,
        )

    def test_augmentations_and_cache_hash_are_deterministic(self):
        samples = [0.0, 0.1, -0.1, 0.2]
        self.assertEqual(time_shift(samples, 1), [0.2, 0.0, 0.1, -0.1])
        self.assertEqual(add_uniform_noise(samples, 0.01, seed=3), add_uniform_noise(samples, 0.01, seed=3))
        masked = specaugment([[1.0, 2.0], [3.0, 4.0]], time_mask=1, freq_mask=1, seed=2)
        self.assertEqual(len(masked), 2)
        config_a = RunConfig(seed=1)
        config_b = RunConfig(seed=2)
        self.assertNotEqual(stable_config_hash(config_a), stable_config_hash(config_b))
        self.assertIn(stable_config_hash(config_a), str(feature_cache_path(Path("out"), "rec/1", config_a)))
        self.assertIn(stable_config_hash(config_a), str(prediction_cache_path(Path("out"), "rec/1", "teacher/v1", config_a)))

    def test_smoke_models_and_kd_losses(self):
        features = [[0.1, 0.2], [0.3, 0.4]]
        teacher = SmokeTeacher()
        student = SmokeStudent()
        teacher_out = teacher.forward(features)
        student_out = student.forward(features)
        self.assertEqual(len(teacher_out.logits), 2)
        self.assertEqual(len(student_out.features), 4)
        self.assertAlmostEqual(sum(softmax([1.0, 2.0])), 1.0)
        self.assertGreater(cross_entropy(1, [0.1, 0.9]), 0.0)
        self.assertGreaterEqual(kl_divergence(teacher_out.logits, student_out.logits), 0.0)
        loss = coughkd_loss(
            1,
            teacher_out.logits,
            student_out.logits,
            teacher_out.features,
            student_out.features,
            teacher_out.attention,
            student_out.attention,
            [teacher_out.embedding, teacher_out.embedding],
            [student_out.embedding, student_out.embedding],
        )
        self.assertGreater(loss["total"], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = save_prediction_cache(
                Path(tmp),
                "rec",
                teacher.name,
                RunConfig(),
                teacher_out.logits,
                teacher_out.embedding,
            )
            self.assertTrue(cache_path.is_file())

    def test_ablation_grid_and_resume_safe_runner(self):
        grid = ablation_grid()
        self.assertGreater(len(grid), 20)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "grid"
            first = run_smoke_grid(out, limit=2)
            self.assertEqual([run["status"] for run in first["runs"]], ["completed", "completed"])
            second = run_smoke_grid(out, limit=2)
            self.assertEqual([run["status"] for run in second["runs"]], ["skipped", "skipped"])
            aggregate = aggregate_results(out, Path(tmp) / "aggregate")
            self.assertEqual(aggregate["num_runs"], 2)

    def test_benchmark_smoke_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = benchmark_report(SmokeStudent(), [[0.1, 0.2], [0.3, 0.4]], Path(tmp))
            self.assertGreater(report["parameter_count"], 0)
            self.assertIn("fp32", report["exports"])
            self.assertGreaterEqual(report["latency"]["repeats"], 1)
            self.assertTrue(all(isinstance(value, int) for value in quantize_values([0.1, -0.2], "int8")))

    def test_subgroup_reporting_and_claim_guard(self):
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        metadata = [{"sex": "f"}, {"sex": "f"}, {"sex": "m"}, {"sex": "m"}]
        report = subgroup_report(labels, scores, metadata, ["sex"], min_n=3)
        self.assertEqual(report["fields"]["sex"]["f"]["status"], "suppressed_small_n")
        assert_no_unsupported_clinical_claims("screening research support")
        with self.assertRaises(ValueError):
            assert_no_unsupported_clinical_claims("this model can diagnose patients")

    def test_paper_table_generation_requires_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp) / "runs"
            run_smoke_grid(runs_dir, limit=2)
            audit = generate_paper_tables(runs_dir, Path(tmp) / "tables", ["smoke_grid_000", "smoke_grid_001"])
            self.assertEqual(audit["num_rows"], 2)
            with self.assertRaises(ValueError):
                generate_paper_tables(runs_dir, Path(tmp) / "bad_tables", ["missing_run"])

    def test_classical_baseline_smoke(self):
        model = NearestCentroidBaseline()
        model.fit([[0.0], [0.1], [1.0], [0.9]], [0, 0, 1, 1])
        scores = model.predict_scores([[0.05], [0.95]])
        self.assertLess(scores[0], scores[1])
        with tempfile.TemporaryDirectory() as tmp:
            report = baseline_smoke_report(Path(tmp))
            self.assertEqual(len(report["baselines"]), 4)

    def test_dataset_metadata_adapter(self):
        self.assertEqual(normalize_label("COVID-19"), "covid_positive")
        self.assertEqual(normalize_label("positive_mild"), "covid_positive")
        self.assertEqual(normalize_split("validation"), "val")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = make_smoke_data(root / "source")
            source_rows = read_manifest(source_manifest)
            metadata = root / "metadata.csv"
            metadata.write_text(
                "audio_path,participant_id,diagnosis,split\n"
                f"{source_rows[0]['path']},{source_rows[0]['subject_id']},positive,train\n"
                f"{source_rows[1]['path']},{source_rows[1]['subject_id']},negative,train\n",
                encoding="utf-8",
            )
            report = build_manifest_from_metadata(
                root,
                metadata,
                root / "imported.csv",
                "adapter_test",
                path_column="audio_path",
                subject_column="participant_id",
                label_column="diagnosis",
                split_column="split",
            )
            self.assertEqual(report.num_records, 2)
            self.assertEqual(report.validation_errors, 0)
            imported = read_manifest(root / "imported.csv")
            self.assertEqual({row["label"] for row in imported}, {"covid_positive", "healthy"})
            smoke_report = dataset_smoke_report(root / "dataset_smoke")
            self.assertEqual(smoke_report["build"]["validation_errors"], 0)

    def test_coswara_manifest_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_manifest = make_smoke_data(root / "source")
            source_rows = read_manifest(source_manifest)
            subject = "subject_coswara"
            audio_dir = root / "Extracted_data" / "20200101" / subject
            audio_dir.mkdir(parents=True)
            copyfile(root / source_rows[0]["path"], audio_dir / "cough-heavy.wav")
            copyfile(root / source_rows[1]["path"], audio_dir / "cough-shallow.wav")
            metadata = root / "combined_data.csv"
            metadata.write_text(
                "id,a,covid_status,g,l_c,cough,fever\n"
                f"{subject},31,positive_mild,female,India,True,False\n",
                encoding="utf-8",
            )
            report = build_coswara_manifest(root, metadata, root / "coswara.csv")
            self.assertEqual(report.num_records, 2)
            self.assertEqual(report.validation_errors, 0)
            imported = read_manifest(root / "coswara.csv")
            self.assertEqual({row["label"] for row in imported}, {"covid_positive"})
            self.assertEqual({row["sex"] for row in imported}, {"female"})
            self.assertEqual({row["symptoms"] for row in imported}, {"cough"})

    def test_torch_smoke_when_available(self):
        try:
            require_torch()
        except RuntimeError:
            self.skipTest("PyTorch is not installed in this environment")
        with tempfile.TemporaryDirectory() as tmp:
            report = run_torch_smoke(Path(tmp), device="cpu", batch_size=4)
            self.assertGreater(report["teacher_params"], report["student_params"])
            self.assertGreater(report["initial_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
