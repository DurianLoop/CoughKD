# Top-Conference Optimization Plan

This plan turns the current CoughKD repository from a working engineering pipeline into a paper-ready experimental system. The target is a credible AAAI / ICASSP / INTERSPEECH-style submission, not a demo. The current evidence is a single-seed Coswara long run with a real PANNs CNN14 16 kHz teacher. The next phase must improve method validity, evaluation depth, and reproducibility.

## Current Position

| Item | Current evidence |
|---|---|
| Dataset | Filtered Coswara cough split, subject-disjoint |
| Train / val / test | 3,671 / 785 / 791 recordings |
| Foundation teacher implemented | PANNs CNN14 16 kHz |
| PANNs teacher test macro OVR AUROC | 0.613420 |
| CE-only student test macro OVR AUROC | 0.563910 |
| KD student test macro OVR AUROC | 0.566423 |
| KD gain over CE-only | +0.002513 AUROC |
| Student / teacher AUROC retention | 92.34% |
| Paper target retention | >=95% |
| Gap to 95% retention | +0.016326 AUROC needed |

Interpretation: the training and inference pipeline is credible, but the method contribution is not yet strong enough. The main gap is not efficiency; the student is already extremely small. The gap is teacher quality, KD signal quality, robustness, and statistical evidence.

## Selected Optimization Direction

Primary direction:

> Improve PANNs-based distillation until the KD student reliably beats the CE-only student and reaches at least 95% teacher AUROC retention, then replicate the same recipe with AST and BEATs.

Why this direction:

- PANNs is already integrated and has completed one long run, so iteration speed is highest.
- The teacher AUROC is higher than compact baselines, proving pretrained audio representations help.
- KD gain is currently too small; fixing this is more important than adding many unvalidated teachers.
- Once KD is stronger on PANNs, AST/BEATs become confirmatory teacher-choice experiments rather than uncontrolled exploration.

## Paper-Level Success Criteria

Minimum credible submission criteria:

| Claim | Required evidence |
|---|---|
| Foundation teacher helps | PANNs/AST/BEATs teacher beats compact teacher on same split and metrics |
| KD helps student | KD student beats CE-only student by meaningful AUROC/F1 margin across repeated seeds |
| Compact model is efficient | Parameter, model-size, and latency reduction table |
| Results are stable | At least 3 seeds for core table, ideally 5 |
| No leakage | Subject-disjoint split audit and manifest hash in every run |
| Generalization is not overclaimed | External dataset claims only after real external manifests exist |
| Clinical claims avoided | All wording uses screening/research language, not diagnosis/medical utility |

Recommended numeric thresholds for the core Coswara table:

| Metric | Target |
|---|---:|
| KD vs CE-only AUROC gain | >= +0.01 absolute |
| Student / teacher AUROC retention | >= 95% |
| KD macro-F1 gain over CE-only | >= +0.01 absolute |
| Student parameter reduction vs foundation teacher | >= 10x |
| Seed count | >= 3 immediately, 5 for final paper |

## Optimization Modules

### Module A: Data Processing and Label Robustness

Goal: make the input signal cleaner and the labels more defensible.

Tasks:

- Add duration buckets: 0.5-1s, 1-2s, 2-4s, 4s+.
- Add train-time random crop for clips longer than `max_duration_sec`.
- Add validation/test deterministic center crop or full-window aggregation.
- Add optional multi-crop inference: 3 crops and 5 crops.
- Add per-recording metadata audit for age, sex, country, symptoms, and recording type.
- Add split-level label and metadata imbalance tables.
- Compare `cough-heavy` only, `cough-shallow` only, and both.
- Compare label sets:
  - 5-class current labels.
  - Binary healthy vs non-healthy.
  - Binary covid_positive vs non-covid.
  - 3-class healthy / covid_positive / other.

Hyperparameters to expose:

| Parameter | Values |
|---|---|
| `min_duration_sec` | 0.5, 1.0 |
| `max_duration_sec` | 2.0, 4.0, 6.0 |
| crop policy | first, center, random_train_center_eval, multi_crop |
| recording types | heavy, shallow, heavy+shallow |
| label schema | 5-class, healthy_binary, covid_binary, 3-class |

Acceptance:

- Every label schema produces a manifest, split report, and label map.
- No subject leakage across schemas.
- Paper uses the most scientifically defensible primary label schema, not just the highest score.

### Module B: Teacher Integration

Goal: turn downloaded checkpoints into real training teachers one at a time.

Current status:

- PANNs CNN14 16 kHz: implemented and long-run tested.
- AST: checkpoint loads offline after `torchvision+cu128` fix, wrapper pending.
- BEATs: source and checkpoint present, wrapper pending.
- HTS-AT: checkpoints and source present, wrapper pending.
- PaSST: `hear21passt` imports, wrapper pending.

Priority:

1. AST wrapper.
2. BEATs wrapper.
3. HTS-AT wrapper.
4. PaSST wrapper.

Wrapper acceptance:

- Loads local checkpoint offline.
- Runs one batch on CUDA.
- Exposes `logits`, `features`, `embedding`.
- Records checkpoint path, source repo, parameter count, and preprocessing config.
- Passes `torch-train --max-records-per-split 8` smoke.
- Passes one short full-split run before any 30-epoch run.

### Module C: Student Model Family

Goal: avoid overfitting claims to one weak student architecture.

Current student:

- `DepthwiseStudent`, 20,717 parameters.

Next students:

| Student | Reason | Initial settings |
|---|---|---|
| DepthwiseStudent width sweep | fastest, isolates capacity | width 16, 24, 32, 48 |
| BC-ResNet-small | strong audio edge baseline | channels 16/32, depth small |
| MobileNetV3-small | standard deployment baseline | width multiplier 0.35, 0.5, 0.75 |
| ECAPA-small | speaker/respiratory embedding baseline | channels 128, 192 |

Student acceptance:

- Same dataloader, split, metrics, and prediction export.
- CE-only and KD variants for every student.
- Parameter count and latency measured for every student.

### Module D: KD Loss Optimization

Goal: make KD visibly better than CE-only.

Current loss:

- CE + response KD + feature-projection MSE + cosine embedding KD + relation KD.
- Temperature, response weight, feature weight, embedding weight, relation weight, and label smoothing are exposed through `torch-train` CLI arguments and saved in `config.json`.
- Per-epoch KD components are saved in `events.jsonl`, which makes short sweeps auditable before expensive PANNs long runs.

Immediate sweep:

| Hyperparameter | Values |
|---|---|
| temperature | 1, 2, 4, 8 |
| response_weight | 0.3, 0.5, 0.7, 1.0 |
| feature_weight | 0.0, 0.05, 0.1, 0.2 |
| CE weight | 1.0, 0.7 |
| label smoothing | 0.0, 0.05, 0.1 |

New KD losses:

- Feature projection before MSE: student feature -> teacher feature dimension. Implemented for PyTorch training when feature dimensions differ.
- Cosine embedding KD on normalized embeddings. Implemented for PyTorch training.
- Attention transfer on time-frequency maps where teacher supports it.
- Relation KD: pairwise cosine similarity matrix within batch. Implemented for PyTorch training.
- Logit calibration KD: temperature-tuned teacher probabilities.

Acceptance:

- At least one KD recipe beats CE-only by >= +0.01 AUROC across 3 seeds.
- Ablation shows which KD component matters.
- No KD result is reported without CE-only counterpart.

### Module E: Training Optimization

Goal: stabilize training and improve generalization.

Training hyperparameters:

| Hyperparameter | Values |
|---|---|
| learning rate | 3e-4, 1e-3, 3e-3 |
| weight decay | 1e-5, 1e-4, 1e-3 |
| batch size | 16, 32, 64 |
| scheduler | none, cosine, cosine+warmup |
| warmup epochs | 0, 3, 5 |
| epochs | 30, 50, 80 |
| early stopping patience | 8, 12 |
| class weights | off, inverse frequency, sqrt inverse |
| optimizer | AdamW, SGD momentum for student only |

Augmentation:

| Augmentation | Values |
|---|---|
| waveform noise | 0, 0.005, 0.01 |
| time shift | 0, 5%, 10% |
| SpecAugment time mask | 0, 16, 32 |
| SpecAugment freq mask | 0, 4, 8 |
| mixup alpha | 0, 0.2, 0.4 |

Acceptance:

- Validation curves are saved and inspected for collapse.
- Best checkpoint is selected by validation macro OVR AUROC.
- Final table reports test metrics only once per selected config.

### Module F: Evaluation and Visualization

Goal: generate paper-grade evidence, not just metrics JSON.

Metrics to add to PyTorch reports:

- Macro AUROC.
- Macro AUPRC. Implemented as macro one-vs-rest AUPRC in the PyTorch report.
- Per-class AUROC.
- Per-class AUPRC.
- Macro-F1.
- Balanced accuracy.
- Sensitivity/specificity for binary schemas.
- ECE and Brier score. Implemented for multiclass PyTorch reports.
- Bootstrap 95% CI.

Figures:

- Class distribution by split.
- Metadata distribution by split.
- Training/validation loss curves.
- Validation AUROC curves.
- Confusion matrix.
- One-vs-rest ROC curves.
- One-vs-rest PR curves.
- Calibration/reliability diagram.
- Teacher vs CE vs KD bar chart.
- Accuracy/AUROC vs parameter/latency Pareto plot.

Generated figure gallery:

- `python scripts/generate_doc_figures.py` generates the current documentation figure set.
- `docs/figures/README.md` indexes 39 SVG figures generated from saved manifests, metrics, events, prediction CSVs, and explicitly labelled planning thresholds.
- Current generated figures cover split/label/filter audits, measured run comparisons, PANNs teacher/CE/KD bars, retention gap, parameter compression, validation and training curves, confusion matrices, one-vs-rest ROC/PR curves for key classes, reliability diagrams, P0 progress, and success-criteria progress.
- These figures remain single-seed or artifact-audit visuals unless their captions explicitly say repeated-seed evidence exists.

Acceptance:

- Every figure is generated from saved artifacts, not manual values.
- Every figure has a matching script/command.
- Captions say whether the result is single-seed or repeated-seed.

### Module G: Deployment and Efficiency

Goal: support the edge-deployment contribution.

Tasks:

- Export student to TorchScript.
- Export student to ONNX.
- Measure model size in MB.
- Measure CPU latency with fixed batch size 1 and fixed audio duration.
- Measure GPU latency for reference.
- Optional: INT8 dynamic/static quantization.

Latency protocol:

| Setting | Value |
|---|---|
| Audio duration | 4 seconds |
| Warmup iterations | 50 |
| Timed iterations | 200 |
| Batch size | 1 |
| Report | mean, p50, p95 |

Acceptance:

- Efficiency table includes params, model size, latency, and AUROC.
- Latency script records hardware and PyTorch version.

### Module H: Code Quality and Reproducibility

Goal: make experiments auditable and maintainable.

Tasks:

- Add config dataclasses for training hyperparameters.
- Add experiment runner that expands YAML/JSON grids.
- Add run registry with run ID, seed, teacher, student, KD recipe, manifest hash.
- Add `RESULTS_AUDIT.md` per experiment family.
- Add tests for teacher wrapper construction and batch forward.
- Add tests for label schemas and no-leak split generation.
- Keep `runs/`, `pretrained/`, and `external/teacher_repos/` out of git.

Acceptance:

- One command can reproduce every table row.
- CI/local tests pass without requiring large checkpoints.
- Checkpoint-dependent tests are skipped cleanly when files are absent.

### Module I: Paper and Claim Calibration

Goal: align claims with evidence.

Tasks:

- Convert target tables into measured tables only after audited runs.
- Add a method section describing the final KD recipe actually used.
- Add a failure analysis section if KD gains remain small.
- Add limitations: single dataset, label noise, no prospective validation, no diagnosis claim.
- Add ablation table for KD components.
- Add teacher choice table.
- Add student choice table.
- Add deployment Pareto table.

Acceptance:

- Claim ledger has no unsupported performance claims.
- Abstract does not imply clinical diagnosis.
- Paper distinguishes compact engineering baseline from PANNs/AST/BEATs foundation results.

## Execution Order

### Sprint 1: Make PANNs Result Stronger

1. Add KD hyperparameter CLI options.
2. Add feature projection and cosine embedding KD.
3. Run PANNs short sweeps with `max_records_per_split` for sanity.
4. Run 3-seed PANNs full experiments for top 2 KD recipes.
5. Generate result audit and plots.

### Sprint 2: Add Second Teacher

1. Implement AST wrapper.
2. Run AST wrapper smoke.
3. Run AST short full-split experiment.
4. Compare AST teacher vs PANNs teacher.
5. If AST teacher is stronger, run AST KD student.

### Sprint 3: Student and Efficiency

1. Add width sweep for `DepthwiseStudent`.
2. Add BC-ResNet-small or MobileNetV3-small.
3. Run CE/KD comparisons for best PANNs or AST teacher.
4. Export best student and measure latency.

### Sprint 4: Paper-Ready Evidence

1. Run 5 seeds for final selected method.
2. Generate all plots and tables from audited artifacts.
3. Update paper sections.
4. Run claim review.
5. Freeze submission result set.

## Immediate Next Commands To Implement

The KD arguments below are implemented; `--student-kind`, scheduler, early stopping, and grid orchestration are still future work:

```bash
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli torch-train \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/panns_kd_sweep_seed7_t4_rw07_fw005_ew01_rs005 \
  --teacher-kind panns_cnn14_16k \
  --teacher-epochs 30 \
  --student-epochs 30 \
  --kd-temperature 4 \
  --kd-response-weight 0.7 \
  --kd-feature-weight 0.05 \
  --kd-embedding-weight 0.1 \
  --kd-relation-weight 0.05 \
  --label-smoothing 0.05 \
  --batch-size 16 \
  --num-workers 4 \
  --max-duration-sec 4.0
```

```bash
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli run-grid \
  --grid configs/grids/panns_kd_sweep.yaml \
  --out runs/grids/panns_kd_sweep_v1
```

```bash
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli make-figures \
  --runs runs/long_coswara_panns_cnn14_16k_seed7_e30 \
  --out paper/figures/panns_seed7
```
