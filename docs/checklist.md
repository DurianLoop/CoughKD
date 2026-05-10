# Project Checklist

## Completed

- [x] Python package scaffold under `src/coughkd`.
- [x] LaTeX paper scaffold under `paper/` with sections, tables, bibliography, and claim-safety notes.
- [x] Reproducible environment files: `pyproject.toml`, `requirements.txt`, `requirements-ml.txt`, `environment.yml`.
- [x] Validation script: `scripts/validate_project.sh`.
- [x] Dataset manifest schema with `recording_id`, `subject_id`, `dataset`, `path`, `label`, `split`, and optional metadata.
- [x] Generic metadata/directory manifest builders.
- [x] Coswara-specific manifest builder for `cough-heavy` and `cough-shallow` WAV files.
- [x] Manifest validation for missing files, duplicate recordings, subject leakage, split distributions, label counts, duration stats, and quality warnings.
- [x] Manifest filtering by minimum duration and dropped labels.
- [x] Subject-disjoint split generation.
- [x] External-selection guard so external data is not used for checkpoint selection.
- [x] Synthetic WAV fixture generation for smoke tests.
- [x] Dependency-light WAV loading, mono conversion, linear resampling, invalid-recording checks, and log-mel-like frontend.
- [x] Segmentation utilities: sliding windows and hysteresis-style interval merging.
- [x] Aggregation utilities: mean, max, top-k, and quality-weighted top-k.
- [x] Augmentation smoke utilities: waveform noise, time shift, and SpecAugment-style masking.
- [x] Feature and prediction cache paths keyed by stable config hash.
- [x] Binary and multiclass metrics: AUROC, AUPRC, macro-F1, sensitivity, specificity, ECE, bootstrap AUROC CI, external AUROC drop.
- [x] Subgroup reporting with small-N suppression and clinical-claim guard.
- [x] Smoke teacher/student interfaces and dependency-light KD loss math.
- [x] Ablation grid generator with 3,456 planned combinations.
- [x] Resume-safe smoke grid runner and aggregate result writer.
- [x] Paper table generator requiring explicit completed run IDs.
- [x] Baseline smoke outputs for the planned classical/neural baseline schema.
- [x] Optional PyTorch manifest dataset, padded collate, compact convolutional teacher, depthwise student, one-step KD smoke path.
- [x] Multi-epoch PyTorch training loop with validation-based checkpointing, test inference, checkpoints, prediction CSVs, config JSON, metrics JSON, and markdown result sheet.
- [x] Real filtered Coswara cough manifest: 5,247 recordings from 2,635 subjects.
- [x] Real subject-disjoint Coswara split: train 3,671, validation 785, test 791 recordings.
- [x] Real closed-loop Coswara engineering baseline on CUDA.
- [x] Real 2+2 epoch CUDA validation against `../datasets/coswara`: teacher macro OVR AUROC 0.566054, student macro OVR AUROC 0.571329.
- [x] Long-training readiness plan and ClickUp-style task list.
- [x] Pre-long-training check on real Coswara: `runs/prelong_check_real_coswara`.
- [x] Real Coswara testround with teacher, CE-only student, and KD student: `runs/testround_real_coswara_e1_control_v2`.

## Partially Complete

- [~] Paper contains a coherent protocol and target tables, plus a measured engineering baseline table. Most target tables still need replacement with audited real experiments.
- [x] PyTorch training works with compact and PANNs CNN14 16 kHz teacher paths.
- [~] Student training works for a MobileNet-style depthwise student, but EfficientNet, BC-ResNet, ECAPA-small, and CE-only comparison runs are not implemented as real PyTorch training families.
- [~] Metrics are implemented broadly in the foundation layer. PyTorch training now exports accuracy, macro-F1, macro one-vs-rest AUROC, macro one-vs-rest AUPRC, ECE, Brier score, loss, per-class metrics, and confusion matrices; bootstrap confidence intervals remain future work.
- [~] Benchmark/export smoke paths exist, but production ONNX/TorchScript/TFLite export and hardware latency benchmarking are not yet measured.
- [~] Visualization plan is documented, but generated figures are not yet produced from saved predictions and logs.
- [~] Compact-baseline long-duration training is supported by preflight and testround artifacts. Foundation-model long training remains gated by pretrained teacher integration and external validation availability.

## Not Yet Complete

- [~] Real pretrained teacher wrappers: PANNs CNN14 16 kHz is implemented and validated; BEATs, AST, PaSST, HTS-AT, and AudioMAE remain pending.
- [ ] Real CE-only student training runs for controlled KD-vs-CE comparisons.
- [~] PyTorch training now supports response KD, feature-projection KD, cosine embedding KD, and relation KD. Attention-transfer KD remains pending for teacher backbones that expose time-frequency attention maps.
- [ ] Five-seed or five-fold repeated experiments with confidence intervals.
- [ ] External validation on COUGHVID, Cambridge COVID-19 Sounds, DiCOVA, Virufy, or ICBHI.
- [ ] Label harmonization checks across multiple datasets.
- [ ] Real cough-event detector integration beyond fallback segmentation utilities.
- [ ] Recording-level inference over segmented long recordings using trained PyTorch models.
- [ ] Calibration curves, reliability diagrams, confusion matrices, ROC/PR plots, and subgroup plots from real predictions.
- [ ] Production deployment export, quantization, model-size measurement, and fixed-hardware latency table.
- [ ] Patient voiceprint/verification protocol with identity metrics such as Top-1, Top-5, EER, and minDCF.
- [ ] Clinical or medical claims. The current repository supports screening research only.
- [x] Environment reconciliation: active `coughkd` reports Python 3.14.4 and `environment.yml` declares Python 3.14.
- [x] Long-run preflight command that validates env, manifest, GPU, output path, and required artifacts.
- [~] PyTorch feature cache and epoch-time profiling for long training. Batch-load profiling is present in `prelong-check`; full feature cache remains future optimization.

## Next Recommended Milestones

1. Improve the PANNs KD recipe before adding more long runs: expose temperature, response weight, feature weight, embedding KD, relation KD, label smoothing, scheduler, and early stopping.
2. Run a PANNs KD hyperparameter sweep and require KD to beat CE-only by a meaningful margin, not just by +0.0025 AUROC.
3. Add 3-seed and then 5-seed orchestration with confidence intervals for the best PANNs recipe.
4. Implement AST wrapper next because its checkpoint now loads offline; then implement BEATs as the second independent teacher.
5. Add label-schema experiments: current 5-class, healthy-vs-nonhealthy, covid-vs-noncovid, and 3-class healthy/covid/other.
6. Add crop-policy and recording-type experiments: cough-heavy only, cough-shallow only, both; first/center/random/multi-crop.
7. Add student family support through `--student-kind`, starting with depthwise width sweep, then BC-ResNet-small or MobileNetV3-small.
8. Generate paper-grade plots from real artifacts: training curves, confusion matrix, ROC/PR, calibration, Pareto efficiency.
9. Add deployment exports and latency benchmarks for the best student.
10. Replace paper target tables only from audited run families.

## Top-Conference TODO

### P0 Method Work

- [x] Add CLI/config fields for KD temperature: `1, 2, 4, 8`.
- [x] Add CLI/config fields for response KD weight: `0.3, 0.5, 0.7, 1.0`.
- [x] Add CLI/config fields for feature KD weight: `0.0, 0.05, 0.1, 0.2`.
- [x] Add label smoothing: `0.0, 0.05, 0.1`.
- [x] Add feature projection KD from student feature dimension to teacher feature dimension.
- [x] Add cosine embedding KD on normalized teacher/student embeddings.
- [x] Add relation KD using pairwise cosine-similarity matrices.
- [x] Save all KD loss components per epoch in `events.jsonl`.

### P0 Data Work

- [ ] Add label-schema switch for 5-class, binary healthy, binary covid, and 3-class tasks.
- [ ] Add label-map audit artifact per run.
- [ ] Add recording-type subset switch for heavy/shallow/both.
- [ ] Add crop policy: first, center, random train crop, multi-crop inference.
- [ ] Add metadata split audit for age, sex, country, symptoms, and recording type.

### P0 Evaluation Work

- [x] Add PyTorch AUPRC.
- [x] Add PyTorch ECE and Brier score.
- [ ] Add bootstrap confidence intervals.
- [ ] Add run-family audit writer.
- [ ] Generate paper figures from run artifacts.

### P1 Teacher/Student Work

- [ ] Implement AST teacher wrapper.
- [ ] Implement BEATs teacher wrapper.
- [ ] Add `--student-kind`.
- [ ] Add depthwise width sweep.
- [ ] Add BC-ResNet-small or MobileNetV3-small.

### P1 Paper Work

- [~] Update method section after final KD recipe is selected. The current method section documents the expanded trainable KD objective, but final weights still require sweep evidence.
- [ ] Add failure analysis if KD gains remain small.
- [ ] Add limitations with explicit no-clinical-claim language.
- [ ] Replace target tables only from audited runs.
