# Long-Training Readiness Plan

This document defines the required operations before launching any large-scale, long-duration GPU training or inference run. It is intentionally strict: long runs should only begin after the pipeline can produce reproducible, auditable, and scientifically interpretable outputs.

## Current Verified State

| Area | Verified fact |
|---|---|
| Available cough dataset | `../datasets/coswara` |
| Missing requested path | `../data` does not exist in the current filesystem |
| Empty cough dataset directories | `../datasets/coughvid`, `../datasets/cambridge_covid19_sounds`, `../datasets/dicova` |
| Non-cough auxiliary datasets present | `../datasets/ESC50`, `../datasets/FSD50K` |
| Main environment | conda environment `coughkd` |
| Runtime Python observed | 3.14.4 |
| Declared Python in `environment.yml` | 3.14 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 5090 |
| Filtered Coswara manifest | 5,247 recordings / 2,635 subjects |
| Split | train 3,671 / validation 785 / test 791 |
| Real CUDA smoke | passed, batch shape `[8, 1, 398, 32]` |
| Real 2+2 epoch training | passed under `runs/real_coswara_torch_train_e2` |
| 2+2 teacher test macro OVR AUROC | 0.566054 |
| 2+2 student test macro OVR AUROC | 0.571329 |
| Pre-long-training check | passed under `runs/prelong_check_real_coswara` |
| CE-only/KD testround | passed under `runs/testround_real_coswara_e1_control_v2` |

## Why Long Training Is Not the Next Blind Step

The project has already called the GPU and completed real training. The reason not to start a multi-hour or multi-day run immediately is not lack of GPU access. The blockers are experimental validity and compute efficiency:

- The compact teacher is `ConvTeacher`; the first validated foundation teacher is `panns_cnn14_16k`.
- The current training path now includes a CE-only depthwise control, so compact KD-vs-CE comparisons can be run under identical splits.
- The current preprocessing path performs CPU-side WAV loading and Python feature extraction on every epoch, making long training inefficient.
- The current long-run output would still lack repeated seeds, confidence intervals, external validation, calibration plots, and deployment measurements.
- Foundation-model long training still needs a real pretrained teacher wrapper. Compact-baseline long training is supported after the completed preflight and testround.
- Most external cough datasets expected by the paper are currently unavailable or empty in this workspace.

Long training should begin only after the gates below are satisfied.

## Gate 0: Reproducibility Lock

Goal: make the runtime environment reproducible before spending long GPU time.

Required actions:

- Keep `environment.yml` and the observed `coughkd` Python version aligned.
- Record the exact `conda list`, `pip freeze`, CUDA, GPU, driver, and git hash for every training run.
- Ensure all training commands save `config.json`, `metrics.json`, `events.jsonl`, checkpoint metadata, and command line.
- Use Python 3.14 for the current `coughkd` environment unless a future compatibility issue requires rebuilding.

Acceptance criteria:

- `conda run -n coughkd python --version` matches the documented environment decision.
- `PYTHONPATH=src conda run -n coughkd python -m unittest discover -s tests` passes.
- A run environment snapshot is saved under each long-run output directory.

## Gate 1: Data Audit and Dataset Scope

Goal: prevent invalid comparisons and accidental leakage.

Required actions:

- Treat `../datasets/coswara` as the only currently available cough-screening dataset.
- Keep `../datasets/coughvid`, `../datasets/cambridge_covid19_sounds`, and `../datasets/dicova` marked unavailable until non-empty and manifest-compatible.
- Do not use ESC50/FSD50K as disease-screening evidence. They can only be used for auxiliary audio pretraining experiments if a separate protocol is written.
- Freeze the filtered Coswara manifest and subject-disjoint split used for long training.
- Keep `under_validation` dropped and the minimum duration threshold at 0.5 seconds unless a new protocol justifies changing them.

Acceptance criteria:

- Manifest validation reports no errors.
- Split report shows 5,247 recordings, 2,635 subjects, train 3,671, validation 785, test 791.
- Subject leakage checker passes.
- Any new dataset has its own manifest report, label mapping, license note, and split policy.

## Gate 2: Feature Pipeline Efficiency

Goal: avoid wasting GPU jobs on repeated CPU feature extraction.

Required actions:

- Add or enable a real feature cache for PyTorch training, keyed by preprocessing config and audio file identity.
- Save cache metadata: sample rate, max duration, mel settings, transform version, source manifest hash.
- Add a cache validation step that refuses stale cache entries when config or manifest changes.
- Measure epoch time before and after caching on the full Coswara training split.

Acceptance criteria:

- Cached and uncached feature shapes match on a deterministic sample.
- A full validation pass can run from cache.
- Epoch time is reduced enough to justify long runs, or the bottleneck is explicitly documented.

## Gate 3: Experimental Controls

Goal: make long-run results interpretable.

Required actions:

- Add CE-only training for the current depthwise student on the same split.
- Keep the compact `ConvTeacher` run as an engineering baseline, not a final paper claim.
- Add at least one controlled KD run with identical preprocessing, seed, batch size, and split.
- Save train/validation/test predictions for teacher, CE-only student, and KD student.

Acceptance criteria:

- The result table can compare teacher vs CE-only student vs KD student.
- KD improvement is reported as a delta relative to CE-only, not only relative to teacher.
- All metrics trace to run directories and saved prediction CSVs.

## Gate 4: Foundation Teacher Path

Goal: align the implementation with the paper's core claim.

Required actions:

- PANNs CNN14 16 kHz wrapper is implemented and has completed a full long run. Remaining wrappers should be added one at a time, with BEATs and AST as the next priorities.
- Verify forward pass, feature extraction, parameter count, and checkpoint saving on a small batch.
- Run one short full-split experiment before any new long foundation-model training.
- Keep compact-teacher and foundation-teacher results separated in tables.

Acceptance criteria:

- Teacher checkpoint metadata identifies the pretrained source.
- Teacher validation selection uses macro OVR AUROC, with ECE included when available.
- The paper does not describe compact-teacher results as foundation-model distillation.

## Gate 5: Metrics, Plots, and Audit Outputs

Goal: produce paper-ready evidence from every long run.

Required actions:

- Extend PyTorch reports beyond accuracy, macro-F1, macro OVR AUROC, and loss.
- Add AUPRC, ECE, per-class F1, per-class AUROC where valid, confusion matrix, and calibration output.
- Generate plots from saved predictions and event logs.
- Write a `RESULTS_AUDIT.md` for each experiment family.

Acceptance criteria:

- Every paper table row links to a run ID, manifest, seed, config, and prediction file.
- Plot captions specify whether the figure is a smoke fixture, compact engineering baseline, or foundation-model result.
- Unsupported clinical wording is absent.

## Gate 6: Long-Run Schedule

After the completed preflight and testround, compact-baseline long training can launch in this order:

1. Compact teacher baseline, 5 seeds, 10-30 epochs with early stopping.
2. Depthwise student CE-only baseline, same seeds and split.
3. Depthwise student KD baseline, same seeds and split.
4. Export/latency benchmark on fixed hardware.

Foundation-model long training remains gated:

5. Download teacher checkpoints with `bash scripts/download_teacher_models.sh`.
6. First foundation teacher short run, 1 seed, after a real pretrained teacher wrapper exists.
7. First foundation teacher long run, only if the short run is stable.
8. Foundation-teacher KD student run.
9. External validation, only after external manifests are real and frozen.

## Gate 7: Top-Conference Optimization Direction

Goal: prioritize experiments that can support a strong paper claim.

Selected direction:

- First optimize the validated PANNs CNN14 16 kHz path until KD provides a meaningful improvement over CE-only and reaches at least 95% teacher AUROC retention.
- Then add AST and BEATs wrappers to test whether the method generalizes across teachers.
- Do not launch additional 30-epoch teacher runs until the corresponding wrapper has passed a smoke run and a short full-split run.

Immediate optimization targets:

| Area | Target |
|---|---|
| KD vs CE-only AUROC delta | >= +0.01 |
| Student / teacher AUROC retention | >= 95% |
| Current required AUROC gain for retention | +0.016326 |
| Seed count before paper table | >= 3, ideally 5 |
| Teacher family count before final claim | PANNs + at least one of AST/BEATs |

Required implementation before next major long run:

- Expose KD hyperparameters in CLI/config.
- Add embedding projection KD and cosine embedding KD.
- Add label smoothing and scheduler options.
- Add run-family audit output.
- Add plots generated from prediction and event artifacts.

See `docs/top_conference_optimization_plan.md` for the full module-level roadmap.

## Initial Long-Run Command Template

Compact-baseline long-run command now supported by `runs/prelong_check_real_coswara` and `runs/testround_real_coswara_e1_control_v2`:

```bash
PYTHONPATH=src conda run -n coughkd python -u -m coughkd.cli torch-train \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/long_coswara_compact_seed7_e30 \
  --device cuda \
  --teacher-epochs 30 \
  --student-epochs 30 \
  --batch-size 64 \
  --num-workers 4 \
  --max-duration-sec 4.0 \
  --seed 7
```

Required before accepting this output:

- Confirm the command records the environment snapshot.
- Confirm CE-only and KD outputs are separate.
- Confirm validation metrics are used for checkpoint selection.
- Confirm no external test set is used during training.

## Stop Conditions

Stop long training and return to implementation if any of the following occurs:

- Validation AUROC is flat or unstable for several epochs while loss does not improve.
- Class predictions collapse to majority class.
- GPU utilization is low because CPU feature extraction dominates.
- Cached features are stale or do not match the manifest.
- Train/validation/test subject leakage is detected.
- Any report uses diagnosis or clinical-utility wording.

## Three-Review Requirement

Before launching long training, perform and record three reviews:

1. Data review: counts, paths, splits, labels, leakage, empty datasets.
2. Command review: environment, exact command, output directory, seed, checkpoint policy.
3. Claim review: paper language, README wording, target-vs-measured distinction, clinical caution.

## Documentation Review Record

This section records the three reviews performed when this readiness plan and ClickUp task list were added.

### Review 1: Data and Numbers

Status: passed after corrections.

Checked:

- Coswara filtered count is consistently documented as 5,247 recordings and 2,635 subjects.
- Split is consistently documented as train 3,671, validation 785, test 791.
- Fresh 2+2 run metrics are consistently documented as teacher macro OVR AUROC 0.566054 and student macro OVR AUROC 0.571329.
- Prior 5+5 compact baseline metrics remain documented separately as teacher macro OVR AUROC 0.577184 and student macro OVR AUROC 0.580959.
- `../datasets/coswara` is documented as the real available cough dataset; `../data` absence is documented.
- Environment is aligned: observed Python 3.14.4 and declared Python 3.14.
- Preflight passed under `runs/prelong_check_real_coswara`.
- CE-only/KD testround passed under `runs/testround_real_coswara_e1_control_v2`.

Correction made:

- Updated `environment.yml`, `STATUS.md`, `README.md`, and `docs/commands.md` so the Python version is aligned with the active environment.
- Added compact-baseline testround results and clarified that foundation-model long training remains gated.

### Review 2: Commands and Artifacts

Status: passed.

Checked:

- `docs/long_training_plan.md` and `docs/clickup_long_training.md` exist.
- Referenced artifacts exist: `runs/real_coswara_cough_filtered_split/manifest_split.csv`, `runs/real_coswara_torch_train_e2/RESULTS.md`, `runs/prelong_check_real_coswara/PRELONG_CHECK.md`, and `runs/testround_real_coswara_e1_control_v2/RESULTS.md`.
- Pre-long-training validation commands use `conda run -n coughkd`.
- Compact-baseline long-run command is supported by preflight and testround artifacts. PANNs CNN14 16 kHz foundation long-run command has completed; other foundation-model long-run commands remain gated until their wrappers pass smoke tests.
- Unit tests pass inside `coughkd`: `PYTHONPATH=src conda run -n coughkd python -m unittest discover -s tests`.

### Review 3: Claims and Paper Boundary

Status: passed after one numbering correction.

Checked:

- Compact `ConvTeacher` results are labeled as engineering baselines.
- Compact runs are not described as BEATs/AST/PANNs or foundation-model distillation.
- Long-duration training is marked blocked until readiness gates pass.
- Clinical and diagnosis claims remain explicitly unsupported.
- Paper protocol now includes long-training readiness requirements.

Correction made:

- Fixed `plan.md` section numbering after adding the pre-long-training gate.
- Updated claim boundary from "all long training blocked" to "compact baseline supported, foundation-model long training gated."
