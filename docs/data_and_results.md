# Data and Results

## Current Dataset Artifacts

### Raw Coswara Cough Manifest

Artifact: `manifests/coswara_cough.csv` and `runs/coswara_cough_split/manifest_report.md`.

The unfiltered Coswara cough manifest contains `cough-heavy` and `cough-shallow` recordings joined with participant metadata. The unfiltered split report contains 5,491 recordings from 2,746 subjects. It includes the label `under_validation` and contains invalid/short audio:

| Quantity | Value |
|---|---:|
| Recordings | 5,491 |
| Subjects | 2,746 |
| Zero-duration recordings | 83 |
| Recordings shorter than 0.5 s | 92 |
| Recordings shorter than 1.0 s | 125 |

### Filtered Coswara Cough Manifest

Artifact: `runs/coswara_cough_filtered/manifest_filtered.csv`.

Filtering removed `under_validation` and recordings shorter than 0.5 seconds. The filtered manifest has no validation errors.

| Label | Count |
|---|---:|
| healthy | 2,804 |
| covid_positive | 1,353 |
| exposed | 496 |
| respiratory_illness | 305 |
| covid_recovered | 289 |
| **Total** | **5,247** |

Duration summary after filtering:

| Metric | Seconds |
|---|---:|
| Minimum | 0.512 |
| Mean | 5.633 |
| Maximum | 30.037 |

### Subject-Disjoint Split

Artifact: `runs/coswara_cough_filtered_split/manifest_split.csv`.

| Split | Recordings | Subjects |
|---|---:|---:|
| train | 3,671 | 1,844 |
| validation | 785 | 395 |
| test | 791 | 396 |
| **Total** | **5,247** | **2,635** |

Label distribution by split:

| Split | covid_positive | covid_recovered | exposed | healthy | respiratory_illness |
|---|---:|---:|---:|---:|---:|
| train | 944 | 206 | 358 | 1,942 | 221 |
| validation | 199 | 35 | 70 | 441 | 40 |
| test | 210 | 48 | 68 | 421 | 44 |

Interpretation: the split is subject-disjoint and usable for an engineering baseline. Class imbalance is substantial, so AUROC, macro-F1, calibration, and class-aware analysis matter more than raw accuracy.

## Measured Closed-Loop Coswara Baseline

Artifact: `runs/coswara_torch_closed_loop_full/FINAL_REPORT.md`.

Training command:

```bash
PYTHONPATH=src python -u -m coughkd.cli torch-train \
  --manifest runs/coswara_cough_filtered_split/manifest_split.csv \
  --root /home/ubuntu/ziyuworkspace/datasets/coswara \
  --out runs/coswara_torch_closed_loop_full \
  --device cuda \
  --teacher-epochs 5 \
  --student-epochs 5 \
  --batch-size 64 \
  --num-workers 4 \
  --max-duration-sec 4.0
```

Measured test results:

| Model | Params | Test accuracy | Test macro-F1 | Test macro OVR AUROC | Test loss |
|---|---:|---:|---:|---:|---:|
| ConvTeacher | 110,277 | 0.390645 | 0.209663 | 0.577184 | 1.565003 |
| DepthwiseStudent + KD | 20,717 | 0.256637 | 0.204042 | 0.580959 | 1.568251 |

Efficiency interpretation:

| Comparison | Value |
|---|---:|
| Parameter reduction | 5.32x |
| Student AUROC / teacher AUROC | 100.65% |
| Accuracy delta, student - teacher | -0.134008 |
| Macro-F1 delta, student - teacher | -0.005621 |
| Macro OVR AUROC delta, student - teacher | +0.003775 |

Scientific interpretation:

- The closed-loop pipeline is functional: train/validation/test splits, checkpoint selection, KD training, held-out inference, and artifact export all work.
- The result is an engineering baseline, not a clinical result.
- The compact student slightly exceeds the compact teacher in macro OVR AUROC on this run, but both models are weak classifiers.
- The teacher is not yet a foundation model, so this result should not be used as evidence for the paper's final CoughKD contribution.
- Repeated seeds, CE-only student control, external validation, calibration, and subgroup analyses are required before making model-performance claims.

## Fresh Real-Data Validation Run

Artifact: `runs/real_coswara_torch_train_e2/RESULTS.md`.

This run was executed against the actual dataset directory discovered at `../datasets/coswara`. The user mentioned `../data`, but the available workspace data directory is `../datasets`; `../data` does not exist in the current filesystem. COUGHVID, Cambridge COVID-19 Sounds, and DiCOVA directories are present but empty. ESC50 and FSD50K are present, but they are general audio datasets rather than the current cough-screening task data.

Commands executed:

```bash
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli build-coswara-manifest \
  --root ../datasets/coswara \
  --metadata combined_data.csv \
  --out manifests/real_coswara_cough.csv

PYTHONPATH=src conda run -n coughkd python -m coughkd.cli filter-manifest \
  --manifest manifests/real_coswara_cough.csv \
  --root ../datasets/coswara \
  --out runs/real_coswara_cough_filtered \
  --min-duration-sec 0.5 \
  --drop-labels under_validation

PYTHONPATH=src conda run -n coughkd python -m coughkd.cli split-manifest \
  --manifest runs/real_coswara_cough_filtered/manifest_filtered.csv \
  --root ../datasets/coswara \
  --out runs/real_coswara_cough_filtered_split \
  --seed 7

PYTHONPATH=src conda run -n coughkd python -m coughkd.cli torch-manifest-smoke \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/real_coswara_torch_manifest_smoke \
  --device cuda \
  --batch-size 8

PYTHONPATH=src conda run -n coughkd python -u -m coughkd.cli torch-train \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/real_coswara_torch_train_e2 \
  --device cuda \
  --teacher-epochs 2 \
  --student-epochs 2 \
  --batch-size 64 \
  --num-workers 4 \
  --max-duration-sec 4.0
```

Environment:

| Item | Value |
|---|---|
| Conda environment | `coughkd` |
| Python reported by env | 3.14.4 |
| PyTorch | 2.11.0+cu128 |
| CUDA available | yes |
| GPU | NVIDIA GeForce RTX 5090 |

Fresh manifest and filtering result:

| Step | Result |
|---|---:|
| Raw Coswara cough manifest | 5,491 recordings / 2,746 subjects |
| Dropped `under_validation` | 162 recordings |
| Dropped shorter than 0.5 s | 82 recordings |
| Filtered manifest | 5,247 recordings / 2,635 subjects |
| Split | train 3,671 / val 785 / test 791 |

Single-step CUDA smoke result:

| Quantity | Value |
|---|---:|
| Dataset records | 5,247 |
| Batch shape | `[8, 1, 398, 32]` |
| Teacher params | 110,277 |
| Student params | 20,717 |

## PANNs CNN14 16 kHz Foundation-Teacher Long Run

Artifact: `runs/long_coswara_panns_cnn14_16k_seed7_e30/RESULTS.md`.

This is the first completed foundation-teacher long run. The teacher uses the official PANNs CNN14 16 kHz checkpoint `pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth`, freezes the AudioSet backbone, and trains a cough-label classifier head on the subject-disjoint Coswara split. The student remains the in-repository depthwise model.

Command:

```bash
PYTHONPATH=src conda run -n coughkd python -u -m coughkd.cli torch-train \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/long_coswara_panns_cnn14_16k_seed7_e30 \
  --device cuda \
  --teacher-kind panns_cnn14_16k \
  --teacher-epochs 30 \
  --student-epochs 30 \
  --batch-size 16 \
  --num-workers 4 \
  --max-duration-sec 4.0 \
  --seed 7
```

Measured held-out test results:

| Model | Params | Test accuracy | Test macro-F1 | Test macro OVR AUROC | Test loss |
|---|---:|---:|---:|---:|---:|
| PANNs CNN14 16 kHz teacher | 81,043,476 | 0.374210 | 0.271545 | 0.613420 | 1.522509 |
| DepthwiseStudent CE-only | 20,717 | 0.445006 | 0.234462 | 0.563910 | 1.501185 |
| DepthwiseStudent KD | 20,717 | 0.361568 | 0.240593 | 0.566423 | 1.539683 |

Interpretation:

- The foundation-teacher interface is now proven on a real checkpoint, real Coswara data, CUDA, validation checkpoint selection, and held-out test inference.
- The PANNs teacher is stronger than the compact teacher in macro OVR AUROC, but still weak for clinical use.
- KD slightly improves over the CE-only student in macro OVR AUROC in this run, but the difference is small and requires repeated seeds.
- BEATs, AST, HTS-AT, and PaSST are downloaded/available but are not yet valid long-run teachers until their dedicated wrappers pass the same smoke and full-run gates.
| Initial KD loss | 1.658685 |
| Final KD loss after one step | 1.614854 |

2+2 epoch held-out test result:

| Model | Accuracy | Macro-F1 | Macro OVR AUROC | Loss |
|---|---:|---:|---:|---:|
| ConvTeacher | 0.361568 | 0.210341 | 0.566054 | 1.580618 |
| DepthwiseStudent + KD | 0.398230 | 0.223478 | 0.571329 | 1.558679 |

Interpretation:

- The real-data training and validation path works end to end in the `coughkd` conda environment.
- The student is 5.32x smaller by parameter count and slightly exceeds the compact teacher on macro OVR AUROC in this short run.
- The result remains an engineering validation. It is short, single-seed, compact-teacher-only, and lacks external validation.
- The environment reports Python 3.14.4, and `environment.yml` is now aligned to Python 3.14.

## Pre-Long-Training Check and Control Testround

Artifacts:

- `runs/prelong_check_real_coswara/PRELONG_CHECK.md`
- `runs/testround_real_coswara_e1_control_v2/RESULTS.md`

The pre-long-training check now validates the frozen real Coswara manifest, subject-disjoint split, audio readability on a sampled batch, CUDA availability, forward-pass loss, manifest hash, and environment snapshot. It passed without blocking issues.

Preflight summary:

| Item | Value |
|---|---|
| Status | ok |
| Manifest SHA256 | `75ad6203c4fb9f600844390b547941c6a0a916f74181de7143fd6995e1928622` |
| Records / subjects | 5,247 / 2,635 |
| Split | train 3,671 / val 785 / test 791 |
| Device | CUDA, NVIDIA GeForce RTX 5090 |
| Python / Torch | 3.14.4 / 2.11.0+cu128 |
| Batch shape | `[8, 1, 398, 32]` |
| Batch load seconds | 0.150844 |
| Forward loss | 1.099105 |

The control testround runs all three compact-baseline models under the same split and seed policy:

- compact `ConvTeacher`;
- depthwise student trained with CE only;
- depthwise student trained with response and feature KD.

1+1+1 epoch held-out test result:

| Model | Accuracy | Macro-F1 | Macro OVR AUROC | Loss |
|---|---:|---:|---:|---:|
| ConvTeacher | 0.337547 | 0.178653 | 0.547007 | 1.550890 |
| DepthwiseStudent CE-only | 0.332491 | 0.206201 | 0.549763 | 1.596538 |
| DepthwiseStudent KD | 0.271808 | 0.160069 | 0.562940 | 1.572238 |

Interpretation:

- The compact-baseline long-run gate is now supported: environment, manifest, GPU, CE-only control, KD control, checkpoints, predictions, manifest hash, environment snapshot, per-class metrics, and confusion matrices are all produced.
- The testround does not prove final model quality because it is intentionally short.
- Foundation-model long training remains gated until a real pretrained teacher wrapper is implemented.
- External validation remains unavailable because the expected external cough dataset directories are empty in this workspace.

## Smoke Outputs

Smoke artifacts prove plumbing and report schemas rather than scientific performance.

| Artifact | Purpose | Interpretation |
|---|---|---|
| `runs/smoke_metrics/metrics.md` | deterministic metric functions | Fixture data is intentionally separable, so perfect AUROC/F1 is expected. |
| `runs/smoke_grid_aggregate/aggregate_results.md` | grid runner and aggregation schema | Two tiny completed grid runs validate resumability and table shape. |
| `runs/smoke_subgroup/subgroup_report.md` | subgroup reporting and small-N behavior | Demonstrates subgroup report format and clinical-caution language. |
| `runs/smoke_benchmark/benchmark_report.json` | export/latency schema | Dependency-free placeholder, not production deployment evidence. |
| `runs/smoke_validation/manifest_report.md` | negative manifest validation fixture | Contains expected split errors for an intentionally unsplit manifest. |

## Current Evidence Boundary

The project can currently claim:

- A reproducible cough-audio research pipeline exists.
- Coswara cough manifest construction, filtering, and subject-disjoint splitting are implemented.
- A compact PyTorch teacher/student KD baseline can train and evaluate end to end.
- The current student is smaller than the current compact teacher and retains comparable held-out macro OVR AUROC in one run.

The project cannot yet claim:

- Medical diagnosis capability.
- SOTA cough disease classification.
- Robustness across datasets or recording devices.
- A successful distillation from BEATs/AST/PANNs.
- A deployable model with measured mobile/edge latency.
- Patient identity verification performance.
