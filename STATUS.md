# Project Status

## Repository Type

Research and experimental pipeline repository for COUGHKD:

- Paper/protocol layer is present and builds to `paper/main.pdf`.
- Executable ML pipeline is being introduced incrementally.
- Current environment has Python 3.14 but no NumPy/SciPy/scikit-learn installed, so the foundation layer is implemented with the Python standard library only.
- Remote/server execution is expected to use the `coughkd` conda environment defined by `environment.yml`.
- Detailed documentation tree is available under `docs/`.
- A measured compact Coswara engineering baseline is available under `runs/coswara_torch_closed_loop_full/`.
- Long-duration training is gated by `docs/long_training_plan.md` and `docs/clickup_long_training.md`.

## Available Commands

- Project validation: `scripts/validate_project.sh`
- Conda setup: `scripts/setup_conda_env.sh coughkd`
- Conda validation: `conda run -n coughkd scripts/validate_project.sh`
- Optional ML dependencies: `python -m pip install -r requirements-ml.txt`
- Paper build: `make -C paper`
- Dataset manifest import: `PYTHONPATH=src python -m coughkd.cli build-manifest --root DATA_ROOT --metadata metadata.csv --dataset coswara --out manifests/coswara.csv --path-column audio_path --subject-column participant_id --label-column diagnosis --split-column split`
- Coswara manifest import: `PYTHONPATH=src python -m coughkd.cli build-coswara-manifest --root DATA_ROOT --metadata combined_data.csv --out manifests/coswara_cough.csv`
- Manifest filtering: `PYTHONPATH=src python -m coughkd.cli filter-manifest --manifest manifests/coswara_cough.csv --root DATA_ROOT --out runs/coswara_cough_filtered --min-duration-sec 0.5 --drop-labels under_validation`
- Dataset adapter smoke: `PYTHONPATH=src python -m coughkd.cli dataset-smoke --out runs/smoke_dataset`
- PyTorch smoke: `PYTHONPATH=src python -m coughkd.cli torch-smoke --out runs/smoke_torch --device auto`
- PyTorch manifest smoke: `PYTHONPATH=src python -m coughkd.cli torch-manifest-smoke --manifest runs/coswara_cough_filtered_split/manifest_split.csv --root DATA_ROOT --out runs/coswara_torch_manifest_smoke --device auto`
- Smoke data generation: `PYTHONPATH=src python -m coughkd.cli make-smoke-data --out runs/smoke_data`
- Manifest validation: `PYTHONPATH=src python -m coughkd.cli validate-manifest --manifest runs/smoke_data/manifest.csv --root . --out runs/smoke_validation`
- Split generation: `PYTHONPATH=src python -m coughkd.cli split-manifest --manifest runs/smoke_data/manifest.csv --out runs/smoke_split --seed 7`
- Preprocessing smoke: `PYTHONPATH=src python -m coughkd.cli preprocess-smoke --manifest runs/smoke_split/manifest_split.csv --root . --out runs/smoke_preprocess`
- Metrics smoke: `PYTHONPATH=src python -m coughkd.cli metrics-smoke --out runs/smoke_metrics`
- Aggregation smoke: `PYTHONPATH=src python -m coughkd.cli aggregation-smoke --out runs/smoke_aggregation`
- Augmentation smoke: `PYTHONPATH=src python -m coughkd.cli augment-smoke --out runs/smoke_augment`
- Model/KD smoke: `PYTHONPATH=src python -m coughkd.cli model-smoke --out runs/smoke_model`
- Ablation grid dry run: `PYTHONPATH=src python -m coughkd.cli grid-dry-run --out runs/smoke_grid`
- Smoke grid: `PYTHONPATH=src python -m coughkd.cli grid-smoke --out runs/smoke_grid --limit 2 --force`
- Aggregate results: `PYTHONPATH=src python -m coughkd.cli aggregate-results --runs-dir runs/smoke_grid --out runs/smoke_grid_aggregate --min-runs 2`
- Paper table smoke: `PYTHONPATH=src python -m coughkd.cli paper-tables-smoke --runs-dir runs/smoke_grid --out runs/smoke_tables --required-run-ids smoke_grid_000,smoke_grid_001`
- Benchmark smoke: `PYTHONPATH=src python -m coughkd.cli benchmark-smoke --out runs/smoke_benchmark`
- Subgroup smoke: `PYTHONPATH=src python -m coughkd.cli subgroup-smoke --out runs/smoke_subgroup`
- Baseline smoke: `PYTHONPATH=src python -m coughkd.cli baseline-smoke --out runs/smoke_baselines`
- Pre-long-training plan: `docs/long_training_plan.md`
- ClickUp task list: `docs/clickup_long_training.md`
- Teacher-model download helper: `bash scripts/download_teacher_models.sh`

## Current Implementation Scope

Main experiment environment:

- Conda environment name: `coughkd`.
- Defined by: `environment.yml`.
- Declared Python version: 3.14 in `environment.yml`.
- Observed Python version: 3.14.4 in the active `coughkd` environment.
- Base dependency policy: foundation smoke path uses only the Python standard library.
- Remote/ML dependency file: `requirements-ml.txt`, installed inside `coughkd`.
- Current optional ML packages: `torch`, `torchaudio`, `numpy` from the CUDA 12.8 PyTorch wheel index plus PyPI fallback.

Completed foundation pieces:

- Manifest schema validation.
- Duplicate recording checks.
- Missing-file checks.
- Duration quality warnings and manifest filtering by minimum duration / dropped labels.
- Subject leakage detection across splits.
- Subject-disjoint split generation.
- Tiny synthetic WAV fixture generation.
- Standard-library WAV loading, mono conversion, lightweight resampling.
- Invalid-recording checks for duration, energy, and clipping.
- Lightweight log-mel-like frontend for smoke testing.
- Binary classification metrics: AUROC, AUPRC, macro-F1, sensitivity, specificity, ECE.
- Bootstrap AUROC confidence intervals and external AUROC drop.
- One-vs-rest multiclass AUROC.
- Recording-level aggregation utilities.
- Multi-level KD loss math on synthetic tensors.
- Teacher/student smoke model interface.
- Ablation grid generation, resume-safe smoke runner, and result aggregation.
- Smoke benchmark/export/quantization report.
- Subgroup reporting with small-N suppression and clinical-claim guard.
- Paper table generation with run-id audit.
- Classical baseline smoke outputs using a dependency-free nearest-centroid implementation.
- Generic metadata/directory dataset adapters that convert public cough corpora into the project manifest schema.
- Coswara-specific cough manifest adapter for extracted `cough-heavy` and `cough-shallow` WAV files with age/sex/country/symptom metadata.
- Optional PyTorch manifest dataset, padded batch collation, compact convolutional teacher, MobileNet-style depthwise student, and one-step KD smoke training.
- Real-manifest PyTorch smoke path that runs one KD optimization step on a project manifest.
- Multi-epoch PyTorch training path with validation checkpointing, test inference, checkpoint export, prediction CSV export, metrics JSON, and markdown summary.
- Filtered Coswara cough manifest with 5,247 recordings from 2,635 subjects.
- Subject-disjoint Coswara split with 3,671 train, 785 validation, and 791 test recordings.
- Compact ConvTeacher vs DepthwiseStudent+KD closed-loop CUDA run. Test macro OVR AUROC: teacher 0.577184, student 0.580959.
- Fresh real-data validation run from `../datasets/coswara`: `runs/real_coswara_torch_train_e2`. This 2+2 epoch CUDA run completed with teacher test macro OVR AUROC 0.566054 and student test macro OVR AUROC 0.571329.
- Documentation tree covering checklist, data/results, commands, and visualization plan.
- Long-training readiness gates covering environment reconciliation, data freeze, profiling, CE-only controls, foundation-teacher integration, metrics/plots, and three-review signoff.
- Pre-long-training check passed for real Coswara under `runs/prelong_check_real_coswara`.
- Real testround with teacher, CE-only student, and KD student passed under `runs/testround_real_coswara_e1_control_v2`.
- Teacher checkpoint downloads verified for BEATs, AST, PANNs CNN14, PANNs CNN14 16 kHz, and HTS-AT AudioSet. AST loads offline as `ASTForAudioClassification` after aligning `torchvision` to CUDA 12.8. `hear21passt` imports successfully for PaSST. Reference source repositories are present under `external/teacher_repos/`, including `unilm/beats/BEATs.py`.
- Real foundation-teacher training interface is implemented and validated for `panns_cnn14_16k` through `PYTHONPATH=src python -m coughkd.cli torch-train --teacher-kind panns_cnn14_16k ...`.
- Full Coswara PANNs CNN14 16 kHz long run completed under `runs/long_coswara_panns_cnn14_16k_seed7_e30`. Test macro OVR AUROC: teacher 0.613420, CE-only student 0.563910, KD student 0.566423.
- Next optimization direction selected for top-conference evidence: strengthen the PANNs KD recipe first, targeting >=95% teacher AUROC retention and >=+0.01 AUROC over CE-only, then replicate with AST/BEATs wrappers.
- Post-download gates passed: unit tests, CUDA torch-smoke, and real Coswara pre-long check under `runs/prelong_check_after_teacher_download/`.
- Top-conference KD method controls are now exposed in `torch-train`: temperature, response KD weight, feature KD weight, embedding KD weight, relation KD weight, and label smoothing. PyTorch training logs response, feature, embedding, and relation KD components to `events.jsonl`, records the learned feature-projector audit in `config.json`, and reports macro one-vs-rest AUPRC, ECE, and Brier score alongside AUROC/F1/loss.

Not yet implemented:

- Dataset-specific recipes and quality checks for COUGHVID/DiCOVA/etc. beyond generic metadata/directory import.
- Coswara quality annotation joins beyond the currently materialized cough-only split.
- Real pretrained teacher wrappers for BEATs/AST/PaSST/HTS-AT beyond the current validated PANNs CNN14 16 kHz wrapper.
- CE-only student comparison runs, repeated-seed evaluation, and measured confidence intervals.
- Attention-transfer KD for teacher backbones with compatible time-frequency attention maps. Response KD, feature-projection KD, cosine embedding KD, and relation KD are implemented in the PyTorch training path.
- Real deployment export and latency benchmarking for PyTorch/ONNX/TFLite models.
- Repeated-seed PANNs long runs and long runs for additional foundation teachers.
- KD hyperparameter sweep, label-schema experiments, artifact-generated plots, run-family audits, and deployment latency benchmarks.
