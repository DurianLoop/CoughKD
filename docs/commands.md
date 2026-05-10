# Commands

All commands are intended to run from the repository root.

## Environment

Primary experiment environment: conda-managed `coughkd`.

```bash
scripts/setup_conda_env.sh coughkd
conda activate coughkd
python -m pip install -r requirements-ml.txt
conda run -n coughkd scripts/validate_project.sh
```

For the dependency-light local path:

```bash
PYTHONPATH=src python -m unittest discover -s tests
scripts/validate_project.sh
```

Optional ML dependencies:

```bash
conda activate coughkd
python -m pip install -r requirements-ml.txt
```

Dependency files:

| File | Role |
|---|---|
| `environment.yml` | Creates the `coughkd` conda environment with Python 3.14 and pip. |
| `requirements.txt` | Documents that the foundation smoke path currently uses only the Python standard library. |
| `requirements-ml.txt` | Installs remote training dependencies inside `coughkd`: CUDA 12.8 PyTorch wheels, `torch`, `torchaudio`, and `numpy`. |

Current observed environment: the active `coughkd` environment reports Python 3.14.4, and `environment.yml` is aligned to Python 3.14.

## Build and Validate Manifests

Generic metadata CSV:

```bash
PYTHONPATH=src python -m coughkd.cli build-manifest \
  --root DATA_ROOT \
  --metadata metadata.csv \
  --dataset coswara \
  --out manifests/coswara.csv \
  --path-column audio_path \
  --subject-column participant_id \
  --label-column diagnosis \
  --split-column split
```

Coswara cough-only manifest:

```bash
PYTHONPATH=src python -m coughkd.cli build-coswara-manifest \
  --root /home/ubuntu/ziyuworkspace/datasets/coswara \
  --metadata combined_data.csv \
  --out manifests/coswara_cough.csv
```

Filter invalid labels and short audio:

```bash
PYTHONPATH=src python -m coughkd.cli filter-manifest \
  --manifest manifests/coswara_cough.csv \
  --root /home/ubuntu/ziyuworkspace/datasets/coswara \
  --out runs/coswara_cough_filtered \
  --min-duration-sec 0.5 \
  --drop-labels under_validation
```

Generate subject-disjoint split:

```bash
PYTHONPATH=src python -m coughkd.cli split-manifest \
  --manifest runs/coswara_cough_filtered/manifest_filtered.csv \
  --root /home/ubuntu/ziyuworkspace/datasets/coswara \
  --out runs/coswara_cough_filtered_split \
  --seed 7
```

Validate selection guard:

```bash
PYTHONPATH=src python -m coughkd.cli check-selection-guard \
  --manifest runs/coswara_cough_filtered_split/manifest_split.csv \
  --selection-splits train,val \
  --out runs/coswara_selection_guard
```

## Smoke Validation

```bash
PYTHONPATH=src python -m coughkd.cli make-smoke-data --out runs/smoke_data
PYTHONPATH=src python -m coughkd.cli validate-manifest --manifest runs/smoke_data/manifest.csv --root . --out runs/smoke_validation
PYTHONPATH=src python -m coughkd.cli split-manifest --manifest runs/smoke_data/manifest.csv --root . --out runs/smoke_split --seed 7
PYTHONPATH=src python -m coughkd.cli preprocess-smoke --manifest runs/smoke_split/manifest_split.csv --root . --out runs/smoke_preprocess
PYTHONPATH=src python -m coughkd.cli metrics-smoke --out runs/smoke_metrics
PYTHONPATH=src python -m coughkd.cli aggregation-smoke --out runs/smoke_aggregation
PYTHONPATH=src python -m coughkd.cli augment-smoke --out runs/smoke_augment
PYTHONPATH=src python -m coughkd.cli model-smoke --out runs/smoke_model
PYTHONPATH=src python -m coughkd.cli benchmark-smoke --out runs/smoke_benchmark
PYTHONPATH=src python -m coughkd.cli subgroup-smoke --out runs/smoke_subgroup
PYTHONPATH=src python -m coughkd.cli baseline-smoke --out runs/smoke_baselines
```

## Ablation and Paper Table Smoke

```bash
PYTHONPATH=src python -m coughkd.cli grid-dry-run --out runs/smoke_grid --limit 3
PYTHONPATH=src python -m coughkd.cli grid-smoke --out runs/smoke_grid --limit 2 --force
PYTHONPATH=src python -m coughkd.cli aggregate-results --runs-dir runs/smoke_grid --out runs/smoke_grid_aggregate --min-runs 2
PYTHONPATH=src python -m coughkd.cli paper-tables-smoke \
  --runs-dir runs/smoke_grid \
  --out runs/smoke_tables \
  --required-run-ids smoke_grid_000,smoke_grid_001
```

## PyTorch Training

One-step real-manifest smoke:

```bash
PYTHONPATH=src python -m coughkd.cli torch-manifest-smoke \
  --manifest runs/coswara_cough_filtered_split/manifest_split.csv \
  --root /home/ubuntu/ziyuworkspace/datasets/coswara \
  --out runs/coswara_torch_manifest_smoke \
  --device auto
```

Closed-loop compact teacher/student training:

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

## Pre-Long-Training Review

Before any multi-hour or multi-day GPU training run, review:

- `docs/long_training_plan.md`
- `docs/clickup_long_training.md`
- `docs/data_and_results.md`

Minimum validation commands:

```bash
conda run -n coughkd python --version
PYTHONPATH=src conda run -n coughkd python -m unittest discover -s tests
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli validate-manifest \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/prelong_manifest_validation
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli check-selection-guard \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --selection-splits train,val \
  --out runs/prelong_selection_guard
PYTHONPATH=src conda run -n coughkd python -m coughkd.cli prelong-check \
  --manifest runs/real_coswara_cough_filtered_split/manifest_split.csv \
  --root ../datasets/coswara \
  --out runs/prelong_check_real_coswara \
  --device cuda \
  --expected-python 3.14
```

Compact-baseline long-run command template, supported after `runs/prelong_check_real_coswara` and `runs/testround_real_coswara_e1_control_v2`:

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

## Teacher Model Downloads

Prepare foundation-teacher checkpoints manually on a networked machine:

```bash
bash scripts/download_teacher_models.sh
```

The script writes checkpoints to `pretrained/teachers/` and source repos to `external/teacher_repos/`; both are ignored by git.

Run the validated PANNs CNN14 16 kHz foundation-teacher long training path:

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

## Paper Build

```bash
make -C paper
```

Manual fallback:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
