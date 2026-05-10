# AAAI Working Directory

This directory is a trimmed working copy for the AAAI 2027 submission track.

## Kept Contents

- `src/coughkd/`: core pipeline code.
- `tests/`: foundation tests.
- `manifests/`: current Coswara manifests.
- `docs/`: selected planning, data, and comparison notes.
- `paper/`: LaTeX paper source without generated PDFs.
- `scripts/`: environment, validation, and teacher download entry points.
- `environment-aaai.yml` and `setup_env_windows.cmd`: proposed environment setup.

## Main Notes

- `docs/comparison_matrix.md`: model, dataset, and metric comparison plan.
- `docs/research_data_model_positioning.md`: data/model contribution positioning.
- `docs/top_conference_optimization_plan.md`: implementation roadmap.
- `docs/data_and_results.md`: current measured results.

## Windows Setup

From `D:\CoughKD\CoughKD-git\AAAI`:

```bat
setup_env_windows.cmd
```

If conda cannot write to its cache directory, run the command from a terminal with normal user permissions or create the environment manually:

```bat
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set CONDA_PKGS_DIRS=%CD%\.conda\pkgs
set CONDA_ENVS_PATH=%CD%\.conda\envs
conda create -y -p .conda\coughkd-aaai python=3.11 pip
conda activate %CD%\.conda\coughkd-aaai
python -m pip install -r requirements-ml.txt
python -m pip install librosa soundfile scipy scikit-learn pandas matplotlib seaborn tqdm
```
