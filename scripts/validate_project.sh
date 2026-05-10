#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Neither python nor python3 was found on PATH" >&2
  exit 1
fi

required_files=(
  "README.md"
  "plan.md"
  "STATUS.md"
  "pyproject.toml"
  "requirements.txt"
  "requirements-ml.txt"
  "environment.yml"
  "reference/literature_and_open_source.md"
  "paper/README.md"
  "paper/main.tex"
  "paper/references.bib"
  "paper/sections/01_introduction.tex"
  "paper/sections/02_related_work.tex"
  "paper/sections/03_method.tex"
  "paper/sections/04_experiments.tex"
  "paper/sections/05_results.tex"
  "paper/sections/06_discussion.tex"
  "paper/sections/07_conclusion.tex"
  "paper/tables/model_comparison.tex"
  "paper/tables/ablation_plan.tex"
  "paper/tables/efficiency_plan.tex"
  "paper/notes/experimental_protocol.md"
  "paper/notes/review_log.md"
  "paper/notes/claim_ledger.md"
  "src/coughkd/__init__.py"
  "src/coughkd/audio.py"
  "src/coughkd/augment.py"
  "src/coughkd/baselines.py"
  "src/coughkd/benchmark.py"
  "src/coughkd/cache.py"
  "src/coughkd/cli.py"
  "src/coughkd/config.py"
  "src/coughkd/datasets.py"
  "src/coughkd/grid.py"
  "src/coughkd/losses.py"
  "src/coughkd/manifest.py"
  "src/coughkd/metrics.py"
  "src/coughkd/models.py"
  "src/coughkd/paper_tables.py"
  "src/coughkd/reporting.py"
  "src/coughkd/segmentation.py"
  "src/coughkd/smoke.py"
  "src/coughkd/torch_models.py"
  "scripts/setup_conda_env.sh"
  "tests/test_foundation.py"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file" >&2
    exit 1
  fi
done

make -C paper

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m unittest discover -s tests

tmp_dir="$(mktemp -d /tmp/coughkd_validate_XXXXXX)"
trap 'rm -rf "$tmp_dir"' EXIT
"$PYTHON" -m coughkd.cli make-smoke-data --out "$tmp_dir/smoke_data" --seed 7
"$PYTHON" -m coughkd.cli split-manifest --manifest "$tmp_dir/smoke_data/manifest.csv" --root "$ROOT_DIR" --out "$tmp_dir/smoke_split" --seed 7
"$PYTHON" -m coughkd.cli validate-manifest --manifest "$tmp_dir/smoke_split/manifest_split.csv" --root "$ROOT_DIR" --out "$tmp_dir/smoke_validation"
"$PYTHON" -m coughkd.cli check-selection-guard --manifest "$tmp_dir/smoke_split/manifest_split.csv" --selection-splits train,val --out "$tmp_dir/smoke_guard"
"$PYTHON" -m coughkd.cli preprocess-smoke --manifest "$tmp_dir/smoke_split/manifest_split.csv" --root "$ROOT_DIR" --out "$tmp_dir/smoke_preprocess" --seed 7
"$PYTHON" -m coughkd.cli metrics-smoke --out "$tmp_dir/smoke_metrics"
"$PYTHON" -m coughkd.cli aggregation-smoke --out "$tmp_dir/smoke_aggregation"
"$PYTHON" -m coughkd.cli augment-smoke --out "$tmp_dir/smoke_augment"
"$PYTHON" -m coughkd.cli model-smoke --out "$tmp_dir/smoke_model"
"$PYTHON" -m coughkd.cli grid-dry-run --out "$tmp_dir/smoke_grid" --limit 3
"$PYTHON" -m coughkd.cli grid-smoke --out "$tmp_dir/smoke_grid" --limit 2 --force
"$PYTHON" -m coughkd.cli grid-smoke --out "$tmp_dir/smoke_grid" --limit 2
"$PYTHON" -m coughkd.cli aggregate-results --runs-dir "$tmp_dir/smoke_grid" --out "$tmp_dir/smoke_grid_aggregate" --min-runs 2
"$PYTHON" -m coughkd.cli paper-tables-smoke --runs-dir "$tmp_dir/smoke_grid" --out "$tmp_dir/smoke_tables" --required-run-ids smoke_grid_000,smoke_grid_001
"$PYTHON" -m coughkd.cli benchmark-smoke --out "$tmp_dir/smoke_benchmark"
"$PYTHON" -m coughkd.cli subgroup-smoke --out "$tmp_dir/smoke_subgroup"
"$PYTHON" -m coughkd.cli baseline-smoke --out "$tmp_dir/smoke_baselines"
"$PYTHON" -m coughkd.cli dataset-smoke --out "$tmp_dir/smoke_dataset"

if "$PYTHON" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
  "$PYTHON" -m coughkd.cli torch-smoke --out "$tmp_dir/smoke_torch" --device auto
else
  echo "- PyTorch smoke skipped: torch is not installed"
fi

if [[ ! -f "paper/main.pdf" ]]; then
  echo "Expected paper/main.pdf to be generated" >&2
  exit 1
fi

if grep -Eq "Undefined control sequence|LaTeX Error|Emergency stop|Fatal error" paper/main.log; then
  echo "LaTeX log contains a hard error" >&2
  exit 1
fi

if grep -Eq "Citation .* undefined|Reference .* undefined|There were undefined references" paper/main.log; then
  echo "LaTeX log contains unresolved citations or references" >&2
  exit 1
fi

if ! rg -q "target|planning values|not measured results|not empirical claims" paper README.md; then
  echo "Target-result safety language is missing" >&2
  exit 1
fi

paper_refs="$(rg '^@' paper/references.bib | wc -l | tr -d ' ')"
project_refs="$(rg '^[0-9]+\.' reference/literature_and_open_source.md | wc -l | tr -d ' ')"

if (( paper_refs < 15 )); then
  echo "Expected at least 15 BibTeX references, found $paper_refs" >&2
  exit 1
fi

if (( project_refs < 50 )); then
  echo "Expected at least 50 project references/resources, found $project_refs" >&2
  exit 1
fi

echo "Validation passed."
echo "- Paper references: $paper_refs"
echo "- Project references/resources: $project_refs"
echo "- Generated: paper/main.pdf"
echo "- Python foundation smoke tests passed"
