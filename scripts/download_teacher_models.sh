#!/usr/bin/env bash
set -euo pipefail

# Download teacher-model checkpoints and reference source repositories.
#
# Run from the repository root:
#   bash scripts/download_teacher_models.sh
#
# Optional environment variables:
#   MODEL_ROOT=pretrained/teachers
#   REPO_ROOT=external/teacher_repos
#   SKIP_HTSAT_GDRIVE=1        # skip Google Drive folder download
#   SKIP_REPOS=1               # skip source repository clones

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/pretrained/teachers}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/external/teacher_repos}"

mkdir -p "${MODEL_ROOT}"/{beats,ast,panns,htsat,passt} "${REPO_ROOT}"

need_python_module() {
  local module="$1"
  local package="$2"
  if ! python -c "import ${module}" >/dev/null 2>&1; then
    echo "[setup] installing ${package}"
    python -m pip install -U "${package}"
  fi
}

download_url() {
  local url="$1"
  local out="$2"
  if [[ -s "${out}" ]]; then
    echo "[skip] ${out}"
    return
  fi
  echo "[download] ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -o "$(basename "${out}")" -d "$(dirname "${out}")" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 5 -o "${out}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
  else
    echo "No downloader found. Install aria2c, curl, or wget." >&2
    exit 1
  fi
}

hf_download_file() {
  local repo="$1"
  local file="$2"
  local out_dir="$3"
  mkdir -p "${out_dir}"
  if [[ -s "${out_dir}/${file}" ]]; then
    echo "[skip] ${out_dir}/${file}"
    return
  fi
  if command -v hf >/dev/null 2>&1; then
    hf download "${repo}" "${file}" --local-dir "${out_dir}"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "${repo}" "${file}" --local-dir "${out_dir}"
  else
    echo "No Hugging Face CLI found. Install huggingface_hub." >&2
    exit 1
  fi
}

clone_repo() {
  local url="$1"
  local dir="$2"
  if [[ -d "${dir}/.git" ]]; then
    echo "[skip] ${dir}"
    return
  fi
  echo "[clone] ${url}"
  git clone --depth 1 "${url}" "${dir}"
}

echo "[setup] project: ${PROJECT_ROOT}"
echo "[setup] model root: ${MODEL_ROOT}"
echo "[setup] repo root: ${REPO_ROOT}"

need_python_module huggingface_hub huggingface_hub
need_python_module gdown gdown

echo "[1/5] BEATs checkpoint"
hf_download_file "lpepino/beats_ckpts" "BEATs_iter3_plus_AS2M.pt" "${MODEL_ROOT}/beats"

echo "[2/5] AST AudioSet checkpoint"
AST_DIR="${MODEL_ROOT}/ast/ast-finetuned-audioset-10-10-0.4593"
hf_download_file "MIT/ast-finetuned-audioset-10-10-0.4593" "config.json" "${AST_DIR}"
hf_download_file "MIT/ast-finetuned-audioset-10-10-0.4593" "preprocessor_config.json" "${AST_DIR}"
hf_download_file "MIT/ast-finetuned-audioset-10-10-0.4593" "model.safetensors" "${AST_DIR}"

echo "[3/5] PANNs CNN14 checkpoints"
download_url \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1" \
  "${MODEL_ROOT}/panns/Cnn14_mAP=0.431.pth"
download_url \
  "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1" \
  "${MODEL_ROOT}/panns/Cnn14_16k_mAP=0.438.pth"

echo "[4/5] HTS-AT checkpoint folder"
if [[ "${SKIP_HTSAT_GDRIVE:-0}" == "1" ]]; then
  echo "[skip] HTS-AT Google Drive folder because SKIP_HTSAT_GDRIVE=1"
else
  mkdir -p "${MODEL_ROOT}/htsat/google_drive_backup"
  gdown --folder "https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6?usp=sharing" \
    -O "${MODEL_ROOT}/htsat/google_drive_backup" || {
      echo "[warn] HTS-AT Google Drive download failed."
      echo "[warn] Manually download HTSAT_AudioSet_Saved_1.ckpt from:"
      echo "[warn] https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6?usp=sharing"
    }
fi

echo "[5/5] PaSST package/release helper"
python -m pip install -U hear21passt
if command -v gh >/dev/null 2>&1; then
  mkdir -p "${MODEL_ROOT}/passt/github_releases"
  gh release download "v.0.0.7-audioset" \
    --repo "kkoutini/PaSST" \
    --dir "${MODEL_ROOT}/passt/github_releases" \
    --skip-existing || true
else
  cat > "${MODEL_ROOT}/passt/MANUAL_DOWNLOAD.txt" <<'EOF'
PaSST can auto-download pretrained models through hear21passt / PaSST APIs.
If you want local release assets, install GitHub CLI and run:

  gh release download v.0.0.7-audioset --repo kkoutini/PaSST --dir pretrained/teachers/passt/github_releases --skip-existing

Release page:
  https://github.com/kkoutini/PaSST/releases
EOF
fi

if [[ "${SKIP_REPOS:-0}" != "1" ]]; then
  echo "[repos] cloning teacher source repositories"
  clone_repo "https://github.com/YuanGongND/ast.git" "${REPO_ROOT}/ast"
  clone_repo "https://github.com/qiuqiangkong/audioset_tagging_cnn.git" "${REPO_ROOT}/audioset_tagging_cnn"
  clone_repo "https://github.com/RetroCirce/HTS-Audio-Transformer.git" "${REPO_ROOT}/HTS-Audio-Transformer"
  clone_repo "https://github.com/kkoutini/PaSST.git" "${REPO_ROOT}/PaSST"
  if [[ ! -d "${REPO_ROOT}/unilm/.git" ]]; then
    git clone --filter=blob:none --sparse https://github.com/microsoft/unilm.git "${REPO_ROOT}/unilm" && \
      git -C "${REPO_ROOT}/unilm" sparse-checkout set beats || {
        echo "[warn] BEATs source repo clone failed."
        echo "[warn] Checkpoint download is enough for the next wrapper step; retry this repo later if needed:"
        echo "[warn] git clone --filter=blob:none --sparse https://github.com/microsoft/unilm.git ${REPO_ROOT}/unilm"
      }
  else
    echo "[skip] ${REPO_ROOT}/unilm"
  fi
fi

cat > "${MODEL_ROOT}/MANIFEST.md" <<EOF
# Teacher Model Download Manifest

Generated by \`scripts/download_teacher_models.sh\`.

## Checkpoints

- BEATs: \`${MODEL_ROOT}/beats/BEATs_iter3_plus_AS2M.pt\`
- AST: \`${AST_DIR}/model.safetensors\`
- PANNs CNN14: \`${MODEL_ROOT}/panns/Cnn14_mAP=0.431.pth\`
- PANNs CNN14 16k: \`${MODEL_ROOT}/panns/Cnn14_16k_mAP=0.438.pth\`
- HTS-AT: \`${MODEL_ROOT}/htsat/google_drive_backup/\`
- PaSST: \`${MODEL_ROOT}/passt/\`

## Source Repositories

- \`${REPO_ROOT}/ast\`
- \`${REPO_ROOT}/audioset_tagging_cnn\`
- \`${REPO_ROOT}/HTS-Audio-Transformer\`
- \`${REPO_ROOT}/PaSST\`
- \`${REPO_ROOT}/unilm/beats\`
EOF

echo "[done] Teacher models prepared under ${MODEL_ROOT}"
