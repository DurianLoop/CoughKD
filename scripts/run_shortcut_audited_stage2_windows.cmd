@echo off
setlocal

rem Stage 2: shortcut-audited KD experiments.
rem Run from: D:\CoughKD\AAAI after activating conda env CoughKD.

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set DEVICE=auto
set SEED=7
set BATCH_SIZE=16
set TEACHER_EPOCHS=0
set STUDENT_EPOCHS=8
set COSWARA_MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set COUGHVID_MANIFEST=manifests\coughvid_adapt_test.csv
set BASELINE_RUN=runs\stage1_panns_response_seed%SEED%
set PANNS_REPO=external\teacher_repos\audioset_tagging_cnn_zip
set FULL_WEIGHT_DIR=runs\shortcut_weights\full_seed%SEED%
set QUALITY_WEIGHT_DIR=runs\shortcut_weights\quality_only_seed%SEED%
set DOMAIN_WEIGHT_DIR=runs\shortcut_weights\domain_only_seed%SEED%

if not exist "%COSWARA_MANIFEST%" (
  echo Missing Coswara split manifest: %COSWARA_MANIFEST%
  echo Run scripts\run_stage1_coswara_windows.cmd first.
  exit /b 1
)

if not exist "%COUGHVID_MANIFEST%" (
  echo Missing COUGHVID adapt/test manifest: %COUGHVID_MANIFEST%
  echo Build it with scripts\build_coughvid_manifest.py and scripts\split_coughvid_adapt_test.py first.
  exit /b 1
)

if not exist "%BASELINE_RUN%\checkpoints\student_best.pt" (
  echo Missing baseline KD checkpoint: %BASELINE_RUN%\checkpoints\student_best.pt
  echo Run the inherited PANNs baseline first.
  exit /b 1
)

if not exist runs mkdir runs

echo.
echo [1/7] Build full shortcut weights
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out "%FULL_WEIGHT_DIR%" ^
  --source-split train ^
  --target-split adapt ^
  --stability-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --device %DEVICE% ^
  --seed %SEED% ^
  --floor 0.05
if errorlevel 1 exit /b 1

echo.
echo [2/7] Quality-only shortcut-audited KD
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out "%QUALITY_WEIGHT_DIR%" ^
  --source-split train ^
  --target-split adapt ^
  --quality-power 1.0 ^
  --domain-power 0.0 ^
  --stability-power 0.0 ^
  --floor 0.05
if errorlevel 1 exit /b 1

python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2_sakd_quality_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.1 ^
  --kd-embedding-weight 0.0 ^
  --kd-relation-weight 0.0 ^
  --kd-sample-weights "%QUALITY_WEIGHT_DIR%\shortcut_weights.csv"
if errorlevel 1 exit /b 1

echo.
echo [3/7] Domain-risk-only shortcut-audited KD
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out "%DOMAIN_WEIGHT_DIR%" ^
  --source-split train ^
  --target-split adapt ^
  --quality-power 0.0 ^
  --domain-power 1.0 ^
  --stability-power 0.0 ^
  --floor 0.05
if errorlevel 1 exit /b 1

python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2_sakd_domain_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.1 ^
  --kd-embedding-weight 0.0 ^
  --kd-relation-weight 0.0 ^
  --kd-sample-weights "%DOMAIN_WEIGHT_DIR%\shortcut_weights.csv"
if errorlevel 1 exit /b 1

echo.
echo [4/7] Full shortcut-audited KD
python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2_sakd_full_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.1 ^
  --kd-embedding-weight 0.0 ^
  --kd-relation-weight 0.0 ^
  --kd-sample-weights "%FULL_WEIGHT_DIR%\shortcut_weights.csv"
if errorlevel 1 exit /b 1

echo.
echo [5/7] External COUGHVID-test evaluation
for %%R in (quality domain full) do (
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%COUGHVID_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage2_sakd_%%R_seed%SEED%\checkpoints\student_best.pt ^
    --out runs\external_coughvid_test_stage2_sakd_%%R_seed%SEED% ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1
)

echo.
echo [6/7] Domain probe comparison
python -B scripts\domain_probe_students.py ^
  --coswara-manifest "%COSWARA_MANIFEST%" ^
  --coughvid-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --run-dir "%BASELINE_RUN%" ^
  --out runs\domain_probe_stage2_sakd_seed%SEED% ^
  --samples-per-dataset 1500 ^
  --seed %SEED% ^
  --batch-size %BATCH_SIZE% ^
  --device %DEVICE% ^
  --extra-checkpoint sakd_quality=runs\stage2_sakd_quality_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint sakd_domain=runs\stage2_sakd_domain_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint sakd_full=runs\stage2_sakd_full_seed%SEED%\checkpoints\student_best.pt
if errorlevel 1 exit /b 1

echo.
echo [7/7] Done
echo Review:
echo   %FULL_WEIGHT_DIR%\SHORTCUT_WEIGHTS.md
echo   runs\stage2_sakd_quality_seed%SEED%\RESULTS.md
echo   runs\stage2_sakd_domain_seed%SEED%\RESULTS.md
echo   runs\stage2_sakd_full_seed%SEED%\RESULTS.md
echo   runs\domain_probe_stage2_sakd_seed%SEED%\DOMAIN_PROBE_REPORT.md

endlocal
