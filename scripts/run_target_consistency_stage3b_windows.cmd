@echo off
setlocal

rem Stage 3b: source-only control + stronger target-consistent distillation.
rem Run from: D:\CoughKD\AAAI after activating conda env CoughKD.

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set DEVICE=auto
set SEED=7
set BATCH_SIZE=16
set EPOCHS=6
set SOURCE_MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set TARGET_MANIFEST=manifests\coughvid_adapt_test.csv
set BASELINE_RUN=runs\stage1_panns_response_seed%SEED%
set PANNS_REPO=external\teacher_repos\audioset_tagging_cnn_zip
set TEACHER_CKPT=pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth

if not exist "%SOURCE_MANIFEST%" (
  echo Missing source manifest: %SOURCE_MANIFEST%
  exit /b 1
)

if not exist "%TARGET_MANIFEST%" (
  echo Missing target manifest: %TARGET_MANIFEST%
  exit /b 1
)

if not exist "%BASELINE_RUN%\checkpoints\student_best.pt" (
  echo Missing student checkpoint: %BASELINE_RUN%\checkpoints\student_best.pt
  exit /b 1
)

if not exist "%BASELINE_RUN%\checkpoints\teacher_best.pt" (
  echo Missing teacher checkpoint: %BASELINE_RUN%\checkpoints\teacher_best.pt
  exit /b 1
)

if not exist "%TEACHER_CKPT%" (
  echo Missing PANNs checkpoint: %TEACHER_CKPT%
  exit /b 1
)

if not exist "%PANNS_REPO%\pytorch" (
  echo Missing PANNs repo pytorch directory: %PANNS_REPO%\pytorch
  exit /b 1
)

echo.
echo [1/6] Source-only continuation control
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint "%TEACHER_CKPT%" ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3b_source_only_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.0 ^
  --confidence-threshold 0.0
if errorlevel 1 exit /b 1

echo.
echo [2/6] TCD strong target weight 0.60
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint "%TEACHER_CKPT%" ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3b_tcd_strong_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.60 ^
  --confidence-threshold 0.0
if errorlevel 1 exit /b 1

echo.
echo [3/6] TCD very strong target weight 1.00
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint "%TEACHER_CKPT%" ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3b_tcd_very_strong_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 1.00 ^
  --confidence-threshold 0.0
if errorlevel 1 exit /b 1

echo.
echo [4/6] TCD confidence threshold 0.35
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint "%TEACHER_CKPT%" ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3b_tcd_conf035_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.60 ^
  --confidence-threshold 0.35
if errorlevel 1 exit /b 1

echo.
echo [5/6] External COUGHVID-test evaluation
for %%R in (source_only tcd_strong tcd_very_strong tcd_conf035) do (
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage3b_%%R_seed%SEED%\checkpoints\student_best.pt ^
    --out runs\external_coughvid_test_stage3b_%%R_seed%SEED% ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1
)

echo.
echo [6/6] Domain probe comparison
python -B scripts\domain_probe_students.py ^
  --coswara-manifest "%SOURCE_MANIFEST%" ^
  --coughvid-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --run-dir "%BASELINE_RUN%" ^
  --out runs\domain_probe_stage3b_tcd_seed%SEED% ^
  --samples-per-dataset 1500 ^
  --seed %SEED% ^
  --batch-size %BATCH_SIZE% ^
  --device %DEVICE% ^
  --extra-checkpoint source_only=runs\stage3b_source_only_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint tcd_strong=runs\stage3b_tcd_strong_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint tcd_very_strong=runs\stage3b_tcd_very_strong_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint tcd_conf035=runs\stage3b_tcd_conf035_seed%SEED%\checkpoints\student_best.pt
if errorlevel 1 exit /b 1

echo.
echo Done. Review:
echo   runs\stage3b_source_only_seed%SEED%\RESULTS.md
echo   runs\stage3b_tcd_strong_seed%SEED%\RESULTS.md
echo   runs\stage3b_tcd_very_strong_seed%SEED%\RESULTS.md
echo   runs\stage3b_tcd_conf035_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3b_source_only_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3b_tcd_strong_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3b_tcd_very_strong_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3b_tcd_conf035_seed%SEED%\RESULTS.md
echo   runs\domain_probe_stage3b_tcd_seed%SEED%\DOMAIN_PROBE_REPORT.md

endlocal
