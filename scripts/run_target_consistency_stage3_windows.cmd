@echo off
setlocal

rem Stage 3: Target-consistent distillation.
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

if not exist "%BASELINE_RUN%\checkpoints\student_best.pt" (
  echo Missing student checkpoint: %BASELINE_RUN%\checkpoints\student_best.pt
  exit /b 1
)

if not exist "%BASELINE_RUN%\checkpoints\teacher_best.pt" (
  echo Missing teacher checkpoint: %BASELINE_RUN%\checkpoints\teacher_best.pt
  exit /b 1
)

echo.
echo [1/5] Target consistency low weight
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3_tcd_low_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.10 ^
  --confidence-threshold 0.0 ^
  --weak-noise-std 0.005 ^
  --strong-noise-std 0.020 ^
  --time-mask-ratio 0.08 ^
  --freq-mask-ratio 0.08
if errorlevel 1 exit /b 1

echo.
echo [2/5] Target consistency mid weight
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3_tcd_mid_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.30 ^
  --confidence-threshold 0.0 ^
  --weak-noise-std 0.005 ^
  --strong-noise-std 0.020 ^
  --time-mask-ratio 0.08 ^
  --freq-mask-ratio 0.08
if errorlevel 1 exit /b 1

echo.
echo [3/5] Target consistency confidence-gated
python -B scripts\train_target_consistency_student.py ^
  --source-manifest "%SOURCE_MANIFEST%" ^
  --target-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth ^
  --teacher-repo "%PANNS_REPO%" ^
  --out runs\stage3_tcd_conf_seed%SEED% ^
  --device %DEVICE% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --lr 5e-5 ^
  --target-weight 0.30 ^
  --confidence-threshold 0.45 ^
  --weak-noise-std 0.005 ^
  --strong-noise-std 0.020 ^
  --time-mask-ratio 0.08 ^
  --freq-mask-ratio 0.08
if errorlevel 1 exit /b 1

echo.
echo [4/5] External COUGHVID-test evaluation
for %%R in (low mid conf) do (
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage3_tcd_%%R_seed%SEED%\checkpoints\student_best.pt ^
    --out runs\external_coughvid_test_stage3_tcd_%%R_seed%SEED% ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1
)

echo.
echo [5/5] Domain probe comparison
python -B scripts\domain_probe_students.py ^
  --coswara-manifest "%SOURCE_MANIFEST%" ^
  --coughvid-manifest "%TARGET_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --run-dir "%BASELINE_RUN%" ^
  --out runs\domain_probe_stage3_tcd_seed%SEED% ^
  --samples-per-dataset 1500 ^
  --seed %SEED% ^
  --batch-size %BATCH_SIZE% ^
  --device %DEVICE% ^
  --extra-checkpoint tcd_low=runs\stage3_tcd_low_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint tcd_mid=runs\stage3_tcd_mid_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint tcd_conf=runs\stage3_tcd_conf_seed%SEED%\checkpoints\student_best.pt
if errorlevel 1 exit /b 1

echo.
echo Done. Review:
echo   runs\stage3_tcd_low_seed%SEED%\RESULTS.md
echo   runs\stage3_tcd_mid_seed%SEED%\RESULTS.md
echo   runs\stage3_tcd_conf_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3_tcd_low_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3_tcd_mid_seed%SEED%\RESULTS.md
echo   runs\external_coughvid_test_stage3_tcd_conf_seed%SEED%\RESULTS.md
echo   runs\domain_probe_stage3_tcd_seed%SEED%\DOMAIN_PROBE_REPORT.md

endlocal
