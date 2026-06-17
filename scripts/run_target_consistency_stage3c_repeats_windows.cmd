@echo off
setlocal EnableDelayedExpansion

rem Stage 3c: repeat the decisive control only.
rem This script reuses the seed-7 baseline teacher/student checkpoints and varies
rem the continuation/training seed. It is not a full independent 3-seed baseline
rem rerun, but it is the cheapest test for whether Stage 3b's tiny TCD gain is stable.

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set DEVICE=auto
set BATCH_SIZE=16
set EPOCHS=6
set SOURCE_MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set TARGET_MANIFEST=manifests\coughvid_adapt_test.csv
set BASELINE_RUN=runs\stage1_panns_response_seed7
set PANNS_REPO=external\teacher_repos\audioset_tagging_cnn_zip
set TEACHER_CKPT=pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth

for %%S in (11 23) do (
  echo.
  echo ===== Stage 3c seed %%S: source-only continuation =====
  python -B scripts\train_target_consistency_student.py ^
    --source-manifest "%SOURCE_MANIFEST%" ^
    --target-manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
    --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
    --teacher-kind panns_cnn14_16k ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --teacher-repo "%PANNS_REPO%" ^
    --out runs\stage3c_source_only_seed%%S ^
    --device %DEVICE% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --seed %%S ^
    --lr 5e-5 ^
    --target-weight 0.0 ^
    --confidence-threshold 0.0
  if errorlevel 1 exit /b 1

  echo.
  echo ===== Stage 3c seed %%S: TCD very strong =====
  python -B scripts\train_target_consistency_student.py ^
    --source-manifest "%SOURCE_MANIFEST%" ^
    --target-manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
    --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
    --teacher-kind panns_cnn14_16k ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --teacher-repo "%PANNS_REPO%" ^
    --out runs\stage3c_tcd_very_strong_seed%%S ^
    --device %DEVICE% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --seed %%S ^
    --lr 5e-5 ^
    --target-weight 1.00 ^
    --confidence-threshold 0.0
  if errorlevel 1 exit /b 1

  echo.
  echo ===== Stage 3c seed %%S: TCD confidence 0.35 =====
  python -B scripts\train_target_consistency_student.py ^
    --source-manifest "%SOURCE_MANIFEST%" ^
    --target-manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --init-student-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
    --init-teacher-checkpoint "%BASELINE_RUN%\checkpoints\teacher_best.pt" ^
    --teacher-kind panns_cnn14_16k ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --teacher-repo "%PANNS_REPO%" ^
    --out runs\stage3c_tcd_conf035_seed%%S ^
    --device %DEVICE% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --seed %%S ^
    --lr 5e-5 ^
    --target-weight 0.60 ^
    --confidence-threshold 0.35
  if errorlevel 1 exit /b 1

  echo.
  echo ===== Stage 3c seed %%S: external evaluation =====
  for %%R in (source_only tcd_very_strong tcd_conf035) do (
    python -B scripts\evaluate_external_checkpoint.py ^
      --manifest "%TARGET_MANIFEST%" ^
      --root "%DATA_ROOT%" ^
      --checkpoint runs\stage3c_%%R_seed%%S\checkpoints\student_best.pt ^
      --out runs\external_coughvid_test_stage3c_%%R_seed%%S ^
      --device %DEVICE% ^
      --batch-size %BATCH_SIZE%
    if errorlevel 1 exit /b 1
  )
)

echo.
echo ===== Stage 3c summary inputs ready =====
echo Review external runs:
echo   runs\external_coughvid_test_stage3c_source_only_seed11
echo   runs\external_coughvid_test_stage3c_tcd_very_strong_seed11
echo   runs\external_coughvid_test_stage3c_tcd_conf035_seed11
echo   runs\external_coughvid_test_stage3c_source_only_seed23
echo   runs\external_coughvid_test_stage3c_tcd_very_strong_seed23
echo   runs\external_coughvid_test_stage3c_tcd_conf035_seed23

endlocal
