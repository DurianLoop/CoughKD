@echo off
setlocal EnableDelayedExpansion

rem Independent PANNs baseline repeats for analysis-paper evidence.
rem Runs seed 11 and 23 from scratch: PANNs teacher head, CE student, vanilla KD student,
rem then COUGHVID-test external evaluation and domain/task probe.

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set DEVICE=auto
set BATCH_SIZE=16
set TEACHER_EPOCHS=8
set STUDENT_EPOCHS=8
set SOURCE_MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set TARGET_MANIFEST=manifests\coughvid_adapt_test.csv
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

if not exist "%TEACHER_CKPT%" (
  echo Missing PANNs checkpoint: %TEACHER_CKPT%
  exit /b 1
)

if not exist "%PANNS_REPO%\pytorch" (
  echo Missing PANNs repo pytorch directory: %PANNS_REPO%\pytorch
  exit /b 1
)

for %%S in (11 23) do (
  echo.
  echo ===== PANNs baseline seed %%S =====
  python -B -m coughkd.cli torch-train ^
    --manifest "%SOURCE_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --out runs\stage1_panns_response_seed%%S ^
    --device %DEVICE% ^
    --teacher-kind panns_cnn14_16k ^
    --teacher-checkpoint "%TEACHER_CKPT%" ^
    --teacher-repo "%PANNS_REPO%" ^
    --teacher-epochs %TEACHER_EPOCHS% ^
    --student-epochs %STUDENT_EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --seed %%S ^
    --kd-temperature 2.0 ^
    --kd-response-weight 0.7 ^
    --kd-feature-weight 0.1 ^
    --kd-embedding-weight 0.0 ^
    --kd-relation-weight 0.0
  if errorlevel 1 exit /b 1

  echo.
  echo ===== External COUGHVID-test seed %%S: CE student =====
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage1_panns_response_seed%%S\checkpoints\ce_student_best.pt ^
    --out runs\external_coughvid_test_ce_seed%%S ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1

  echo.
  echo ===== External COUGHVID-test seed %%S: KD student =====
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage1_panns_response_seed%%S\checkpoints\student_best.pt ^
    --out runs\external_coughvid_test_kd_seed%%S ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1

  echo.
  echo ===== Domain/task probe seed %%S =====
  python -B scripts\domain_probe_students.py ^
    --coswara-manifest "%SOURCE_MANIFEST%" ^
    --coughvid-manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --run-dir runs\stage1_panns_response_seed%%S ^
    --out runs\domain_probe_stage1_panns_seed%%S ^
    --samples-per-dataset 1500 ^
    --seed %%S ^
    --batch-size %BATCH_SIZE% ^
    --device %DEVICE%
  if errorlevel 1 exit /b 1
)

echo.
echo Done. Review:
echo   runs\stage1_panns_response_seed11\RESULTS.md
echo   runs\stage1_panns_response_seed23\RESULTS.md
echo   runs\external_coughvid_test_ce_seed11\RESULTS.md
echo   runs\external_coughvid_test_kd_seed11\RESULTS.md
echo   runs\external_coughvid_test_ce_seed23\RESULTS.md
echo   runs\external_coughvid_test_kd_seed23\RESULTS.md
echo   runs\domain_probe_stage1_panns_seed11\DOMAIN_PROBE_REPORT.md
echo   runs\domain_probe_stage1_panns_seed23\DOMAIN_PROBE_REPORT.md

endlocal
