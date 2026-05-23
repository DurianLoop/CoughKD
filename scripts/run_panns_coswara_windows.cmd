@echo off
setlocal

rem Run the inherited PANNs CNN14 16 kHz teacher -> DepthwiseStudent experiment.
rem Run from: D:\CoughKD\AAAI

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set PANNS_CKPT=%CD%\pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth
set PANNS_REPO=%CD%\external\teacher_repos\audioset_tagging_cnn_zip
set DEVICE=auto
set BATCH_SIZE=16
set TEACHER_EPOCHS=30
set STUDENT_EPOCHS=30
set SEED=7

if not exist "%MANIFEST%" (
  echo Missing split manifest: %MANIFEST%
  echo Run scripts\run_stage1_coswara_windows.cmd first.
  exit /b 1
)

if not exist "%PANNS_CKPT%" (
  echo Missing PANNs checkpoint: %PANNS_CKPT%
  echo Download teacher models first.
  exit /b 1
)

if not exist "%PANNS_REPO%\pytorch" (
  echo Missing PANNs source repo pytorch directory: %PANNS_REPO%\pytorch
  echo Download teacher source repositories first.
  exit /b 1
)

python -m coughkd.cli torch-train ^
  --manifest "%MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage1_panns_cnn14_16k_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind panns_cnn14_16k ^
  --teacher-checkpoint "%PANNS_CKPT%" ^
  --teacher-repo "%PANNS_REPO%" ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --max-duration-sec 4.0 ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.1 ^
  --kd-embedding-weight 0.0 ^
  --kd-relation-weight 0.0

endlocal
