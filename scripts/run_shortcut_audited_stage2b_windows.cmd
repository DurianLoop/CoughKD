@echo off
setlocal

rem Stage 2b: tempered shortcut-audited KD.
rem Purpose: Stage 2 improved in-domain AUROC but hurt external AUROC and did not reduce domain probe AUC.
rem This sweep tests whether the first weights were too aggressive.

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

if not exist "%BASELINE_RUN%\checkpoints\teacher_best.pt" (
  echo Missing teacher checkpoint: %BASELINE_RUN%\checkpoints\teacher_best.pt
  exit /b 1
)

echo.
echo [1/6] Tempered full weights: floor 0.30, quality 0.50, domain 0.25, stability 0.50
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\shortcut_weights\tempered_full_seed%SEED% ^
  --source-split train ^
  --target-split adapt ^
  --stability-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --device %DEVICE% ^
  --seed %SEED% ^
  --quality-power 0.50 ^
  --domain-power 0.25 ^
  --stability-power 0.50 ^
  --floor 0.30
if errorlevel 1 exit /b 1

python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2b_sakd_tempered_full_seed%SEED% ^
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
  --kd-sample-weights runs\shortcut_weights\tempered_full_seed%SEED%\shortcut_weights.csv
if errorlevel 1 exit /b 1

echo.
echo [2/6] Tempered domain weights: floor 0.30, domain 0.25 only
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\shortcut_weights\tempered_domain_seed%SEED% ^
  --source-split train ^
  --target-split adapt ^
  --quality-power 0.0 ^
  --domain-power 0.25 ^
  --stability-power 0.0 ^
  --floor 0.30
if errorlevel 1 exit /b 1

python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2b_sakd_tempered_domain_seed%SEED% ^
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
  --kd-sample-weights runs\shortcut_weights\tempered_domain_seed%SEED%\shortcut_weights.csv
if errorlevel 1 exit /b 1

echo.
echo [3/6] Mild full weights: floor 0.50, quality 0.25, domain 0.15, stability 0.25
python -B scripts\build_shortcut_weights.py ^
  --source-manifest "%COSWARA_MANIFEST%" ^
  --target-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\shortcut_weights\mild_full_seed%SEED% ^
  --source-split train ^
  --target-split adapt ^
  --stability-checkpoint "%BASELINE_RUN%\checkpoints\student_best.pt" ^
  --device %DEVICE% ^
  --seed %SEED% ^
  --quality-power 0.25 ^
  --domain-power 0.15 ^
  --stability-power 0.25 ^
  --floor 0.50
if errorlevel 1 exit /b 1

python -B -m coughkd.cli torch-train ^
  --manifest "%COSWARA_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --out runs\stage2b_sakd_mild_full_seed%SEED% ^
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
  --kd-sample-weights runs\shortcut_weights\mild_full_seed%SEED%\shortcut_weights.csv
if errorlevel 1 exit /b 1

echo.
echo [4/6] External COUGHVID-test evaluation
for %%R in (tempered_full tempered_domain mild_full) do (
  python -B scripts\evaluate_external_checkpoint.py ^
    --manifest "%COUGHVID_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --checkpoint runs\stage2b_sakd_%%R_seed%SEED%\checkpoints\student_best.pt ^
    --out runs\external_coughvid_test_stage2b_sakd_%%R_seed%SEED% ^
    --device %DEVICE% ^
    --batch-size %BATCH_SIZE%
  if errorlevel 1 exit /b 1
)

echo.
echo [5/6] Domain probe comparison
python -B scripts\domain_probe_students.py ^
  --coswara-manifest "%COSWARA_MANIFEST%" ^
  --coughvid-manifest "%COUGHVID_MANIFEST%" ^
  --root "%DATA_ROOT%" ^
  --run-dir "%BASELINE_RUN%" ^
  --out runs\domain_probe_stage2b_sakd_seed%SEED% ^
  --samples-per-dataset 1500 ^
  --seed %SEED% ^
  --batch-size %BATCH_SIZE% ^
  --device %DEVICE% ^
  --extra-checkpoint sakd_tempered_full=runs\stage2b_sakd_tempered_full_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint sakd_tempered_domain=runs\stage2b_sakd_tempered_domain_seed%SEED%\checkpoints\student_best.pt ^
  --extra-checkpoint sakd_mild_full=runs\stage2b_sakd_mild_full_seed%SEED%\checkpoints\student_best.pt
if errorlevel 1 exit /b 1

echo.
echo [6/6] Done
echo Review:
echo   runs\stage2b_sakd_tempered_full_seed%SEED%\RESULTS.md
echo   runs\stage2b_sakd_tempered_domain_seed%SEED%\RESULTS.md
echo   runs\stage2b_sakd_mild_full_seed%SEED%\RESULTS.md
echo   runs\domain_probe_stage2b_sakd_seed%SEED%\DOMAIN_PROBE_REPORT.md

endlocal
