@echo off
setlocal

set ROOT=D:\CoughKD\AAAI
set PY=D:\conda\envs\CoughKD\python.exe
set DATA_DIR=D:\CoughKD\external\ukcovid_open

cd /d "%ROOT%"
if errorlevel 1 exit /b 1

echo [1/5] Checking UK COVID-19 audio readiness...
"%PY%" scripts\check_ukcovid_audio_ready.py --dataset-dir "%DATA_DIR%" --manifest manifests\ukcovid_open_test_external.csv
if errorlevel 1 goto fail

"%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_ready_summary.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_EVALUATION' else 2)"
if errorlevel 2 goto not_ready
if errorlevel 1 goto fail

echo [2/5] Rebuilding formal UK test manifest with recursive audio search...
"%PY%" scripts\build_ukcovid_manifest.py ^
  --audio-metadata "%DATA_DIR%\audio_metadata.csv" ^
  --participant-metadata "%DATA_DIR%\participant_metadata.csv" ^
  --splits "%DATA_DIR%\train_test_splits.csv" ^
  --dataset-dir "%DATA_DIR%" ^
  --out manifests\ukcovid_open_test_external.csv ^
  --keep-split-values test ^
  --recursive-audio-search
if errorlevel 1 goto fail

echo [3/5] Running overlap audit...
"%PY%" scripts\audit_manifest_overlap.py ^
  --source-manifest manifests\coswara_cough.csv ^
  --target-manifest manifests\ukcovid_open_test_external.csv ^
  --out runs\overlap_audit\coswara_vs_ukcovid_open_test
if errorlevel 1 goto fail

echo [4/5] Onboarding UK external target with existing model set...
"%PY%" scripts\onboard_external_target.py ^
  --manifest manifests\ukcovid_open_test_external.csv ^
  --target-tag ukcovid_open ^
  --skip-existing ^
  --device auto ^
  --batch-size 16
if errorlevel 1 goto fail

echo [5/5] Refreshing claim and submission audits...
"%PY%" scripts\audit_claim_boundary.py
if errorlevel 1 goto fail
"%PY%" scripts\audit_submission_readiness.py
if errorlevel 1 goto fail

echo.
echo UK COVID-19 external onboarding complete.
echo Review:
echo   runs\submission_readiness\SUBMISSION_READINESS.md
echo   runs\kd_failure_analysis\SHIFT_AUDIT_MULTITARGET_TABLE.md
exit /b 0

:not_ready
echo.
echo UK audio is not ready yet. Download/extract the split archive first, then rerun this script.
echo Check:
echo   runs\ukcovid_open_metadata_audit\UKCOVID_AUDIO_READY_CHECK.md
exit /b 2

:fail
echo.
echo UK onboarding failed. Inspect the output above and rerun after fixing the issue.
exit /b 1
