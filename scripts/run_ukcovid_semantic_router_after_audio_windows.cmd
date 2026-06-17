@echo off
setlocal

set ROOT=D:\CoughKD\AAAI
set PY=D:\conda\envs\CoughKD\python.exe
set DATA_DIR=D:\CoughKD\external\ukcovid_open
set MANIFEST=manifests\ukcovid_open_test_external.csv
set DRY_RUN=0

if /I "%~1"=="--dry-run" set DRY_RUN=1

cd /d "%ROOT%"
if errorlevel 1 exit /b 1

if "%DRY_RUN%"=="1" (
  echo DRY-RUN: commands will be printed but not executed.
)

echo.
echo [1/16] Checking UKCOVID audio readiness
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\check_ukcovid_audio_ready.py --dataset-dir "%DATA_DIR%" --manifest "%MANIFEST%"
) else (
  "%PY%" -B scripts\check_ukcovid_audio_ready.py --dataset-dir "%DATA_DIR%" --manifest "%MANIFEST%"
)
if errorlevel 1 goto fail

echo.
echo [2/16] Verifying UKCOVID audio verdict
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_ready_summary.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_EVALUATION' else 2)"
) else (
  "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_ready_summary.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_EVALUATION' else 2)"
)
if errorlevel 2 goto not_ready
if errorlevel 1 goto fail

echo.
echo [3/16] Rebuilding UKCOVID test manifest with recursive audio search
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\build_ukcovid_manifest.py --audio-metadata "%DATA_DIR%\audio_metadata.csv" --participant-metadata "%DATA_DIR%\participant_metadata.csv" --splits "%DATA_DIR%\train_test_splits.csv" --dataset-dir "%DATA_DIR%" --out "%MANIFEST%" --keep-split-values test --recursive-audio-search
) else (
  "%PY%" -B scripts\build_ukcovid_manifest.py ^
    --audio-metadata "%DATA_DIR%\audio_metadata.csv" ^
    --participant-metadata "%DATA_DIR%\participant_metadata.csv" ^
    --splits "%DATA_DIR%\train_test_splits.csv" ^
    --dataset-dir "%DATA_DIR%" ^
    --out "%MANIFEST%" ^
    --keep-split-values test ^
    --recursive-audio-search
)
if errorlevel 1 goto fail

echo.
echo [4/16] Running Coswara-vs-UKCOVID overlap audit
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_manifest_overlap.py --source-manifest manifests\coswara_cough.csv --target-manifest "%MANIFEST%" --out runs\overlap_audit\coswara_vs_ukcovid_open_test
) else (
  "%PY%" -B scripts\audit_manifest_overlap.py ^
    --source-manifest manifests\coswara_cough.csv ^
    --target-manifest "%MANIFEST%" ^
    --out runs\overlap_audit\coswara_vs_ukcovid_open_test
)
if errorlevel 1 goto fail

echo.
echo [5/16] Onboarding UKCOVID with the existing external model set
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\onboard_external_target.py --manifest "%MANIFEST%" --target-tag ukcovid_open --skip-existing --device auto --batch-size 16
) else (
  "%PY%" -B scripts\onboard_external_target.py ^
    --manifest "%MANIFEST%" ^
    --target-tag ukcovid_open ^
    --skip-existing ^
    --device auto ^
    --batch-size 16
)
if errorlevel 1 goto fail

echo.
echo [6/16] Checking semantic-router target readiness
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\check_semantic_router_target_ready.py --target UKCOVID
) else (
  "%PY%" -B scripts\check_semantic_router_target_ready.py --target UKCOVID
)
if errorlevel 1 goto fail

echo.
echo [7/16] Verifying semantic-router readiness verdict
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/semantic_router_third_target_readiness/ukcovid_semantic_router_readiness.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_SEMANTIC_ROUTER_AUDIT' else 2)"
) else (
  "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/semantic_router_third_target_readiness/ukcovid_semantic_router_readiness.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_SEMANTIC_ROUTER_AUDIT' else 2)"
)
if errorlevel 2 goto semantic_not_ready
if errorlevel 1 goto fail

echo.
echo [8/16] Running 1000-repeat semantic-router audit on COUGHVID, TosCOVID, and UKCOVID
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py --targets COUGHVID TosCOVID UKCOVID --n-repeats 1000 --inner-guard-margin 0.01 --group-split-column subject_id --out runs\semantic_constrained_transfer_router_with_ukcovid_1000
) else (
  "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py ^
    --targets COUGHVID TosCOVID UKCOVID ^
    --n-repeats 1000 ^
    --inner-guard-margin 0.01 ^
    --group-split-column subject_id ^
    --out runs\semantic_constrained_transfer_router_with_ukcovid_1000
)
if errorlevel 1 goto fail

echo.
echo [9/16] Running UKCOVID demographic-only negative control
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py --targets UKCOVID --n-repeats 1000 --group-split-column subject_id --semantic-slice-columns age sex --out runs\semantic_router_ukcovid_inverted_demographic_1000
) else (
  "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py ^
    --targets UKCOVID ^
    --n-repeats 1000 ^
    --group-split-column subject_id ^
    --semantic-slice-columns age sex ^
    --out runs\semantic_router_ukcovid_inverted_demographic_1000
)
if errorlevel 1 goto fail

echo.
echo [10/16] Running UKCOVID all-slice negative control
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py --targets UKCOVID --n-repeats 1000 --group-split-column subject_id --semantic-slice-columns symptom_cough_any symptom_fatigue symptom_headache symptom_onset symptom_none age sex --out runs\semantic_router_ukcovid_all_slice_1000
) else (
  "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py ^
    --targets UKCOVID ^
    --n-repeats 1000 ^
    --group-split-column subject_id ^
    --semantic-slice-columns symptom_cough_any symptom_fatigue symptom_headache symptom_onset symptom_none age sex ^
    --out runs\semantic_router_ukcovid_all_slice_1000
)
if errorlevel 1 goto fail

echo.
echo [11/16] Running UKCOVID no-slice negative control
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py --targets UKCOVID --n-repeats 1000 --group-split-column subject_id --semantic-slice-columns none --out runs\semantic_router_ukcovid_no_slice_1000
) else (
  "%PY%" -B scripts\audit_semantic_constrained_transfer_router.py ^
    --targets UKCOVID ^
    --n-repeats 1000 ^
    --group-split-column subject_id ^
    --semantic-slice-columns none ^
    --out runs\semantic_router_ukcovid_no_slice_1000
)
if errorlevel 1 goto fail

echo.
echo [12/16] Running pre-registered UKCOVID third-target success gate
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_router_third_target_success.py
) else (
  "%PY%" -B scripts\audit_semantic_router_third_target_success.py
)
if errorlevel 1 goto fail

echo.
echo [13/16] Verifying UKCOVID third-target success verdict
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/semantic_router_third_target_success/semantic_router_third_target_success.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'THIRD_TARGET_SUPPORTS_CLAIM' else 2)"
) else (
  "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/semantic_router_third_target_success/semantic_router_third_target_success.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'THIRD_TARGET_SUPPORTS_CLAIM' else 2)"
)
if errorlevel 2 goto third_target_not_support
if errorlevel 1 goto fail

echo.
echo [14/16] Refreshing semantic-router submission readiness gate
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_router_submission_readiness.py
) else (
  "%PY%" -B scripts\audit_semantic_router_submission_readiness.py
)
if errorlevel 1 goto fail

echo.
echo [15/16] Rebuilding semantic-router claim dossier
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\build_semantic_router_claim_dossier.py
) else (
  "%PY%" -B scripts\build_semantic_router_claim_dossier.py
)
if errorlevel 1 goto fail

echo.
echo [16/16] Refreshing semantic-router submission readiness gate after dossier update
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\audit_semantic_router_submission_readiness.py
) else (
  "%PY%" -B scripts\audit_semantic_router_submission_readiness.py
)
if errorlevel 1 goto fail

echo.
if "%DRY_RUN%"=="1" (
  echo UKCOVID semantic-router dry-run preview complete. No files were evaluated beyond printing commands.
) else (
  echo UKCOVID semantic-router post-download validation complete.
  echo Review:
  echo   runs\semantic_constrained_transfer_router_with_ukcovid_1000
  echo   runs\semantic_router_ukcovid_inverted_demographic_1000
  echo   runs\semantic_router_ukcovid_all_slice_1000
  echo   runs\semantic_router_ukcovid_no_slice_1000
  echo   runs\semantic_router_third_target_success\SEMANTIC_ROUTER_THIRD_TARGET_SUCCESS.md
  echo   runs\semantic_router_claim_dossier\SEMANTIC_ROUTER_CLAIM_DOSSIER.md
  echo   runs\semantic_router_submission_readiness\SEMANTIC_ROUTER_SUBMISSION_READINESS.md
)
exit /b 0

:not_ready
echo.
echo UKCOVID audio is not ready yet. Download and extract the split archive first, then rerun this script.
echo Check:
echo   runs\ukcovid_open_metadata_audit\UKCOVID_AUDIO_READY_CHECK.md
exit /b 2

:semantic_not_ready
echo.
echo UKCOVID semantic-router target readiness failed. Inspect the readiness report before running 1000-repeat audits.
echo Check:
echo   runs\semantic_router_third_target_readiness\UKCOVID_SEMANTIC_ROUTER_READINESS.md
exit /b 3

:third_target_not_support
echo.
echo UKCOVID third-target success gate did not support the current claim. Do not upgrade the ICASSP claim without inspecting the success-gate report.
echo Check:
echo   runs\semantic_router_third_target_success\SEMANTIC_ROUTER_THIRD_TARGET_SUCCESS.md
exit /b 4

:fail
echo.
echo UKCOVID semantic-router post-download validation failed. Inspect the command output above and rerun after fixing the issue.
exit /b 1
