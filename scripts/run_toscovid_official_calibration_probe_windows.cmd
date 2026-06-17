@echo off
setlocal

cd /d "%~dp0\.."

set "PY=D:\conda\envs\CoughKD\python.exe"
set "CALIB=%~1"
set "MODE=%~2"
set "CONFIRM=%~3"
set "RUN_MODE=%~4"
if "%CALIB%"=="" set "CALIB=calib10"
if "%MODE%"=="" set "MODE=stable"
if "%RUN_MODE%"=="" set "RUN_MODE=run"
set "DRYRUN_ARG="

if /I not "%MODE%"=="stable" if /I not "%MODE%"=="full" (
  echo Usage: scripts\run_toscovid_official_calibration_probe_windows.cmd [calib10^|calib20^|calib30^|calib50^|full_train] [stable^|full] [yes] [run^|dryrun]
  exit /b 1
)
if /I not "%RUN_MODE%"=="run" if /I not "%RUN_MODE%"=="dryrun" (
  echo Usage: scripts\run_toscovid_official_calibration_probe_windows.cmd [calib10^|calib20^|calib30^|calib50^|full_train] [stable^|full] [yes] [run^|dryrun]
  exit /b 1
)
if /I "%RUN_MODE%"=="dryrun" set "DRYRUN_ARG=--dry-run"

set "MANIFEST="
set "TARGET_TAG="
if /I "%CALIB%"=="calib10" (
  set "MANIFEST=manifests\toscovid2021_train_calib10_external.csv"
  set "TARGET_TAG=toscovid2021_train_calib10"
)
if /I "%CALIB%"=="calib20" (
  set "MANIFEST=manifests\toscovid2021_train_calib20_external.csv"
  set "TARGET_TAG=toscovid2021_train_calib20"
)
if /I "%CALIB%"=="calib30" (
  set "MANIFEST=manifests\toscovid2021_train_calib30_external.csv"
  set "TARGET_TAG=toscovid2021_train_calib30"
)
if /I "%CALIB%"=="calib50" (
  set "MANIFEST=manifests\toscovid2021_train_calib50_external.csv"
  set "TARGET_TAG=toscovid2021_train_calib50"
)
if /I "%CALIB%"=="full_train" (
  set "MANIFEST=manifests\toscovid2021_train_external.csv"
  set "TARGET_TAG=toscovid2021_train"
)

if "%MANIFEST%"=="" (
  echo Unknown calibration subset: %CALIB%
  echo Usage: scripts\run_toscovid_official_calibration_probe_windows.cmd [calib10^|calib20^|calib30^|calib50^|full_train] [stable^|full] [yes] [run^|dryrun]
  exit /b 1
)

if not exist "%MANIFEST%" (
  echo Missing %MANIFEST%
  echo Run scripts\build_toscovid_official_calibration_subsets.py first.
  exit /b 1
)

echo This will run TosCOVID official-train calibration probe inference.
echo Calibration: %CALIB%
echo Manifest: %MANIFEST%
echo Mode: %MODE%
echo Run mode: %RUN_MODE%
echo stable = source_only, ce, kd only; full = all router methods.
echo It does not train new models and does not download new data.
if /I not "%CONFIRM%"=="yes" (
  echo Stop now if this has not been approved.
  pause
) else (
  echo Approval confirmation supplied by command argument: yes
)

if /I "%MODE%"=="stable" (
  "%PY%" -B scripts\evaluate_external_model_set.py ^
    --manifest "%MANIFEST%" ^
    --target-tag "%TARGET_TAG%" ^
    --root D:\CoughKD ^
    --device auto ^
    --batch-size 16 ^
    --skip-existing ^
    --methods source_only ce kd ^
    %DRYRUN_ARG%
) else (
  "%PY%" -B scripts\evaluate_external_model_set.py ^
    --manifest "%MANIFEST%" ^
    --target-tag "%TARGET_TAG%" ^
    --root D:\CoughKD ^
    --device auto ^
    --batch-size 16 ^
    --skip-existing ^
    %DRYRUN_ARG%
)
if errorlevel 1 exit /b 1

if /I "%RUN_MODE%"=="dryrun" (
  echo Dry-run complete. Skipping gate refresh because no predictions were written.
  exit /b 0
)

echo Refreshing TosCOVID official calibration budget gate...
"%PY%" -B scripts\audit_toscovid_official_calibration_budget.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_toscovid_calib10_result_decision.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_toscovid_calib10_decision_selftest.py
if errorlevel 1 exit /b 1

echo Refreshing semantic-router gates...
"%PY%" -B scripts\summarize_semantic_router_local_only_innovation_queue.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\summarize_semantic_router_branch_controller.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_semantic_router_submission_readiness.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\build_semantic_router_claim_dossier.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_semantic_router_goal_completion.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_semantic_router_submission_readiness.py
if errorlevel 1 exit /b 1

echo Done. Review:
echo runs\toscovid_official_calibration_budget\TOSCOVID_OFFICIAL_CALIBRATION_BUDGET.md
echo runs\semantic_router_goal_completion\SEMANTIC_ROUTER_GOAL_COMPLETION.md
