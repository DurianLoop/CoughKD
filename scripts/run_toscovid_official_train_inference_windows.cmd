@echo off
setlocal

cd /d "%~dp0\.."

set "PY=D:\conda\envs\CoughKD\python.exe"
set "MODE=%~1"
set "CONFIRM=%~2"
if "%MODE%"=="" set "MODE=stable"
if /I not "%MODE%"=="stable" if /I not "%MODE%"=="full" (
  echo Usage: scripts\run_toscovid_official_train_inference_windows.cmd [stable^|full]
  echo stable: source_only, ce, kd only
  echo full: all router methods
  exit /b 1
)

if not exist "manifests\toscovid2021_train_external.csv" (
  echo Missing manifests\toscovid2021_train_external.csv
  echo Build manifests\toscovid2021_full_external.csv and filter split=train first.
  exit /b 1
)

echo This will run inference on 4,803 existing TosCOVID 2021 official-train audio rows.
echo Mode: %MODE%
echo It does not train new models and does not download new data.
if /I not "%CONFIRM%"=="yes" (
  echo Stop now if this has not been approved.
  pause
) else (
  echo Approval confirmation supplied by command argument: yes
)

echo [1/4] Running official-train inference for existing checkpoint set...
if /I "%MODE%"=="stable" (
  "%PY%" -B scripts\evaluate_external_model_set.py ^
    --manifest manifests\toscovid2021_train_external.csv ^
    --target-tag toscovid2021_train ^
    --root D:\CoughKD ^
    --device auto ^
    --batch-size 16 ^
    --skip-existing ^
    --methods source_only ce kd
) else (
  "%PY%" -B scripts\evaluate_external_model_set.py ^
    --manifest manifests\toscovid2021_train_external.csv ^
    --target-tag toscovid2021_train ^
    --root D:\CoughKD ^
    --device auto ^
    --batch-size 16 ^
    --skip-existing
)
if errorlevel 1 exit /b 1

echo [2/4] Refreshing official split readiness...
"%PY%" -B scripts\audit_toscovid_official_split_readiness.py
if errorlevel 1 exit /b 1

echo [3/4] Computing official train-calibration/test-evaluation semantic-router result...
"%PY%" -B scripts\audit_toscovid_official_split_semantic_router.py
if errorlevel 1 exit /b 1

echo [4/4] Refreshing semantic-router gates...
"%PY%" -B scripts\audit_semantic_router_submission_readiness.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\build_semantic_router_claim_dossier.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_semantic_router_goal_completion.py
if errorlevel 1 exit /b 1
"%PY%" -B scripts\audit_semantic_router_submission_readiness.py
if errorlevel 1 exit /b 1

echo Done. Review:
echo runs\toscovid_official_split_semantic_router\TOSCOVID_OFFICIAL_SPLIT_SEMANTIC_ROUTER.md
echo runs\semantic_router_claim_dossier\SEMANTIC_ROUTER_CLAIM_DOSSIER.md
echo runs\semantic_router_goal_completion\SEMANTIC_ROUTER_GOAL_COMPLETION.md
