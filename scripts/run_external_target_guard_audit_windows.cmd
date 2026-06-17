@echo off
setlocal

REM Usage:
REM   scripts\run_external_target_guard_audit_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]

cd /d "%~dp0\.."

if "%~1"=="" (
  echo Missing manifest path.
  echo Usage: scripts\run_external_target_guard_audit_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]
  exit /b 2
)

if "%~2"=="" (
  echo Missing target tag.
  echo Usage: scripts\run_external_target_guard_audit_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]
  exit /b 2
)

set MANIFEST=%~1
set TARGET_TAG=%~2
set DRY_RUN=%~3
set PYTHON_EXE=D:\conda\envs\CoughKD\python.exe

if not exist "%PYTHON_EXE%" (
  set PYTHON_EXE=python
)

set NUMBA_CACHE_DIR=%CD%\.numba_cache

echo [1/2] Evaluating checkpoints on %TARGET_TAG%
if "%DRY_RUN%"=="--dry-run" (
  "%PYTHON_EXE%" scripts\evaluate_external_model_set.py --manifest "%MANIFEST%" --target-tag "%TARGET_TAG%" --skip-existing --dry-run
) else (
  "%PYTHON_EXE%" scripts\evaluate_external_model_set.py --manifest "%MANIFEST%" --target-tag "%TARGET_TAG%" --skip-existing
)
if errorlevel 1 exit /b %errorlevel%

echo [2/2] Summarizing CoughKD-Guard multi-target audit
if not "%DRY_RUN%"=="--dry-run" (
  "%PYTHON_EXE%" scripts\summarize_coughkd_guard_multitarget.py
  if errorlevel 1 exit /b %errorlevel%
)

echo Done. Review runs\coughkd_guard_multitarget\COUGHKD_GUARD_MULTITARGET_AUDIT.md
