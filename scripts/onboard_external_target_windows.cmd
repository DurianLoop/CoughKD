@echo off
setlocal

REM Usage:
REM   scripts\onboard_external_target_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]

cd /d "%~dp0\.."

if "%~1"=="" (
  echo Missing manifest path.
  echo Usage: scripts\onboard_external_target_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]
  exit /b 2
)

if "%~2"=="" (
  echo Missing target tag.
  echo Usage: scripts\onboard_external_target_windows.cmd manifests\TARGET_external.csv TARGET [--dry-run]
  exit /b 2
)

set PYTHON_EXE=D:\conda\envs\CoughKD\python.exe
if not exist "%PYTHON_EXE%" (
  set PYTHON_EXE=python
)

set NUMBA_CACHE_DIR=%CD%\.numba_cache

"%PYTHON_EXE%" scripts\onboard_external_target.py --manifest "%~1" --target-tag "%~2" --skip-existing %~3
exit /b %errorlevel%
