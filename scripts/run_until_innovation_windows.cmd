@echo off
setlocal

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
if not exist ".numba_cache" mkdir ".numba_cache"
set NUMBA_CACHE_DIR=%CD%\.numba_cache

set PYTHON_EXE=python
if exist "D:\conda\envs\CoughKD\python.exe" (
  set PYTHON_EXE=D:\conda\envs\CoughKD\python.exe
)

if not exist "runs\coswara_cough_filtered_split\manifest_split.csv" (
  echo Missing Coswara manifest: runs\coswara_cough_filtered_split\manifest_split.csv
  exit /b 1
)

if not exist "manifests\coughvid_adapt_test.csv" (
  echo Missing COUGHVID manifest: manifests\coughvid_adapt_test.csv
  exit /b 1
)

if not exist "pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth" (
  echo Missing PANNs checkpoint: pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth
  exit /b 1
)

"%PYTHON_EXE%" -B scripts\innovation_runner.py %*
if errorlevel 1 exit /b 1

echo.
echo Done. Review:
echo   docs\progress\5.29_autonomous_innovation_loop.md
echo   runs\innovation_loop_summary.json

endlocal
