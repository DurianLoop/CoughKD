@echo off
setlocal

cd /d "%~dp0\.."

set "PY=D:\conda\envs\CoughKD\python.exe"
set "DATA_DIR=D:\CoughKD\external\tos_covid19"
set "AUDIO_DIR=%DATA_DIR%\audio_2021"

if not exist "%DATA_DIR%\tos-covid-19.zip" (
  echo Missing %DATA_DIR%\tos-covid-19.zip
  echo Run scripts\download_toscovid_windows.cmd first.
  exit /b 1
)

if not exist "%AUDIO_DIR%\.extracted" (
  echo [1/4] Extracting Tos COVID-19 2021 audio...
  if not exist "%AUDIO_DIR%" mkdir "%AUDIO_DIR%"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%DATA_DIR%\tos-covid-19.zip' -DestinationPath '%AUDIO_DIR%' -Force"
  if errorlevel 1 exit /b 1
  type nul > "%AUDIO_DIR%\.extracted"
) else (
  echo [1/4] Audio already extracted.
)

echo [2/4] Building official-test manifest...
"%PY%" -B scripts\build_toscovid_manifest.py ^
  --metadata "%DATA_DIR%\tos-covid-19.csv" ^
  --batch-name 2021 ^
  --audio-root "%AUDIO_DIR%" ^
  --out manifests\toscovid2021_test_external.csv ^
  --report runs\toscovid_metadata_audit\TOSCOVID2021_TEST_METADATA_AUDIT.md ^
  --test-only
if errorlevel 1 exit /b 1

echo [3/4] Auditing source-target overlap...
"%PY%" -B scripts\audit_manifest_overlap.py ^
  --source-manifest manifests\coswara_cough.csv ^
  --target-manifest manifests\toscovid2021_test_external.csv ^
  --out runs\overlap_audit\coswara_vs_toscovid2021_test
if errorlevel 1 exit /b 1

echo [4/4] Running external onboarding/evaluation...
"%PY%" -B scripts\onboard_external_target.py ^
  --manifest manifests\toscovid2021_test_external.csv ^
  --target-tag toscovid2021_test ^
  --root D:\CoughKD ^
  --device auto ^
  --batch-size 16 ^
  --skip-existing
if errorlevel 1 exit /b 1

echo Done. Review:
echo runs\kd_failure_analysis\SHIFT_AUDIT_MULTITARGET_TABLE.md
echo runs\submission_readiness\SUBMISSION_READINESS.md

