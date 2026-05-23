@echo off
setlocal

rem Stage 1: reproduce inherited Coswara baseline and run first KD ablations.
rem Run from: D:\CoughKD\AAAI
rem Assumes the conda environment CoughKD already exists.

cd /d "%~dp0\.."

set PYTHONPATH=%CD%\src
set DATA_ROOT=D:\CoughKD
set METADATA=D:\CoughKD\datasets\Coswara-Data-master\combined_data.csv
set EXTRACTED_DIR=coswara_extracted2
set DEVICE=auto
set BATCH_SIZE=16
set TEACHER_EPOCHS=8
set STUDENT_EPOCHS=8
set SEED=7

if not exist "%DATA_ROOT%" (
  echo Missing Coswara dataset root: %DATA_ROOT%
  echo Edit DATA_ROOT in this script or download Coswara first.
  exit /b 1
)

if not exist "%METADATA%" (
  echo Missing Coswara metadata: %METADATA%
  echo Edit METADATA in this script or prepare combined_data.csv first.
  exit /b 1
)

python scripts\check_assets.py --coswara-root "%DATA_ROOT%" --coswara-metadata "%METADATA%"

if not exist manifests mkdir manifests
if not exist runs mkdir runs

echo.
echo [1/6] Build Coswara cough manifest
python -m coughkd.cli build-coswara-manifest ^
  --root "%DATA_ROOT%" ^
  --metadata "%METADATA%" ^
  --out manifests\coswara_cough.csv ^
  --extracted-dir "%EXTRACTED_DIR%" ^
  --recording-types cough-heavy,cough-shallow
if errorlevel 1 exit /b 1

echo.
echo [2/6] Filter manifest
python -m coughkd.cli filter-manifest ^
  --manifest manifests\coswara_cough.csv ^
  --root "%DATA_ROOT%" ^
  --out runs\coswara_cough_filtered ^
  --min-duration-sec 0.5 ^
  --drop-labels under_validation
if errorlevel 1 exit /b 1

echo.
echo [3/6] Subject-disjoint split
python -m coughkd.cli split-manifest ^
  --manifest runs\coswara_cough_filtered\manifest_filtered.csv ^
  --root "%DATA_ROOT%" ^
  --out runs\coswara_cough_filtered_split ^
  --seed %SEED%
if errorlevel 1 exit /b 1

echo.
echo [4/6] Pre-long check
python -m coughkd.cli prelong-check ^
  --manifest runs\coswara_cough_filtered_split\manifest_split.csv ^
  --root "%DATA_ROOT%" ^
  --out runs\prelong_coswara_stage1 ^
  --device %DEVICE%
if errorlevel 1 exit /b 1

echo.
echo [5/6] Compact teacher baseline with response KD off/on comparison
python -m coughkd.cli torch-train ^
  --manifest runs\coswara_cough_filtered_split\manifest_split.csv ^
  --root "%DATA_ROOT%" ^
  --out runs\stage1_compact_response_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind compact ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.0 ^
  --kd-embedding-weight 0.0 ^
  --kd-relation-weight 0.0
if errorlevel 1 exit /b 1

echo.
echo [6/6] Compact teacher multi-level KD probe
python -m coughkd.cli torch-train ^
  --manifest runs\coswara_cough_filtered_split\manifest_split.csv ^
  --root "%DATA_ROOT%" ^
  --out runs\stage1_compact_multilevel_seed%SEED% ^
  --device %DEVICE% ^
  --teacher-kind compact ^
  --teacher-epochs %TEACHER_EPOCHS% ^
  --student-epochs %STUDENT_EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --seed %SEED% ^
  --kd-temperature 2.0 ^
  --kd-response-weight 0.7 ^
  --kd-feature-weight 0.1 ^
  --kd-embedding-weight 0.1 ^
  --kd-relation-weight 0.1
if errorlevel 1 exit /b 1

echo.
echo Done. Review runs\stage1_compact_response_seed%SEED%\RESULTS.md and runs\stage1_compact_multilevel_seed%SEED%\RESULTS.md
endlocal
