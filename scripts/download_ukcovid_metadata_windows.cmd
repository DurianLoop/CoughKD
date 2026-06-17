@echo off
setlocal

set OUT_DIR=%~1
if "%OUT_DIR%"=="" set OUT_DIR=D:\CoughKD\external\ukcovid_open

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo Downloading UK COVID-19 Vocal Audio Dataset metadata to:
echo   %OUT_DIR%
echo.

set CURL=curl.exe
set RETRY=--retry 8 --retry-all-errors --retry-delay 5 --connect-timeout 30 --max-time 600
set TLS=--ssl-no-revoke --http1.1

if exist "%OUT_DIR%\audio_metadata.csv" (
  echo [skip] audio_metadata.csv already exists
) else (
  %CURL% -L --fail %TLS% %RETRY% --continue-at - -o "%OUT_DIR%\audio_metadata.csv" "https://zenodo.org/records/10043978/files/audio_metadata.csv?download=1"
  if errorlevel 1 goto fail
)

if exist "%OUT_DIR%\participant_metadata.csv" (
  echo [skip] participant_metadata.csv already exists
) else (
  %CURL% -L --fail %TLS% %RETRY% --continue-at - -o "%OUT_DIR%\participant_metadata.csv" "https://zenodo.org/records/10043978/files/participant_metadata.csv?download=1"
  if errorlevel 1 goto fail
)

if exist "%OUT_DIR%\train_test_splits.csv" (
  echo [skip] train_test_splits.csv already exists
) else (
  %CURL% -L --fail %TLS% %RETRY% --continue-at - -o "%OUT_DIR%\train_test_splits.csv" "https://zenodo.org/records/10043978/files/train_test_splits.csv?download=1"
  if errorlevel 1 goto fail
)

if exist "%OUT_DIR%\README.md" (
  echo [skip] README.md already exists
) else (
  %CURL% -L --fail %TLS% %RETRY% --continue-at - -o "%OUT_DIR%\README.md" "https://zenodo.org/records/10043978/files/README.md?download=1"
  if errorlevel 1 goto fail
)

echo.
echo Metadata download complete.
echo Next:
echo   D:\conda\envs\CoughKD\python.exe scripts\build_ukcovid_manifest.py --audio-metadata "%OUT_DIR%\audio_metadata.csv" --participant-metadata "%OUT_DIR%\participant_metadata.csv" --splits "%OUT_DIR%\train_test_splits.csv" --dataset-dir "%OUT_DIR%" --out manifests\ukcovid_open_external.csv --allow-missing-audio
exit /b 0

:fail
echo.
echo Download failed. If you use a local proxy, run:
echo   set HTTPS_PROXY=http://127.0.0.1:7897
echo   set HTTP_PROXY=http://127.0.0.1:7897
echo then retry this script.
exit /b 1
