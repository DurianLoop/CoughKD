@echo off
setlocal

set ROOT=D:\CoughKD\AAAI
set PY=D:\conda\envs\CoughKD\python.exe
set DATA_DIR=D:\CoughKD\external\ukcovid_open
set DRY_RUN=0

if /I "%~1"=="--dry-run" set DRY_RUN=1

cd /d "%ROOT%"
if errorlevel 1 exit /b 1

if "%DRY_RUN%"=="1" (
  echo DRY-RUN: commands will be printed but not executed.
)

echo.
echo [1/6] Running no-download UKCOVID audio preflight
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\preflight_ukcovid_audio_download.py --dataset-dir "%DATA_DIR%"
) else (
  "%PY%" -B scripts\preflight_ukcovid_audio_download.py --dataset-dir "%DATA_DIR%"
)
if errorlevel 1 goto fail

echo.
echo [2/6] Verifying preflight verdict
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_download_preflight.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_TO_DOWNLOAD_OR_RESUME' else 2)"
) else (
  "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_download_preflight.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_TO_DOWNLOAD_OR_RESUME' else 2)"
)
if errorlevel 2 goto preflight_not_ready
if errorlevel 1 goto fail

echo.
echo [3/6] Downloading or resuming UKCOVID split audio archive
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: set HTTPS_PROXY=http://127.0.0.1:7897
  echo DRY-RUN: set HTTP_PROXY=http://127.0.0.1:7897
  echo DRY-RUN: runs\ukcovid_open_metadata_audit\download_ukcovid_audio_archive_windows.cmd "%DATA_DIR%"
) else (
  set HTTPS_PROXY=http://127.0.0.1:7897
  set HTTP_PROXY=http://127.0.0.1:7897
  runs\ukcovid_open_metadata_audit\download_ukcovid_audio_archive_windows.cmd "%DATA_DIR%"
)
if errorlevel 1 goto fail

echo.
echo [4/6] Verifying UKCOVID split archive size and checksum
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\verify_ukcovid_audio_archive.py --dataset-dir "%DATA_DIR%"
) else (
  "%PY%" -B scripts\verify_ukcovid_audio_archive.py --dataset-dir "%DATA_DIR%"
)
if errorlevel 1 goto fail

echo.
echo [5/6] Verifying archive integrity verdict
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_archive_integrity.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_EXTRACTION' else 2)"
) else (
  "%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/ukcovid_open_metadata_audit/ukcovid_audio_archive_integrity.json'); d=json.loads(p.read_text(encoding='utf-8')); print('verdict=', d.get('verdict')); sys.exit(0 if d.get('verdict') == 'READY_FOR_EXTRACTION' else 2)"
)
if errorlevel 2 goto archive_not_ready
if errorlevel 1 goto fail

echo.
echo [6/6] Checking extraction preflight
if "%DRY_RUN%"=="1" (
  echo DRY-RUN: "%PY%" -B scripts\check_ukcovid_extraction_preflight.py --dataset-dir "%DATA_DIR%"
) else (
  "%PY%" -B scripts\check_ukcovid_extraction_preflight.py --dataset-dir "%DATA_DIR%"
)
if errorlevel 1 goto fail

echo.
echo Next manual step:
echo   Inspect runs\ukcovid_open_metadata_audit\UKCOVID_EXTRACTION_PREFLIGHT.md
echo Then extract the verified split archive and run:
echo   scripts\run_ukcovid_semantic_router_after_audio_windows.cmd
exit /b 0

:preflight_not_ready
echo.
echo UKCOVID audio download preflight is not ready. Inspect:
echo   runs\ukcovid_open_metadata_audit\UKCOVID_AUDIO_DOWNLOAD_PREFLIGHT.md
exit /b 2

:archive_not_ready
echo.
echo UKCOVID split archive is not ready for extraction. Inspect:
echo   runs\ukcovid_open_metadata_audit\UKCOVID_AUDIO_ARCHIVE_INTEGRITY.md
exit /b 3

:fail
echo.
echo UKCOVID audio download/verify workflow failed. Inspect the command output above.
exit /b 1
