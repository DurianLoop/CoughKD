@echo off
setlocal

rem Stage 3c domain-probe follow-up.
rem Run after scripts\run_target_consistency_stage3c_repeats_windows.cmd finishes.

cd /d "%~dp0\.."

set DATA_ROOT=D:\CoughKD
set DEVICE=auto
set BATCH_SIZE=16
set SOURCE_MANIFEST=runs\coswara_cough_filtered_split\manifest_split.csv
set TARGET_MANIFEST=manifests\coughvid_adapt_test.csv
set BASELINE_RUN=runs\stage1_panns_response_seed7

for %%S in (11 23) do (
  echo.
  echo ===== Stage 3c domain probe seed %%S =====
  python -B scripts\domain_probe_students.py ^
    --coswara-manifest "%SOURCE_MANIFEST%" ^
    --coughvid-manifest "%TARGET_MANIFEST%" ^
    --root "%DATA_ROOT%" ^
    --run-dir "%BASELINE_RUN%" ^
    --out runs\domain_probe_stage3c_tcd_seed%%S ^
    --samples-per-dataset 1500 ^
    --seed %%S ^
    --batch-size %BATCH_SIZE% ^
    --device %DEVICE% ^
    --extra-checkpoint source_only=runs\stage3c_source_only_seed%%S\checkpoints\student_best.pt ^
    --extra-checkpoint tcd_very_strong=runs\stage3c_tcd_very_strong_seed%%S\checkpoints\student_best.pt ^
    --extra-checkpoint tcd_conf035=runs\stage3c_tcd_conf035_seed%%S\checkpoints\student_best.pt
  if errorlevel 1 exit /b 1
)

echo.
echo Done. Review:
echo   runs\domain_probe_stage3c_tcd_seed11\DOMAIN_PROBE_REPORT.md
echo   runs\domain_probe_stage3c_tcd_seed23\DOMAIN_PROBE_REPORT.md

endlocal
