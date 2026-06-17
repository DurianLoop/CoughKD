@echo off
setlocal

set ROOT=D:\CoughKD\AAAI
set PY=D:\conda\envs\CoughKD\python.exe
set MODEL_DIR=%ROOT%\pretrained\teachers\hear_pytorch

cd /d "%ROOT%"
if errorlevel 1 exit /b 1

echo [1/3] Checking HeAR PyTorch preflight...
"%PY%" scripts\audit_hear_pytorch_embedding_upper_bound.py --preflight-only --model-path "%MODEL_DIR%"
if errorlevel 1 goto fail

"%PY%" -B -c "import json, pathlib, sys; p=pathlib.Path('runs/hear_pytorch_embedding_upper_bound_seed7/hear_pytorch_embedding_upper_bound_audit.json'); d=json.loads(p.read_text(encoding='utf-8')); pre=d.get('preflight', {}); print('asset_present=', pre.get('asset_present'), 'environment_ready=', pre.get('environment_ready')); sys.exit(0 if pre.get('asset_present') and pre.get('environment_ready') else 2)"
if errorlevel 2 goto not_ready
if errorlevel 1 goto fail

echo [2/3] Running HeAR frozen embedding Tos gate...
"%PY%" scripts\audit_hear_pytorch_embedding_upper_bound.py --model-path "%MODEL_DIR%" --device auto --batch-size 8
if errorlevel 1 goto fail

echo [3/3] Gate complete.
echo Review:
echo   runs\hear_pytorch_embedding_upper_bound_seed7\HEAR_PYTORCH_EMBEDDING_UPPER_BOUND_AUDIT.md
echo   runs\hear_pytorch_embedding_upper_bound_seed7\hear_pytorch_embedding_upper_bound_audit.json
exit /b 0

:not_ready
echo.
echo HeAR PyTorch is not ready. Accept the gated terms and place/download google/hear-pytorch under:
echo   %MODEL_DIR%
echo Then rerun this script.
exit /b 2

:fail
echo.
echo HeAR gate failed. Inspect the output above and rerun after fixing the issue.
exit /b 1
