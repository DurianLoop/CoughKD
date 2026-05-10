@echo off
setlocal

cd /d %~dp0

set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set CONDA_PKGS_DIRS=%CD%\.conda\pkgs
set CONDA_ENVS_PATH=%CD%\.conda\envs

conda create -y -p .conda\coughkd-aaai python=3.11 pip
if errorlevel 1 exit /b 1

set PYEXE=%CD%\.conda\coughkd-aaai\python.exe
%PYEXE% -m pip install --upgrade pip
%PYEXE% -m pip install -r requirements-ml.txt
%PYEXE% -m pip install librosa soundfile scipy scikit-learn pandas matplotlib seaborn tqdm

set PYTHONPATH=%CD%\src
%PYEXE% -m unittest discover -s tests
