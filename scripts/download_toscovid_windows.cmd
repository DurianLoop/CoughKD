@echo off
setlocal

cd /d "%~dp0\.."

set "DATA_DIR=D:\CoughKD\external\tos_covid19"
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo [1/3] Downloading Tos COVID-19 metadata...
curl.exe -L -C - --fail --ssl-no-revoke --http1.1 --retry 20 --retry-all-errors --retry-delay 3 --connect-timeout 60 ^
  -o "%DATA_DIR%\tos-covid-19.csv" ^
  "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-salud/tos-covid-19/tos-covid-19.csv"
if errorlevel 1 exit /b 1

curl.exe -L -C - --fail --ssl-no-revoke --http1.1 --retry 20 --retry-all-errors --retry-delay 3 --connect-timeout 60 ^
  -o "%DATA_DIR%\tos-covid-2022.csv" ^
  "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-salud/tos-covid-19/tos-covid-2022.csv"
if errorlevel 1 exit /b 1

echo [2/3] Downloading Tos COVID-19 2021 audio ZIP with byte ranges...
D:\conda\envs\CoughKD\python.exe -B scripts\download_range_file.py ^
  --url "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-salud/tos-covid-19/tos-covid-19.zip" ^
  --out "%DATA_DIR%\tos-covid-19.zip" ^
  --size 60105734 ^
  --part-size 2097152 ^
  --jobs 4 ^
  --retries 8
if errorlevel 1 exit /b 1

echo [3/3] Done.
dir "%DATA_DIR%\tos-covid-19.csv" "%DATA_DIR%\tos-covid-2022.csv" "%DATA_DIR%\tos-covid-19.zip"
