@echo off
chcp 65001 >nul
title KP Download
cd /d "%~dp0.."

echo ============================================================
echo   KP Download
echo ============================================================
echo.
echo   Progress / speed / ETA show in this window.
echo   Shards: out\data\curated_danbooru\_shards
echo   Resumable - run this file again to continue.
echo.
set /p N=How many shards? [4]:

if "%N%"=="" set N=4

echo.
set https_proxy=http://127.0.0.1:7897
set http_proxy=http://127.0.0.1:7897
set PYTHONIOENCODING=utf-8

echo Downloading %N% shards...
echo ============================================================
echo.

.venv\Scripts\python.exe -u -m kp.data.fetch_tui --serve --max-shards %N%

echo.
echo ============================================================
echo   Done. Shards in: out\data\curated_danbooru\_shards
echo ============================================================
pause
