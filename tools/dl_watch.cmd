@echo off
chcp 65001 >nul
title KP Download Monitor
cd /d "%~dp0.."

echo ============================================================
echo   KP Download Monitor
echo ============================================================
echo.
echo   This window ONLY watches. Closing it does NOT stop
echo   the download. Run it again to re-attach.
echo.
echo   To start downloading, double-click: tools\dl_run.cmd
echo.
echo ------------------------------------------------------------

.venv\Scripts\python.exe -u -m kp.data.fetch_tui --tui

echo.
echo Monitor closed.
pause
