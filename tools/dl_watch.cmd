@echo off
chcp 65001 >nul
title KP 下载监视（可随时关掉，不影响下载）
cd /d "%~dp0.."

echo ============================================================
echo   KP 下载监视窗口
echo ============================================================
echo.
echo   这个窗口**只看不下载**。
echo   关掉它，下载照常进行；再双击本文件可重新接上看。
echo.
echo   如果你还没开始下载，请另开一个窗口运行：tools\dl_run.cmd
echo.
echo ------------------------------------------------------------

.venv\Scripts\python.exe -u -m kp.data.fetch_tui --tui

echo.
echo 监视窗口已退出。
pause
