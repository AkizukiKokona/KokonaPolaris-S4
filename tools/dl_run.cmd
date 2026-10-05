@echo off
chcp 65001 >nul
title KP 数据下载（双击运行）
cd /d "%~dp0.."

echo ============================================================
echo   KP 数据下载器
echo ============================================================
echo.
echo   这个窗口会显示：速度 / 进度 / 剩余时间
echo.
echo   下载位置：out\data\curated_danbooru\_shards
echo   支持断点续传 —— 中断后再双击本文件即可继续
echo.
echo   要下几个分片？（1 片约 1.1GB，建议先下 4 片）
echo.
set /p N=分片数量 [4]:

if "%N%"=="" set N=4

echo.
echo 正在启动代理环境...
set https_proxy=http://127.0.0.1:7897
set http_proxy=http://127.0.0.1:7897

echo 开始下载 %N% 个分片...
echo ============================================================
echo.

.venv\Scripts\python.exe -u -m kp.data.fetch_tui --serve --max-shards %N%

echo.
echo ============================================================
echo   下载结束。图片在：out\local_gen 之类的输出目录
echo   分片在：out\data\curated_danbooru\_shards
echo ============================================================
pause
