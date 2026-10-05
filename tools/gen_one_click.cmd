@echo off
chcp 65001 >nul
title KP 本地出图器（内容只留在你这台机器）
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gen_cli.ps1"
exit /b
