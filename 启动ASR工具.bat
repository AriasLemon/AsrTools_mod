@echo off
chcp 65001 >nul
title ASR字幕工具
cd /d "%~dp0"

where pythonw >nul 2>&1
if %errorlevel% equ 0 (
    start "" pythonw asr_gui.py
) else (
    start "" python asr_gui.py
)
