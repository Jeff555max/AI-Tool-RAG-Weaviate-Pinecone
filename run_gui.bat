@echo off
title RAG Vector Store - AI Tool
echo ========================================
echo RAG Vector Store GUI Application
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [1/2] Creating virtual environment...
    python -m venv venv
    echo Done.
    echo.
)

REM Install dependencies
echo [2/2] Installing dependencies...
venv\Scripts\pip.exe install -q -r requirements.txt
echo Done.
echo.

echo Starting GUI...
echo.
venv\Scripts\pythonw.exe gui_app.py

if errorlevel 1 (
    echo.
    echo Error starting GUI. Press any key to exit.
    pause >nul
)
