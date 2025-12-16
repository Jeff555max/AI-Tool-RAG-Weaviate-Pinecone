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
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo Done.
    echo.
)

REM Install dependencies
echo [2/2] Installing dependencies...
venv\Scripts\pip.exe install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Done.
echo.

echo Starting GUI...
echo.
venv\Scripts\python.exe gui_app.py
pause
