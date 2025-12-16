@echo off
title RAG GUI Application
cd /d "%~dp0"

echo ========================================
echo RAG Vector Store GUI
echo ========================================
echo.
echo Starting application...
echo.

venv\Scripts\python.exe gui_app.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Application failed to start
    echo ========================================
    pause
)
