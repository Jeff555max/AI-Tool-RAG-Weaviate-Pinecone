@echo off
echo Stopping all Python processes...
taskkill /F /IM python.exe /T 2>nul
taskkill /F /IM pythonw.exe /T 2>nul
timeout /t 2 /nobreak >nul

echo Starting GUI...
cd /d "%~dp0"
start "" venv\Scripts\pythonw.exe gui_app.py
