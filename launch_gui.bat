@echo off
echo Launching RAG GUI...
cd /d "%~dp0"
venv\Scripts\python.exe gui_app.py
pause
