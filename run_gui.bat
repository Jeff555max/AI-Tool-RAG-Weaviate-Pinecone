@echo off
echo Starting RAG GUI Application...
echo.

REM Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate venv and install dependencies
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -q -r requirements.txt

echo.
echo Launching GUI...
python gui_app.py

pause
