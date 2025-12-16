@echo off
echo ========================================
echo Полная перезагрузка GUI
echo ========================================
echo.

echo [1/3] Остановка всех процессов Python...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
echo Готово.
echo.

echo [2/3] Ожидание 3 секунды...
timeout /t 3 /nobreak >nul
echo Готово.
echo.

echo [3/3] Запуск GUI...
cd /d "%~dp0"
venv\Scripts\python.exe gui_app.py
pause
