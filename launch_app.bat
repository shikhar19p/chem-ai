@echo off
title Reactive Extraction Predictor
echo ========================================
echo  Reactive Extraction Predictor Launcher
echo ========================================
echo.
echo [1/2] Starting Python API backend on port 8000...
start "API Backend" /min python -m uvicorn api:app --port 8000 --log-level warning

echo [2/2] Waiting for API to load models (15 seconds)...
timeout /t 15 /nobreak > nul

echo [3/3] Opening React frontend in browser...
start "" "%~dp0frontend\index.html"

echo.
echo App is running!
echo   - API:      http://localhost:8000
echo   - Docs:     http://localhost:8000/docs  (Swagger UI)
echo   - Frontend: frontend/index.html
echo.
echo Press any key to stop the backend server...
pause > nul

echo Stopping backend...
taskkill /FI "WINDOWTITLE eq API Backend*" /F > nul 2>&1
