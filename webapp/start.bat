@echo off
setlocal enabledelayedexpansion

echo ==============================================
echo Brain MRI Classifier Web Application
echo ==============================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if Hadoop is running
echo [INFO] Checking prerequisites...
jps 2>nul | findstr "NameNode" >nul
if errorlevel 1 (
    echo [WARN] Hadoop not detected. You may need to start it manually.
    echo        Run: start-dfs.cmd
) else (
    echo [OK] Hadoop is running
)

REM Check for model file
if exist "..\best_model_stage1.keras" (
    echo [OK] Model file found
) else (
    echo [ERROR] Model file not found
    echo         Please ensure best_model_stage1.keras exists in the project root.
    pause
    exit /b 1
)

REM Start Backend
echo.
echo [INFO] Starting Backend (Flask API)...
cd backend

REM Check if virtual environment exists
if not exist "venv\" (
    if not exist "..\..\dtsvenv\" (
        echo [INFO] Creating virtual environment...
        python -m venv venv
        call venv\Scripts\activate
        pip install -q -r requirements.txt
    ) else (
        call ..\..\dtsvenv\Scripts\activate
    )
) else (
    call venv\Scripts\activate
)

REM Start Flask in new window
start "Brain MRI Backend" cmd /k "python app.py"
echo [OK] Backend starting in new window...

cd ..

REM Wait for backend to initialize
echo [INFO] Waiting for backend to initialize (30 seconds)...
timeout /t 30 /nobreak >nul

REM Check if backend is responding
curl -s http://localhost:5000/api/health >nul 2>&1
if errorlevel 1 (
    echo [WARN] Backend may still be initializing. Please wait...
) else (
    echo [OK] Backend is responding
)

REM Start Frontend
echo.
echo [INFO] Starting Frontend (React)...
cd frontend

REM Install dependencies if needed
if not exist "node_modules\" (
    echo [INFO] Installing npm dependencies...
    call npm install
)

REM Start React in new window
start "Brain MRI Frontend" cmd /k "npm start"
echo [OK] Frontend starting in new window...

cd ..

REM Summary
echo.
echo ==============================================
echo Application Started Successfully!
echo ==============================================
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5000
echo.
echo Two new command windows have been opened:
echo   - "Brain MRI Backend" - Flask server
echo   - "Brain MRI Frontend" - React development server
echo.
echo Close those windows to stop the application.
echo.
pause
