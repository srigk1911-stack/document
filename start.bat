@echo off
echo ===================================================
echo   Document RAG System: Setup and Startup Launcher
echo ===================================================
echo.

cd "%~dp0backend"

:: Check python launcher or executable
set PYTHON_CMD=py
where py >nul 2>nul
if %errorlevel% neq 0 (
    set PYTHON_CMD=python
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Python was not found on your system!
        echo Please install Python 3.9+ and add it to your PATH environment variable.
        pause
        exit /b 1
    )
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating Python virtual environment (venv)...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing requirements (this might take a moment on first load)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. We will attempt to run anyway...
)

echo [INFO] Starting FastAPI application server...
echo Point your browser to: http://localhost:8000
echo.
python main.py

pause
