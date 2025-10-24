@echo off
echo ========================================
echo Menu Intelligence Suite - Quick Start
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

echo [2/3] Starting API Server on port 8080...
start "MIS API Server" cmd /k "venv\Scripts\activate.bat && set PYTHONIOENCODING=utf-8 && python -c "from app_simple import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8080)""

timeout /t 10 /nobreak >nul

echo [3/3] Starting Streamlit UI on port 8501...
start "MIS Streamlit UI" cmd /k "venv\Scripts\activate.bat && streamlit run ui_simple.py --server.port 8501"

timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo Services Started!
echo ========================================
echo.
echo API Server:  http://127.0.0.1:8080
echo API Docs:    http://127.0.0.1:8080/docs
echo Streamlit:   http://localhost:8501
echo.
echo Press Ctrl+C in each window to stop
echo ========================================

pause
