@echo off
REM PharmaLens — Launch the application
REM Starts the FastAPI backend which also serves the frontend

echo.
echo   ============================================
echo     PharmaLens — Starting Application
echo   ============================================
echo.

call venv\Scripts\activate.bat

echo   API Docs:  http://localhost:8000/docs
echo   Frontend:  http://localhost:8000/
echo.
echo   Press Ctrl+C to stop.
echo.

cd backend
python main.py
