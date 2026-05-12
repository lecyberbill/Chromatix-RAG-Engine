@echo off
cd /d "%~dp0"
echo [DEBUG] Script commence...
echo [DEBUG] Dossier : %CD%

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv non trouve !
    pause
    exit
)

echo [DEBUG] Lancement Backend...
start "BACKEND" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe app.py"

echo [DEBUG] Lancement Frontend...
start "FRONTEND" cmd /k "cd /d %~dp0frontend && npm run dev"

echo [DEBUG] Termine.
pause
