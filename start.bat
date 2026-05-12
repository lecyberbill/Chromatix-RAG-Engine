@echo off
echo ==========================================
echo [SCIENTIFIC RAG] Dmarrage du systme...
echo ==========================================

:: Ollama (Inference Engine)
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" > nul
if errorlevel 1 (
    echo [OLLAMA] Tentative de lancement d'Ollama...
    start "" "ollama" serve
    timeout /t 3
)

:: Activation venv et Backend
start cmd /k "echo [BACKEND] Lancement sur port 8080... && cd backend && ..\venv\Scripts\activate && python main.py"

:: Frontend
start cmd /k "echo [FRONTEND] Lancement sur port 5180... && cd frontend && npm run dev"

echo Le systme sera prt dans quelques secondes.
echo [LOCAL] http://localhost:5180
echo [NETWORK] Consultez l'IP affiche dans le terminal Frontend.
timeout /t 5
start http://localhost:5180
