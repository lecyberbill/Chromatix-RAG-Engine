@echo off
echo ==========================================
echo [SCIENTIFIC RAG] Dmarrage du systme...
echo ==========================================


:: Activation venv et Backend
start cmd /k "echo [BACKEND] Lancement sur port 8000... && venv\Scripts\activate && python app.py"

:: Frontend
start cmd /k "echo [FRONTEND] Lancement sur port 5180... && cd frontend && npm run dev"

echo Le systme sera prt dans quelques secondes.
echo [LOCAL] http://localhost:5180
echo [NETWORK] Consultez l'IP affiche dans le terminal Frontend.
timeout /t 5
start http://localhost:5180
