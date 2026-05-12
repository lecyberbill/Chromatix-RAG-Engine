@echo off
echo ==========================================
echo [SCIENTIFIC RAG] Installation Automatique
echo ==========================================

:: 1. Vérification Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas install. Veuillez l'installer sur python.org
    pause
    exit /b
)

:: 2. Création du venv
if not exist venv (
    echo [PYTHON] Cration de l'environnement virtuel...
    python -m venv venv
)

:: 3. Installation des dépendances Python
echo [PYTHON] Installation des bibliothques (Backend)...
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install fastapi uvicorn pypdf langchain-chroma langchain-huggingface langchain-docling langchain-ollama docling requests torch --extra-index-url https://download.pytorch.org/whl/cu124

:: 4. Installation des dépendances Node
echo [FRONTEND] Installation des bibliothques (Interface)...
cd frontend
call npm install
cd ..

echo ==========================================
echo [SUCCES] Installation terminee !
echo Pour lancer le systeme, utilisez : start.bat
echo ==========================================
pause
