@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo [SCIENTIFIC RAG] Ingestion Massive
echo ==========================================
echo.
echo Ce script va scanner un dossier et envoyer tous les PDF/DOCX 
echo vers le serveur en cours d'execution.
echo.

set /p FOLDER_PATH="Collez le chemin du dossier a scanner : "

:: Nettoyage des guillemets si presents
set FOLDER_PATH=%FOLDER_PATH:"=%

if not exist "!FOLDER_PATH!" (
    echo [ERREUR] Le chemin n'existe pas : !FOLDER_PATH!
    pause
    exit /b
)

echo [INFO] Demarrage de l'ingestion depuis : !FOLDER_PATH!
echo.

.\venv\Scripts\python backend\ingest_folder.py "!FOLDER_PATH!"

echo.
echo ==========================================
echo Ingestion terminee.
echo ==========================================
pause
