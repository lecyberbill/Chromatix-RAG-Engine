# [WFGY] Zone: TRANSIT | λ: 0.1 | Action: Update status endpoint model name
import os
import sys
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(r"d:\image_to_text\chromatix")

from fastapi.staticfiles import StaticFiles
from backend.engine import RAGEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("   [STARTUP] DÉMARRAGE DU SERVEUR DE NEURAL RAG WORKSTATION")
    print("   [STARTUP] Initialisation anticipée du modèle Ministral-3 FP8...")
    print("   [STARTUP] Note: La compilation des kernels Triton s'exécute")
    print("             au premier démarrage ou prefill (attente possible).")
    print("="*60 + "\n", flush=True)
    
    # Lancement non bloquant dans un thread séparé
    asyncio.create_task(asyncio.to_thread(get_engine))
    yield

app = FastAPI(title="Chromatix Neural RAG Workstation", lifespan=lifespan)

# Assurer la presence des dossiers de données
os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/brain_vault", exist_ok=True)

# Montage du dossier PDF pour accès direct depuis le frontend
app.mount("/pdfs", StaticFiles(directory="data/pdfs"), name="pdfs")

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine instance
engine = None

def get_engine():
    global engine
    if engine is None:
        print("--- [API] Initialisation du moteur RAG... ---", flush=True)
        engine = RAGEngine(step_seq=2, zstd_level=3)
        print("\n" + "="*60)
        print("   [STARTUP] MOTEUR CHARGÉ ET INITIALISÉ SUR GPU (VRAM OK)")
        print("   [STARTUP] Serveur FastAPI prêt à écouter les requêtes")
        print("="*60 + "\n", flush=True)
    return engine

from typing import List

class QueryRequest(BaseModel):
    query: str
    history: List[dict] = []

@app.get("/status")
async def get_status():
    eng = get_engine()
    return {"status": "ready", "model": eng.model_id if eng else "initializing"}

@app.post("/ingest")
async def ingest_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Save file temporarily
    temp_dir = "data/pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    import uuid
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(temp_dir, unique_filename)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Run ingestion in background to not block the UI
    # We'll use the ingest_docling script via subprocess to ensure total isolation
    background_tasks.add_task(run_standalone_ingest, file_path)
    
    return {"message": "Ingestion démarrée", "filename": file.filename}

# Verrou pour s'assurer qu'une seule ingestion tourne a la fois
ingest_lock = asyncio.Lock()

# Suivi de progression global
ingestion_progress = {"current": 0, "total": 0, "status": "idle", "filename": "", "original_name": "", "phase": "idle"}

@app.get("/ingest_status")
async def get_ingest_status():
    async def event_generator():
        while True:
            yield f"data: {json.dumps(ingestion_progress)}\n\n"
            if ingestion_progress["status"] in ["complete", "error"]:
                await asyncio.sleep(1)
                break
            await asyncio.sleep(0.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

async def run_standalone_ingest(file_path):
    async with ingest_lock:
        global ingestion_progress
        filename = os.path.basename(file_path)
        # Extraire le nom original (apres le premier _)
        original_name = "_".join(filename.split("_")[1:]) if "_" in filename else filename
        
        ingestion_progress = {
            "current": 0, 
            "total": 0, 
            "status": "processing", 
            "filename": filename,
            "original_name": original_name,
            "phase": "idle"
        }
        
        # On utilise le thread pool executor pour ne pas bloquer l'event loop
        loop = asyncio.get_event_loop()
        
        def update_progress(current, total, phase="processing"):
            # Cette fonction sera appellee depuis le thread de l'ingestion
            ingestion_progress["current"] = current
            ingestion_progress["total"] = total
            ingestion_progress["phase"] = phase
            
        try:
            rag = get_engine()
            # Execution en thread separe pour laisser les SSE passer
            await loop.run_in_executor(None, rag.ingest_high_fidelity, file_path, update_progress)
            ingestion_progress["status"] = "complete"
        except Exception as e:
            print(f"!!! [API] Erreur ingestion: {e}")
            ingestion_progress["status"] = "error"

@app.post("/ask")
async def ask_question(request: QueryRequest):
    rag = get_engine()
    
    async def event_generator():
        q = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        def producer():
            try:
                for item in rag.ask(request.query, history=request.history):
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, {"error": str(e)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        # Start the blocking generator in a background thread
        loop.run_in_executor(None, producer)

        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    if "error" in item:
                        yield f"data: {json.dumps(item)}\n\n"
                    else:
                        yield f"data: {json.dumps(item)}\n\n"
                else:
                    yield f"data: {json.dumps({'chunk': item})}\n\n"
                q.task_done()
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/open_pdf")
async def open_pdf(path: str, page: int = 1):
    """
    Ouvre le PDF localement avec le lecteur par défaut.
    """
    if os.path.exists(path):
        try:
            # Sur Windows, startfile ouvre le lecteur par défaut
            # Pour ouvrir a une page specifique, certains lecteurs acceptent des arguments
            # Mais startfile est le plus compatible.
            os.startfile(path)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "Fichier introuvable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
