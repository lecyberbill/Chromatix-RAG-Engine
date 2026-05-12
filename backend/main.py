from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import json
from engine import engine

app = FastAPI(title="NemoChain : Advanced Scientific RAG")

# Configuration des dossiers
UPLOAD_DIR = "data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# CORS pour la communication avec le frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Montage du dossier PDF pour accès direct depuis le frontend (iframe)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

@app.get("/")
async def root():
    return {"status": "online", "message": "Local RAG System is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.pptx']
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Formats acceptés: {', '.join(allowed_extensions)}")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Traitement par le moteur RAG
    num_chunks = engine.process_file(file_path)
        
    return {
        "filename": file.filename, 
        "chunks": num_chunks,
        "message": f"Analyse terminée : {num_chunks} segments extraits."
    }

from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    q: str
    history: Optional[List[dict]] = None

@app.post("/query")
async def query_rag(request: QueryRequest):
    return StreamingResponse(
        engine.ask(request.q, history=request.history), 
        media_type="text/plain"
    )

# Dossier pour les sessions
SESSIONS_DIR = "data/sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

class Session(BaseModel):
    id: str
    title: str
    messages: List[dict]

@app.get("/sessions")
async def list_sessions():
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    sessions = []
    for f in files:
        try:
            with open(os.path.join(SESSIONS_DIR, f), "r", encoding="utf-8") as file:
                data = json.load(file)
                sessions.append({"id": data["id"], "title": data["title"]})
        except Exception as e:
            print(f"Skipping corrupted session file {f}: {e}")
            continue
    return sessions

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session introuvable")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/sessions")
async def save_session(session: Session):
    path = os.path.join(SESSIONS_DIR, f"{session.id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.model_dump(), f, ensure_ascii=False, indent=2)
    return {"status": "saved"}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session introuvable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
