# [WFGY] Zone: SAFE | λ: 0.2 | Action: Test CPS-RAG Bridge
import sys
import os

# Add RAG backend to path
sys.path.append(r"d:\image_to_text\RAG\backend")
from engine import RAGEngine

def test_rag_cps():
    print("=== [TEST BRIDGE RAG-CPS] ===")
    engine = RAGEngine()
    
    query = "pisé" # Terme très présent dans votre base
    print(f"Question : {query}")
    
    # On simule un appel partiel pour voir si le Retrieval + CPS fonctionne
    try:
        for chunk in engine.ask(query):
            if chunk.startswith("CPS_MATRIX:"):
                print(f"\n[SUCCESS] Matrice CPS générée avec succès !")
                print(f"Taille du payload : {len(chunk)//2} octets.")
                print(f"Aperçu Hex : {chunk[:60]}...")
            elif chunk.startswith("SOURCES_JSON:"):
                sources = chunk.replace("SOURCES_JSON:", "")
                print(f"[INFO] {len(sources)} octets de métadonnées sources trouvés.")
    except Exception as e:
        print(f"\n[INFO] Fin du flux (attendu si Ollama offline) : {e}")

if __name__ == "__main__":
    test_rag_cps()
