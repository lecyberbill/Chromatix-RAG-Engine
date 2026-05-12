import sys
import os
import time
import io

# Add RAG and Chromatix to path
sys.path.append(r"d:\image_to_text\RAG\backend")
sys.path.append(r"d:\image_to_text\chromatix")

from engine import RAGEngine

def test_full_pipeline():
    print("--- [TEST] Initialisation de l'Engine RAG (Phase 3) ---")
    engine = RAGEngine()
    
    # 1. Test d'Ingestion (Désactivé pour garder les données Docling)
    # pdf_to_test = r"d:\image_to_text\RAG\backend\data\pdfs\RCM_1_2020.pdf"
    # if os.path.exists(pdf_to_test):
    #     print(f"\n--- [1/2] Test d'Ingestion : {os.path.basename(pdf_to_test)} ---")
    #     num_chunks = engine.process_file(pdf_to_test)
    #     print(f"Succès : {num_chunks} segments indexés avec Brain States.")
    # else:
    #     print(f"\n[SKIP] Fichier {pdf_to_test} non trouvé. Ingestion ignorée.")

    # 2. Test de Query (Neural Teleportation)
    print("\n--- [2/2] Test de Téléportation Neuronale ---")
    query = "Résume les points clés de ce document."
    
    print(f"Question : {query}")
    print("Réponse (Streaming) :")
    
    start_time = time.time()
    first_token_time = None
    
    for chunk in engine.ask(query):
        if not first_token_time:
            first_token_time = time.time()
            print(f"\n[PERF] Temps jusqu'au premier token : {first_token_time - start_time:.2f}s")
        
        print(chunk, end="", flush=True)
    
    print(f"\n\n[PERF] Temps total de génération : {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    test_full_pipeline()
