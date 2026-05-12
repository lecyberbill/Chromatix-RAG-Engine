# [WFGY] Zone: SAFE | λ: 0.1 | Action: Bulk Ingestion via API
import os
import requests
import sys
import time

def ingest_folder(folder_path):
    api_url = "http://localhost:8000/ingest"
    supported_extensions = ['.pdf', '.docx', '.pptx']
    
    if not os.path.exists(folder_path):
        print(f"[ERREUR] Le dossier n'existe pas : {folder_path}")
        return

    print(f"--- [BULK INGEST] Scan de : {folder_path} ---")
    
    files_to_process = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                files_to_process.append(os.path.join(root, file))

    total = len(files_to_process)
    print(f"[INFO] {total} fichiers trouvés.")
    
    success_count = 0
    error_count = 0

    for i, file_path in enumerate(files_to_process):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{total}] Ingestion de : {filename}...", end="\r")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f)}
                response = requests.post(api_url, files=files)
                
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"\n[ERREUR] {filename} : {response.text}")
                error_count += 1
        except Exception as e:
            print(f"\n[ERREUR] Impossible de contacter le serveur : {e}")
            print("Assurez-vous que le serveur est bien lancé (start.bat)")
            break

    print(f"\n--- [TERMINE] Succès: {success_count} | Erreurs: {error_count} ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = input("Entrez le chemin du dossier à scanner : ").strip('"')
    else:
        path = sys.argv[1].strip('"')
    
    ingest_folder(path)
