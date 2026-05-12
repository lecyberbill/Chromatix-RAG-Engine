import sys
import os

print("1. Test os/sys... OK")

try:
    import torch
    print(f"2. Test torch ({torch.__version__})... OK")
    print(f"   CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"2. Test torch... ERREUR: {e}")

try:
    import transformers
    print(f"3. Test transformers ({transformers.__version__})... OK")
except Exception as e:
    print(f"3. Test transformers... ERREUR: {e}")

try:
    import bitsandbytes
    print(f"4. Test bitsandbytes... OK")
except Exception as e:
    print(f"4. Test bitsandbytes... ERREUR: {e}")

try:
    from langchain_chroma import Chroma
    print("5. Test ChromaDB... OK")
except Exception as e:
    print(f"5. Test ChromaDB... ERREUR: {e}")

try:
    print("6. Tentative d'importation de RAGEngine...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
    from engine import RAGEngine
    print("7. Importation RAGEngine... SUCCÈS")
except Exception as e:
    print(f"ERREUR FATALE lors de l'importation de l'engine : {e}")
