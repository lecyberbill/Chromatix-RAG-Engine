import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import time

model_id = "ibm-granite/granite-docling-258M"

print(f"--- [DEBUG] Tentative de chargement brute de {model_id}... ---")
start = time.time()

try:
    print("1. Chargement du Processor...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Processor OK en {time.time() - start:.2f}s")
    
    print("2. Chargement du Modele (float16 sur GPU)...")
    model_start = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Modele OK en {time.time() - model_start:.2f}s (Total: {time.time() - start:.2f}s)")
    print(f"Memoire CUDA allouee : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
except Exception as e:
    print(f"--- [ERREUR] : {e} ---")
    import traceback
    traceback.print_exc()
