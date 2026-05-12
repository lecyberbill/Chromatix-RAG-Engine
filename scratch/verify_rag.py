import torch
import os
import io
import time
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from chromatix_cps.integrations import TransformersHook

# Configuration
MODEL_ID = "google/gemma-4-E2B-it"
# On va chercher un fichier dans le vault (on prend le plus recent)
VAULT_DIR = "backend/vault"
exr_files = [f for f in os.listdir(VAULT_DIR) if f.endswith(".exr")]
if not exr_files:
    print("ERROR: No EXR files found in vault.")
    exit()

TARGET_EXR = os.path.join(VAULT_DIR, sorted(exr_files)[-1])
print(f"--- [TEST] Verification de {TARGET_EXR} ---")

# 1. Chargement Modele (Eco-Mode)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map={"": 0}, 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
hook = TransformersHook(model, "RAG-SECURE-STREAM", processor=processor)

# 2. Injection du State
with open(TARGET_EXR, "rb") as f:
    stream = io.BytesIO(f.read())

print("--- [TEST] Injection des neurones... ---")
try:
    # On injecte et on demande une continuation
    # On pose une question simple qui n'a de sens que si le contexte est present
    question = "Peux-tu me donner le titre ou le sujet principal de ce document ?"
    
    outputs = hook.inject_brain_state(stream, next_text=question)
    
    print("\n--- [RESULTAT DE L'IA] ---")
    print(processor.decode(outputs[0], skip_special_tokens=True))
    print("--------------------------")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
