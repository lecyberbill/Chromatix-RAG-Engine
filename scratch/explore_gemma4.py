import torch
from transformers import AutoModelForCausalLM

model_id = "google/gemma-4-E2B-it"
print(f"--- [DEBUG] Inspection de {model_id} ---")

try:
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="cpu")
    print("\n--- CONFIG ATTRIBUTES ---")
    for attr in dir(model.config):
        if not attr.startswith("_"):
            print(f"{attr}: {getattr(model.config, attr)}")
            
    print("\n--- MODEL ATTRIBUTES ---")
    for attr in dir(model):
        if "think" in attr.lower() or "reason" in attr.lower():
            print(f"FOUND: {attr}")

except Exception as e:
    print(f"ERROR: {e}")
