import os
import sys
import torch
import gc
import time
import io
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Paths
sys.path.append(r"d:\image_to_text\RAG\backend")
sys.path.append(r"d:\image_to_text\chromatix")

from chromatix_cps.integrations import TransformersHook
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_full_evaluation(pdf_path):
    print(f"--- [EVALUATION] Évaluation Globale CGC 2x : {os.path.basename(pdf_path)} ---")
    
    # 1. Extraction
    import pypdf
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages[:5]: # On évalue sur 5 pages
        text += page.extract_text() + "\n"
    segments = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 300][:5]
    print(f"[*] {len(segments)} segments techniques sélectionnés pour l'évaluation.")

    if not segments:
        print("!!! [ERROR] Aucun segment extrait.")
        return

    # 2. Modèle
    model_id = "google/gemma-4-E2B-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True)
    hook = TransformersHook(model, master_key="EVAL-KEY")
    
    # 3. Traitement
    results = []
    print("[*] Génération des cristaux CGC 2x...")
    for i, seg in enumerate(segments):
        stream = hook.extract_brain_state(seg, geometric=True, step_dim=2)
        results.append({"text": seg, "stream": stream})
        print(f"    [Segment {i+1}] Cristal généré (Size: {stream.getbuffer().nbytes/1024:.1f} KB)")

    # 4. Questions d'évaluation
    queries = [
        "Quel est le sujet principal du Tome 1 ?",
        "Quels sont les phénomènes physiques mentionnés ?",
        "À quel Eurocode le document fait-il référence ?"
    ]
    
    print("-" * 30)
    print("RÉSULTATS DE L'ÉVALUATION SÉMANTIQUE :")
    for q in queries:
        # On simule un retrieval (on injecte le cristal le plus pertinent - ici le premier pour le test)
        stream = results[0]["stream"]
        stream.seek(0)
        _, pkv = hook.inject_brain_state(stream)
        
        prompt = f"Basé sur le document, répond précisément à : {q}\nRéponse :"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Mask fix
        cache_len = pkv.get_seq_length() if hasattr(pkv, "get_seq_length") else 0
        full_mask = torch.ones((1, cache_len + inputs["input_ids"].shape[1]), device="cuda", dtype=torch.long)
        
        with torch.no_grad():
            gen = model.generate(input_ids=inputs["input_ids"], attention_mask=full_mask, past_key_values=pkv, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
        
        ans = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"\nQUERY : {q}")
        print(f"CGC 2X : {ans}")
    print("-" * 30)

if __name__ == "__main__":
    target_pdf = r"d:\image_to_text\RAG\test\Pages de guide_silo_A4_T1.pdf"
    run_full_evaluation(target_pdf)
