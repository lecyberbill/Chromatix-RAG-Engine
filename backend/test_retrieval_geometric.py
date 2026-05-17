import os
import sys
import time

print("[DEBUG] SCRIPT_START")

try:
    import torch
    import gc
    import io
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from chromatix_cps.integrations import TransformersHook
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    print("[DEBUG] IMPORTS_OK")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def test_retrieval_geometric(query):
    try:
        print(f"--- [RAG_TEST] Requête : {query} ---")
        
        # 1. Access DB Direct (No embeddings needed for metadata extraction)
        db_path = r"d:\image_to_text\RAG\backend\data\db_geometric"
        print(f"[*] Ouverture directe de la base Chroma : {db_path}")
        db = Chroma(persist_directory=db_path) # No embedding function needed for .get()
        
        # 2. Récupération directe du premier cristal
        print("[*] Récupération du premier cristal stocké...")
        data = db.get(limit=1, include=["documents", "metadatas"])
        if not data["metadatas"]:
            print("!!! [ERROR] Aucun cristal trouvé dans la base.")
            return
        
        doc_text = data["documents"][0]
        metadata = data["metadatas"][0]
        print(f"[*] Cristal trouvé pour : {doc_text[:50]}...")
        brain_state_hex = metadata["brain_state"]
        brain_state_bytes = bytes.fromhex(brain_state_hex)
        
        # 3. Modèle
        print("[*] Chargement du modèle...")
        model_id = "google/gemma-4-E2B-it"
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True)
        hook = TransformersHook(model, master_key="RAG-SECURE-STREAM")
        
        # 4. Injection
        print("[*] INJECTION CGC...")
        start = time.time()
        stream = io.BytesIO(brain_state_bytes)
        outputs, pkv = hook.inject_brain_state(stream)
        print(f"[SUCCESS] Injecté en {(time.time()-start)*1000:.2f} ms")
        print(f"      [DEBUG] PKV Type: {type(pkv)}")
        
        try:
            if hasattr(pkv, "key_cache"):
                print(f"      [DEBUG] DynamicCache detected. Layer 0 K shape: {pkv.key_cache[0].shape}")
                print(f"      [DEBUG] Cache length: {pkv.get_seq_length()}")
            else:
                print(f"      [DEBUG] PKV Layer 0 K shape: {pkv[0][0].shape}")
        except:
            print("      [DEBUG] Could not inspect PKV shape (unsupported format).")
        
        # 5. Gen
        print("[*] Génération...")
        messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # [NITRO-FIX] Attention Mask for Injected Cache
        cache_len = pkv.get_seq_length() if hasattr(pkv, "get_seq_length") else 0
        prompt_len = inputs["input_ids"].shape[1]
        
        # Create full mask (1s for both cache and prompt)
        full_mask = torch.ones((1, cache_len + prompt_len), device="cuda", dtype=torch.long)
        
        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=full_mask,
                past_key_values=pkv, 
                max_new_tokens=30,
                use_cache=True,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        print(f"RÉPONSE : {tokenizer.decode(gen[0], skip_special_tokens=True)}")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval_geometric(sys.argv[1] if len(sys.argv) > 1 else "Qu'est-ce que Chromatix ?")
