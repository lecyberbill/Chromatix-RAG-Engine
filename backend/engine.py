# [WFGY] Zone: TRANSIT | λ: 0.4 | Action: Switch RAG to Gemma-4 with shape-aligned SDPA patch
import os
import torch
from PIL import Image
import json
import gc
import sys
import uuid
import io
import time
from threading import Thread
import struct

# Chromatix CPS
# Add chromatix to path
sys.path.append(r"d:\image_to_text\chromatix")
from chromatix_cps.core import CPSPacket

# PDF Processing
from pypdf import PdfReader, PdfWriter

# [CRITICAL] Docling pour l'ingestion unifiee
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from typing import Optional
import transformers.integrations.sdpa_attention as sdpa_module

from langchain_chroma import Chroma

print("--- [DEBUG] Import Chromatix... ---")
# Add chromatix to path
sys.path.append(r"d:\image_to_text\chromatix")
from chromatix_cps.core import CPSPacket
from chromatix_cps.integrations import TransformersHook

# Extensible Multilingual Stopwords list
MULTILINGUAL_STOPWORDS = {
    # French
    "le", "la", "les", "des", "dans", "une", "un", "pour", "avec", "qui", "que", "est", "sont", "et", "ou", "mais", "dans", "sur", "par", "ce", "ces", "mon", "ton", "son", "faire", "cas",
    # English
    "the", "and", "this", "that", "with", "from", "for", "are", "was", "were", "been", "would", "what", "to", "do", "in", "of", "how", "why", "who", "which", "will", "shall", "should", "your", "my", "his", "her", "their",
    # Italian
    "il", "la", "i", "gli", "del", "della", "con", "per", "che", "sono", "questo", "quello", "e", "o", "ma", "in", "su", "da", "faire", "caso",
    # Spanish
    "el", "la", "los", "las", "del", "con", "para", "que", "este", "esta", "como", "puede", "y", "o", "pero", "en", "sobre", "por", "hacer", "caso"
}

def detect_text_language(text: str) -> str:
    text_lower = text.lower()
    markers = {
        "fr": ["le", "la", "les", "des", "dans", "une", "pour", "avec", "qui", "que", "est", "sont"],
        "en": ["the", "and", "this", "that", "with", "from", "for", "are", "was", "were", "been", "would"],
        "it": ["il", "la", "i", "gli", "del", "della", "con", "per", "che", "sono", "questo", "quello"],
        "es": ["el", "la", "los", "las", "del", "con", "para", "que", "este", "esta", "como", "puede"]
    }
    scores = {lang: 0 for lang in markers}
    words = text_lower.split()
    for word in words:
        word_clean = word.strip(".,:;()\"'-")
        for lang, words_list in markers.items():
            if word_clean in words_list:
                scores[lang] += 1
                
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] == 0:
        return "fr"  # Fallback par défaut sur le français pour notre workstation
    return best_lang

DOMAIN_CONCEPTS = {
    "helicopter": {
        "en": {
            "lte": ["lte", "loss of tail rotor", "tail rotor effectiveness"],
            "tail_rotor": ["tail rotor", "tail rotor failure", "anti-torque"],
            "left_rotation": ["left rotation", "yaw to the left", "yawing to the left", "spin to the left"],
            "right_rotation": ["right rotation", "yaw to the right", "yawing to the right", "spin to the right"]
        },
        "fr": {
            "lte": ["lte", "perte d'efficacite du rotor", "perte d'efficacite rotor"],
            "tail_rotor": ["rotor de queue", "defaillance rotor", "anti-couple"],
            "left_rotation": ["rotation a gauche", "rotation vers la gauche", "dévier vers la gauche", "déviation vers la gauche"],
            "right_rotation": ["rotation a droite", "rotation vers la droite", "dévier vers la droite", "déviation vers la droite"]
        },
        "it": {
            "lte": ["lte", "perdita di efficacia del rotore"],
            "tail_rotor": ["rotore di coda", "guasto del rotore"],
            "left_rotation": ["rotazione a sinistra", "imbardata a sinistra"],
            "right_rotation": ["rotazione a destra", "imbardata a destra"]
        },
        "es": {
            "lte": ["lte", "perdida de eficacia del rotor"],
            "tail_rotor": ["rotor de cola", "fallo del rotor"],
            "left_rotation": ["rotacion a la izquierda", "guiñada a la izquierda"],
            "right_rotation": ["rotacion a la droite", "guiñada a la droite"]
        }
    }
}

class RAGEngine:
    def __init__(self, model_id="google/gemma-4-E2B-it", step_seq: int = 2, zstd_level: int = 3):
        # 1. Configuration
        self.model_id = model_id
        self.step_seq = step_seq
        self.zstd_level = zstd_level
        self.db_path = "data/db"
        self.pdf_path = "data/pdfs"
        self.use_seh = True
        self.max_references = 3
        self.max_seh_candidates = 5
        
        # Apply Gemma 4 patches dynamically if loading Gemma 4
        if "gemma-4" in self.model_id:
            print("--- [LLM] Appliquant les patches d'attention/cache Gemma-4... ---")
            # 1. SDPA Forward Patch to prevent shape mismatch crashes
            original_sdpa_attention_forward = sdpa_module.sdpa_attention_forward
            def custom_sdpa_attention_forward(module, query, key, value, attention_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
                if attention_mask is not None:
                    mask_seq_len = attention_mask.shape[-1]
                    key_seq_len = key.shape[-2]
                    if mask_seq_len != key_seq_len:
                        if mask_seq_len < key_seq_len:
                            padding_size = key_seq_len - mask_seq_len
                            padding = torch.zeros((*attention_mask.shape[:-1], padding_size), device=attention_mask.device, dtype=attention_mask.dtype)
                            attention_mask = torch.cat([padding, attention_mask], dim=-1)
                        else:
                            attention_mask = attention_mask[..., :key_seq_len]
                return original_sdpa_attention_forward(module, query, key, value, attention_mask=attention_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
            sdpa_module.sdpa_attention_forward = custom_sdpa_attention_forward

            # 2. ALL_ATTENTION_FUNCTIONS patch (v6.6): Gemma 4 imports SDPA during module compilation, bypassing module-level patch
            try:
                import transformers.models.gemma4.modeling_gemma4 as gemma4_module
                if hasattr(gemma4_module, 'ALL_ATTENTION_FUNCTIONS'):
                    for attn_key in list(gemma4_module.ALL_ATTENTION_FUNCTIONS.keys()):
                        gemma4_module.ALL_ATTENTION_FUNCTIONS[attn_key] = custom_sdpa_attention_forward
                    print("--- [LLM] ALL_ATTENTION_FUNCTIONS monkey-patch appliqué ---")
            except (ImportError, AttributeError) as e:
                print(f"--- [LLM] ALL_ATTENTION_FUNCTIONS patch non disponible: {e} ---")

            # 3. DynamicCache usable length patch
            def custom_get_usable_length(self, seq_len: int, layer_idx: Optional[int] = None) -> int:
                if len(self.key_cache) == 0:
                    return 0
                idx = layer_idx if (layer_idx is not None and layer_idx < len(self.key_cache)) else 0
                return self.key_cache[idx].shape[-2]
            DynamicCache.get_usable_length = custom_get_usable_length
        
        # 1. Embedding Model (Custom Lightweight to avoid sentence-transformers crash)
        model_name = "BAAI/bge-small-en-v1.5"
        print(f"--- [EMBEDDINGS] Initialisation Custom (Pure Transformers)... ---")
        try:
            from transformers import AutoModel, AutoTokenizer
            class CustomEmbeddings:
                def __init__(self, model_id):
                    print(f"--- [EMBEDDINGS] Chargement sur GPU... ---")
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self.model = AutoModel.from_pretrained(model_id).to("cuda").half() # PASSAGE SUR GPU + HALF
                    self.model.eval()
                def embed_documents(self, texts):
                    inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    # Utilisation de la moyenne des tokens (mean pooling) pour plus de precision
                    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()
                def embed_query(self, text):
                    return self.embed_documents([text])[0]
            
            self.embeddings = CustomEmbeddings(model_name)
            print("--- [EMBEDDINGS] OK (GPU Accelere) ! ---")
        except Exception as e:
            print(f"!!! [EMBEDDINGS] Erreur fatale : {e}")
            raise e
        
        print(f"--- [LLM] Chargement de {self.model_id}... ---")
        from transformers import AutoProcessor
        
        import time
        start_time = time.time()
        
        if "gemma-4" in self.model_id:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.tokenizer = processor.tokenizer
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="cuda",
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
            
            # Squeeze logits for Gemma-4 to 3D to eliminate generate dimension crashes
            original_model_forward = self.model.forward
            def custom_model_forward(*args, **kwargs):
                outputs = original_model_forward(*args, **kwargs)
                if hasattr(outputs, "logits") and outputs.logits is not None:
                    if len(outputs.logits.shape) == 4 and outputs.logits.shape[1] == 1:
                        outputs.logits = outputs.logits.squeeze(1)
                return outputs
            self.model.forward = custom_model_forward
            
            self.hook = TransformersHook(self.model, "RAG-SECURE-STREAM", processor=processor, custom_layout=[256, 256, 256, 256, 512]*3)
            
        elif "Ministral-3-8B-Instruct-2512" in self.model_id:
            from transformers import Mistral3ForConditionalGeneration
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="cuda",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.hook = TransformersHook(self.model, "RAG-SECURE-STREAM", processor=self.tokenizer)
        else:
            from transformers import Mistral3ForConditionalGeneration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.hook = TransformersHook(self.model, "RAG-SECURE-STREAM", processor=self.tokenizer)

        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is None:
            if hasattr(self.tokenizer, 'eos_token_id'):
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"--- [LLM] Modele et Tokenizer charges en {time.time() - start_time:.2f}s ---")
        print("--- [LLM] Tokenizer et Template prets ! ---")
        
        # 3. Chromatix CPS
        self.cps = CPSPacket(master_key="RAG-SECURE-STREAM", recovery_key="RAG-RECOVERY-TETHER")
        
        # 4. Vector DB
        self.vector_db = None
        print("--- [DB] Initialisation de la base vectorielle... ---")
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            os.makedirs(self.db_path, exist_ok=True)

    def detect_text_language(self, text: str) -> str:
        return detect_text_language(text)

    def align_query_language(self, query: str, doc_lang: str) -> str:
        query_lang = self.detect_text_language(query)
        print(f"[+] Alignement linguistique: Requête '{query_lang}' -> Ouvrage '{doc_lang}'", flush=True)
        
        if query_lang == doc_lang:
            print("[+] Langues déjà alignées. Aucune traduction requise.", flush=True)
            return query
            
        print(f"[+] Traduction de la requête de '{query_lang}' vers '{doc_lang}'...", flush=True)
        lang_names = {
            "en": "English",
            "fr": "French",
            "it": "Italian",
            "es": "Spanish"
        }
        src_lang_name = lang_names.get(query_lang, "French")
        target_lang_name = lang_names.get(doc_lang, "English")
        
        messages_trans = [
            {
                "role": "user",
                "content": f"Translate the following {src_lang_name} text into {target_lang_name}. Respond ONLY with the direct {target_lang_name} translation. Do not add any conversational filler, quotes, or introduction.\n\n{src_lang_name}: {query}"
            }
        ]
        translation_prompt = self.tokenizer.apply_chat_template(messages_trans, tokenize=False, add_generation_prompt=True)
        
        inputs_trans = self.tokenizer(translation_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs_trans = self.model.generate(
                input_ids=inputs_trans["input_ids"],
                attention_mask=inputs_trans["attention_mask"],
                max_new_tokens=45,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        prompt_len_trans = inputs_trans["input_ids"].shape[1]
        raw_translation = self.tokenizer.decode(outputs_trans[0][prompt_len_trans:], skip_special_tokens=True).strip()
        
        translated = raw_translation.split("<")[0].split("\n")[0].strip()
        prefixes = ["translation:", "english:", "french:", "italian:", "spanish:"]
        for p in prefixes:
            if translated.lower().startswith(p):
                translated = translated[len(p):].strip()
        translated = translated.strip('"\'')
        print(f"[+] Requête alignée : '{translated}'", flush=True)
        return translated

    def arbitrate_pages_with_seh(self, ranked_pages: list, query: str = "") -> list:
        if not ranked_pages:
            return []
            
        import backend.seh as seh
        
        # 1. Pré-filtrage : on élimine les scores nuls/négatifs et on garde les top max_seh_candidates
        competing_candidates = [item for item in ranked_pages if item[0] > 0]
        competing_candidates = competing_candidates[:self.max_seh_candidates]
        
        if not competing_candidates:
            print("[SEH] Aucun candidat avec un score positif. Fallback direct.", flush=True)
            return [ranked_pages[0][1]]
            
        print(f"--- [SEH] Arbitrage neuronal sur {len(competing_candidates)} pages candidates actives (sur {len(ranked_pages)}) ---", flush=True)
        
        n_candidates = len(competing_candidates)
        
        # Determiner la taille du reseau en fonction de max_seh_candidates
        field_n = max(60, 2 ** (n_candidates.bit_length()))
        
        config = seh.SehConfig(
            n=field_n,
            inhib=0.02,        # Inhibition faible : laisse plusieurs attracteurs coexister
            gain=1.0,          # Bistabilite forte : attracteurs stables a ±sqrt(gain/beta) ≈ ±1.4
            beta=0.5,
            coupling=0.30,
            damping=0.05,       # Amortissement reduit : convergence rapide sans sous-amortissement
            c_target=0.001,     # Equalisation quasi-nulle : ne force pas les projections vers 0
            alpha_pol=0.05,
            cycles=200,
            threshold=0.01,
            noise_amp=0.01,
            allow_negative=False,
            max_winners=1
        )
        
        # Patterns strictement orthogonaux (Walsh-Hadamard) + seed deterministe lie a la requete
        patterns = seh.walsh_hadamard_patterns(config.n, list(range(1, n_candidates + 1)))
        seed = seh.seed_from_query(query) if query else 0xA5A5A5A5A5A5A5A5
        
        # Extraction et normalisation des coefficients
        raw_scores = [score for score, _ in competing_candidates]
        max_score = max(raw_scores) if raw_scores else 1.0
        
        if max_score > 0:
            coeffs = [s / max_score for s in raw_scores]
        else:
            coeffs = [0.0] * n_candidates
            
        decision = seh.decide(coeffs, patterns, config, seed=seed)
        projections = decision.get("projections", [])
        
        # SEH comme arbitre : les projections SEH amplifient ou penalisent le score LKPT.
        # final_score = LKPT * max(0.1, projection_SEH + 1.0)
        #   projection +1.4 (attracteur fort)  → score × 2.4  (fort boost)
        #   projection  0.0 (neutre)           → score × 1.0  (preserve)
        #   projection -0.5 (inhibe partiel)   → score × 0.5  (penalise)
        #   projection -1.0 (presque elimine)  → score × 0.1  (survie minimale)
        reranked = []
        for i, (score, page) in enumerate(competing_candidates):
            proj = projections[i] if i < len(projections) else 0.0
            boost = max(0.1, proj + 1.0)
            final_score = score * boost
            reranked.append((final_score, page, proj, boost))
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        
        print(f"--- [SEH] Arbitrage termine (cycles: {decision.get('cycles_used')}, early_stop: {decision.get('early_stop')}) ---", flush=True)
        
        # Log pour les candidats actifs
        for i, (score, page) in enumerate(competing_candidates):
            proj_val = projections[i] if i < len(projections) else 0.0
            final = reranked[i][0] if i < len(reranked) else 0.0
            boost = reranked[i][3] if i < len(reranked) else 1.0
            status = "🏆 GAGNANT (TELEPORTÉ)" if i == reranked[0][1] else "⬇️ DÉCLASSÉ"
            doc_name = os.path.basename(page.metadata.get("source", "Doc"))
            p_num = page.metadata.get("page", 1)
            print(f"   * Candidat {i+1} : {doc_name} (p.{p_num}) | LKPT: {score:.0f} | Proj SEH: {proj_val:+.3f} | Boost: {boost:.2f}x | Final: {final:.0f} -> {status}", flush=True)
            
        # Log pour les candidats exclus d'office
        for i, (score, page) in enumerate(ranked_pages):
            if (score, page) not in competing_candidates:
                doc_name = os.path.basename(page.metadata.get("source", "Doc"))
                p_num = page.metadata.get("page", 1)
                print(f"   * Filtré {i+1} : {doc_name} (p.{p_num}) | Score: {score:.0f} | -> ❌ HORS-JEU (Score trop faible)", flush=True)
        
        # Une seule page gagnante pour eviter la corruption du cache attentionnel
        selected_pages = [reranked[0][1]] if reranked else []
        
        if not selected_pages:
            print("[SEH] Aucun vainqueur n'a émergé de la compétition SEH. Fallback sur la meilleure page.", flush=True)
            selected_pages = [ranked_pages[0][1]]
            
        return selected_pages

    def rank_crystal_pages(self, pages: list, query: str, doc_lang: str) -> list:
        if not pages:
            return []
            
        query = query.lower()
        
        # Extract search terms dynamically
        words = query.split()
        keywords = []
        for w in words:
            w_clean = w.strip("?,.!:;()\"'-")
            if w_clean and len(w_clean) > 2 and w_clean not in MULTILINGUAL_STOPWORDS:
                keywords.append(w_clean)
                
        if not keywords:
            keywords = [w.strip("?,.!:;()\"'-") for w in words if w.strip("?,.!:;()\"'-")]
            
        print(f"[+] Mots-clés de recherche extraits: {keywords}", flush=True)
        
        scored = []
        for page in pages:
            text = page.page_content.lower()
            score = 0
            
            # 1. Terminology phrase reinforcement across all domains
            for domain, concepts in DOMAIN_CONCEPTS.items():
                lang_syns = concepts.get(doc_lang, concepts.get("en", {}))
                for concept, phrases in lang_syns.items():
                      concept_active = False
                      for phrase in phrases:
                          if phrase in text:
                              score += 150
                              concept_active = True
                      if concept_active:
                          if concept == "left_rotation" and any(k in ["left", "gauche", "sinistra", "izquierda"] for k in keywords):
                              score += 100
                          if concept == "right_rotation" and any(k in ["right", "droite", "destra", "derecha"] for k in keywords):
                              score += 100
                              
            # 2. Generalized keyword occurrences
            for kw in keywords:
                count = 0
                if kw in ["rotation", "rotazione", "rotacion"]:
                    count = max(0, text.count(kw) - text.count("auto" + kw))
                else:
                    count = text.count(kw)
                    
                score += count * 20
                
            scored.append((score, page))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def ingest_high_fidelity(self, file_path, progress_callback=None):
        """
        High-fidelity ingestion using Docling (Sequential Page-by-Page).
        """
        print(f"--- [INGEST] Strategie Sequentielle Granite + Brain Vault : {os.path.basename(file_path)} ---")
        vault_path = "data/brain_vault"
        os.makedirs(vault_path, exist_ok=True)
        
        # Determine the unified crystal path for this document
        import re
        import hashlib
        doc_filename = os.path.basename(file_path)
        doc_clean_name, _ = os.path.splitext(doc_filename)
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', doc_clean_name.replace(' ', '_'))
        doc_id = f"{safe_name}_{int(time.time())}"
        cgc_path = os.path.join(vault_path, f"{doc_id}.cgc")
        if os.path.exists(cgc_path):
            try:
                os.remove(cgc_path)
            except Exception:
                pass
        try:
            from pypdf import PdfReader
            import subprocess
            import json
            import tempfile
            import gc
            
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            all_texts = []
            all_metadatas = []
            
            # 1. Isolated Docling Extraction via Subprocess
            if progress_callback: progress_callback(1, total_pages, "extraction")
            
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                temp_json_path = tmp.name
                
            script_path = os.path.join(os.path.dirname(__file__), "extract_pdf.py")
            python_exe = sys.executable
            
            print("--- [INGEST] Spawning isolated Docling extraction process... ---")
            start_ext = time.time()
            
            # Run extraction in isolated process
            cmd = [python_exe, script_path, file_path, temp_json_path]
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            
            if proc.returncode != 0:
                print(f"!!! [INGEST] Extraction failed (code {proc.returncode}): {proc.stderr}")
                raise RuntimeError(f"Docling extraction failed: {proc.stderr}")
                
            print(f"Docling extraction completed in {time.time() - start_ext:.2f}s.")
            
            with open(temp_json_path, "r", encoding="utf-8") as f:
                pages_data = json.load(f)
                
            try: os.remove(temp_json_path)
            except: pass
            
            # 2. Main Process GPU Neural Capture Loop
            processed_count = 0
            for page_entry in pages_data:
                current_page_num = page_entry["page"]
                page_text = page_entry["text"]
                
                if not page_text.strip():
                    print(f"--- [RAG] Page {current_page_num}/{total_pages} (Vide) ---")
                    if progress_callback: progress_callback(current_page_num, total_pages)
                    continue
                
                print(f"--- [RAG] Page {current_page_num}/{total_pages} (Neural Scan) ---")
                
                # Capture avec retry (max 2 tentatives) si le premier essai echoue
                data = None
                for attempt in range(2):
                    if attempt > 0:
                        print(f"--- [RAG] Retry capture page {current_page_num} (tentative {attempt+1}) ---", flush=True)
                    
                    # Capture Neurale Gemma 4 (GPU)
                    if progress_callback: progress_callback(current_page_num, total_pages, "embedding")
                    start_cap = time.time()
                    try:
                        stream = self.hook.extract_brain_state(
                            page_text, 
                            epoch_id=2026, 
                            geometric=True, 
                            quantize=True, 
                            step_dim=1, 
                            step_seq=self.step_seq, 
                            level=self.zstd_level
                        )
                        cap_duration = time.time() - start_cap
                        raw = stream.getvalue()
                        if len(raw) > 64:
                            data = raw
                            break
                        else:
                            print(f"   [!] Crystal vide ({len(raw)} bytes), retry...", flush=True)
                    except Exception as cap_err:
                        print(f"   [!] Capture echouee: {cap_err}, retry...", flush=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                
                if data is None:
                    print(f"!!! [INGEST] Abandon page {current_page_num} apres 2 tentatives.", flush=True)
                    if progress_callback: progress_callback(current_page_num, total_pages)
                    continue
                
                # Stockage Direct NATIVE (Unified Chromatix CGC format)
                if progress_callback: progress_callback(current_page_num, total_pages, "storage")
                
                data = stream.getvalue()
                
                # Integrity verification : checksum SHA-256 du crystal avant stockage
                data_hash = hashlib.sha256(data).hexdigest()
                
                with open(cgc_path, "ab") as f:
                    offset = f.tell()
                    # Write big-endian 4-byte length prefix
                    f.write(struct.pack('>I', len(data)))
                    f.write(data)
                
                page_lang = self.detect_text_language(page_text)
                metadata = {
                    "source": file_path, 
                    "page": current_page_num,
                    "brain_state_file": cgc_path,
                    "brain_state_offset": offset + 4,
                    "brain_state_len": len(data),
                    "brain_state_img": cgc_path,
                    "index": f"{processed_count}",
                    "detected_language": page_lang,
                    "step_seq": self.step_seq,
                    "step_dim": 1,
                    "crc": data_hash
                }
                
                # Streaming page-by-page db indexing
                if not self.vector_db:
                    self.vector_db = Chroma.from_texts(texts=[page_text], embedding=self.embeddings, persist_directory=self.db_path, metadatas=[metadata])
                else:
                    self.vector_db.add_texts(texts=[page_text], metadatas=[metadata])
                    
                processed_count += 1
                
                # NETTOYAGE AGRESSIF
                torch.cuda.empty_cache()
                gc.collect()
                
                if progress_callback:
                      progress_callback(current_page_num, total_pages, "idle")
                  
                mem_used = torch.cuda.memory_allocated() / 1024**2
                print(f"   [OK] Capture terminee en {cap_duration:.2f}s | VRAM: {mem_used:.2f} MB")
                
            print(f"--- [INGEST] Ingested {processed_count} pages sequentially. ---")
            return processed_count
            
        except Exception as e:
            print(f"!!! [INGEST] Erreur globale : {e}")
            import traceback
            traceback.print_exc()
            return 0

    def process_file(self, file_path):
        """
        Simplified ingestion using PyPDF for quick tests.
        For high-fidelity scientific ingestion, use 'ingest_docling.py'.
        """
        print(f"--- [INGEST] Ingestion Rapide (PyPDF) : {os.path.basename(file_path)} ---")
        try:
            reader = PdfReader(file_path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"!!! [INGEST] Erreur : {e}")
            return 0

        segments = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 150]
        texts = []
        metadatas = []
        
        print(f"--- [CPS] Capture neuronale de {len(segments)} segments... ---")
        for i, seg in enumerate(segments[:10]):
            try:
                # Store brain state in 4-bit NF4 format with no spatial downsampling (Option D: CGC-x1 + NF4)
                stream = self.hook.extract_brain_state(seg, epoch_id=2026, geometric=True, quantize=True, step_dim=1)
                brain_state_hex = stream.getvalue().hex()
                texts.append(seg)
                metadatas.append({"source": file_path, "brain_state": brain_state_hex, "index": i})
            except Exception as e:
                continue

        if texts:
            if not self.vector_db:
                self.vector_db = Chroma.from_texts(texts=texts, embedding=self.embeddings, persist_directory=self.db_path, metadatas=metadatas)
            else:
                self.vector_db.add_texts(texts=texts, metadatas=metadatas)
            
            gc.collect()
            torch.cuda.empty_cache()
            return len(texts)
        return 0
        
    def ask(self, query, history=[]):
        if not self.vector_db:
            yield "Base de données non initialisée."
            return

        # Detect query language
        query_lang = self.detect_text_language(query)
        print(f"--- [RAG] Langue de la requête détectée : {query_lang} ---")

        # 0. Condense Question (pour eviter la pollution du Retrieval)
        search_query = query
        if history:
            condense_prompt = [
                {
                    "role": "system", 
                    "content": "Tu es un assistant de recherche. Écris UNIQUEMENT une courte requête de recherche sémantique (quelques mots-clés) combinant la question avec l'historique. Interdiction stricte d'écrire des phrases d'introduction, du gras (**), des listes, des guillemets ou des explications. Écris juste la requête brute."
                },
            ]
            for msg in history[-3:]:
                condense_prompt.append(msg)
            condense_prompt.append({"role": "user", "content": f"Reformule uniquement cette question pour recherche vectorielle : {query}"})
            
            condensed_prompt_str = self.tokenizer.apply_chat_template(condense_prompt, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(condensed_prompt_str, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
            search_query = self.tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Nettoyage robuste de la requete condensée
            search_query = search_query.replace('**', '').replace('"', '').replace("'", "").strip()
            search_query = search_query.split("\n")[0].strip()
            
            # Suppression des préfixes introduits par le modèle
            prefixes_to_clean = [
                "requête reformulée :", "requête :", "recherche :", "reformulation :", 
                "recherche sémantique :", "recherche sémantique rag :", "requête de recherche :"
            ]
            for prefix in prefixes_to_clean:
                if search_query.lower().startswith(prefix):
                    search_query = search_query[len(prefix):].strip()
            
            print(f"--- [RAG] Query Condensee Finale : {search_query} ---", flush=True)

        # 1. Retrieval
        import time
        start_retrieval = time.time()
        print(f"--- [RAG] Recherche Vectorielle: {search_query} ---")
        docs = self.vector_db.similarity_search(search_query, k=15) 
        print(f"--- [RAG] Retrieval termine en {time.time() - start_retrieval:.2f}s ---")
        print(f"[DEBUG] similarity_search returned {len(docs) if docs is not None else 'None'} documents", flush=True)
        
        if not docs:
            yield "Aucun document pertinent trouvé."
            return

        # Group retrieved pages by source document
        docs_by_source = {}
        for doc in docs:
            src = doc.metadata.get("source", "")
            if src:
                if src not in docs_by_source:
                    docs_by_source[src] = []
                docs_by_source[src].append(doc)

        # Select top page for each unique document using LKPT ranking
        all_ranked_pages = []
        for src, src_docs in docs_by_source.items():
            doc_lang = src_docs[0].metadata.get("detected_language")
            if not doc_lang:
                doc_lang = self.detect_text_language(src_docs[0].page_content)
            
            ranked = self.rank_crystal_pages(src_docs, query, doc_lang)
            for score, page in ranked:
                all_ranked_pages.append((score, page))
                
        # Sort globally by score descending
        all_ranked_pages.sort(key=lambda x: x[0], reverse=True)
        
        # Arbitrage ou Rollback classique
        if self.use_seh:
            # Arbitrage neuronal dynamique via SEH (Winner-Take-All / Compétition d'inhibition)
            selected_top_docs = self.arbitrate_pages_with_seh(all_ranked_pages, query=query)
        else:
            # ROLLBACK : Sélection classique d'origine (mise à jour pour retourner jusqu'à max_references pages)
            selected_top_docs = [page for score, page in all_ranked_pages if score > 0]
            if not selected_top_docs:
                selected_top_docs = [docs[0]]
            else:
                selected_top_docs = selected_top_docs[:self.max_references]

        # Enable/Disable Sliding Window Neighbors (Disabled to avoid context/cache bloat and respect user choice)
        ENABLE_SLIDING_WINDOW = False
        
        enriched_docs = []
        already_present_pages = {} # map: source -> set of page numbers
        
        for doc in selected_top_docs:
            src_path = doc.metadata.get("source", "")
            page_num = doc.metadata.get("page", 1)
            
            if src_path not in already_present_pages:
                already_present_pages[src_path] = set()
                
            if page_num not in already_present_pages[src_path]:
                enriched_docs.append(doc)
                already_present_pages[src_path].add(page_num)
            
            if ENABLE_SLIDING_WINDOW and src_path:
                print(f"--- [RAG] Chargement du voisinage glissant pour {os.path.basename(src_path)} (Page {page_num}) ---")
                for p_offset in [-1, 1]:
                    neighbor_page = page_num + p_offset
                    if neighbor_page > 0 and neighbor_page not in already_present_pages[src_path]:
                        try:
                            res = self.vector_db.get(where={"$and": [{"source": src_path}, {"page": neighbor_page}]})
                            if res and res["documents"]:
                                from langchain_core.documents import Document
                                new_doc = Document(
                                    page_content=res["documents"][0],
                                    metadata=res["metadatas"][0]
                                )
                                enriched_docs.append(new_doc)
                                already_present_pages[src_path].add(neighbor_page)
                                print(f"   [+] Voisin charge: Page {neighbor_page}")
                        except Exception as e:
                              print(f"   [!] Erreur voisin page {neighbor_page}: {e}")
        # Keep original score order (do not sort by page number to keep the best score page first)

        # Build sources list for UI
        sources_info = []
        for idx, d in enumerate(enriched_docs):
            src_metadata = {
                "source": d.metadata.get("source", "Doc"),
                "page": d.metadata.get("page", 1)
            }
            sources_info.append({
                "id": idx + 1, 
                "file": os.path.basename(src_metadata["source"]), 
                "page": src_metadata["page"], 
                "content": d.page_content[:200],
                "metadata": src_metadata
            })

        # Partition context text by document and page
        context_text = ""
        current_src = ""
        for d in enriched_docs:
            src_name = os.path.basename(d.metadata.get("source", ""))
            if src_name != current_src:
                current_src = src_name
                context_text += f"\n\n=== Document: {current_src} ===\n"
            context_text += f"[Page {d.metadata.get('page', 1)}]:\n{d.page_content}\n"

        # Aligned prompt instructions in user's query language
        lang_names = {
            "en": "English",
            "fr": "French / Français",
            "it": "Italian / Italiano",
            "es": "Spanish / Español"
        }
        user_lang_name = lang_names.get(query_lang, "French / Français")
        
        system_prompt = (
            f"Instructions: Tu es un expert de haut niveau dont la mémoire a été fusionnée "
            f"avec le manuel technique correspondant. Réponds précisément en {user_lang_name} "
            f"en te basant sur le contexte technique fourni."
        )

        # Sequential brain state accumulation
        start_injection = time.time()
        pkv = None
        injected_paths = []
        import hashlib
        
        for doc in selected_top_docs:
            cgc_file = doc.metadata.get("brain_state_file")
            offset = doc.metadata.get("brain_state_offset")
            length = doc.metadata.get("brain_state_len")
            stored_hash = doc.metadata.get("crc", "")
            
            # Key to uniquely identify if we've already injected this specific page from this book
            injection_key = f"{cgc_file}:{offset}" if (cgc_file and offset is not None) else doc.metadata.get("brain_state_img", "")
            
            if injection_key and injection_key not in injected_paths:
                if cgc_file and os.path.exists(cgc_file) and offset is not None and length is not None:
                    print(f"--- [CPS] Teleportation de l'ouvrage {os.path.basename(cgc_file)} (Page {doc.metadata.get('page')}) ---")
                    try:
                        with open(cgc_file, "rb") as f:
                            f.seek(int(offset))
                            payload = f.read(int(length))
                            
                        # Verifier l'integrite du crystal avant injection
                        if stored_hash:
                            actual_hash = hashlib.sha256(payload).hexdigest()
                            if actual_hash != stored_hash:
                                print(f"!!! [CPS_INTEGRITY] Echec checksum pour {os.path.basename(cgc_file)} page {doc.metadata.get('page')}. Injection ignorée.", flush=True)
                                continue
                        
                        brain_stream = io.BytesIO(payload)
                        cgc_step_seq = doc.metadata.get("step_seq", self.step_seq)
                        _, pkv = self.hook.inject_brain_state(brain_stream, pkv=pkv, step_seq=cgc_step_seq)
                        injected_paths.append(injection_key)
                    except Exception as e:
                        print(f"!!! [CPS_FAILURE] Erreur lors de l'injection unifiée : {e}")
                else:
                    # Fallback to old behavior: loading the whole file from brain_state_img
                    old_path = doc.metadata.get("brain_state_img", "")
                    if old_path and os.path.exists(old_path):
                        print(f"--- [CPS] Teleportation de l'ouvrage (Legacy) {os.path.basename(old_path)}... ---")
                        try:
                            with open(old_path, "rb") as f:
                                brain_stream = io.BytesIO(f.read())
                                _, pkv = self.hook.inject_brain_state(brain_stream, pkv=pkv, step_seq=self.step_seq)
                            injected_paths.append(old_path)
                        except Exception as e:
                            print(f"!!! [CPS_FAILURE] Erreur lors de l'injection legacy : {e}")

        # Fallback standard if no injections succeeded or if bypassed for debugging
        BYPASS_INJECTION = False  # METTRE A False POUR RE-ACTIVER L'INJECTION DE CACHE
        if BYPASS_INJECTION or not injected_paths:
            print("--- [CPS] Bypassing injection: Fallback sur prefill standard (Mode Safe) ---", flush=True)
            pkv = None

        # Inject conversation history (last 2 exchanges, 2000 tokens max each)
        history_context = []
        for msg in history[-4:]:
            truncated = msg["content"][:2000] if len(msg["content"]) > 2000 else msg["content"]
            history_context.append({"role": msg["role"], "content": truncated})

        # Build prompt messages dynamically depending on KV cache presence
        if pkv is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                *history_context,
                {"role": "user", "content": f"Question : {query}\nRéponds de manière concise et précise, en citant les numéros de page si nécessaire."}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                *history_context,
                {"role": "user", "content": f"Contexte technique :\n{context_text}\n\nQuestion : {query}\nRéponds de manière concise et précise, en citant les numéros de page si nécessaire."}
            ]

        # Build inputs based on model type
        if "gemma-4" in self.model_id:
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokenized = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        else:
            tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        input_ids = tokenized["input_ids"].to("cuda")

        # [WFGY] Action: CUDA_CLEANUP_BEFORE_GEN
        # Clean up intermediate tensors from dequantization to prevent WDDM memory thrashing
        gc.collect()
        torch.cuda.empty_cache()

        print(f"--- [CPS] Nitro-Boost (Multi-Doc) termine en {time.time() - start_injection:.2f}s ---")
        yield {"sources": sources_info}

        # 3. Generation Stream
        print(f"--- [LLM] Lancement de la generation... ---")
        start_gen = time.time()
        
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        if pkv is not None:
            past_len = pkv.get_seq_length()
            new_len = input_ids.shape[1]
            attention_mask = torch.ones((1, past_len + new_len), device="cuda", dtype=torch.long)
            position_ids = torch.arange(past_len, past_len + new_len, device="cuda", dtype=torch.long).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=pkv,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True
                )
                
            first_token_logits = outputs.logits[:, -1, :]
            first_token_id = torch.argmax(first_token_logits, dim=-1)
            
            # [DEBUG] Inspect what token is generated first
            decoded_first = self.tokenizer.decode([first_token_id.item()], skip_special_tokens=False)
            print(f"--- [DEBUG LLM] First token ID: {first_token_id.item()} | Value: {repr(decoded_first)} ---", flush=True)
            
            updated_pkv = outputs.past_key_values
            gen_input_ids = first_token_id.unsqueeze(0)
            attention_mask_gen = torch.ones((1, updated_pkv.get_seq_length() + 1), device="cuda", dtype=torch.long)
            
            generation_kwargs = dict(
                input_ids=gen_input_ids,
                past_key_values=updated_pkv,
                attention_mask=attention_mask_gen,
                use_cache=True,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=False
            )
        else:
            generation_kwargs = dict(
                input_ids=input_ids,
                past_key_values=None,
                attention_mask=None,
                use_cache=True,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=False
            )

        def safe_generate(**kwargs):
            try:
                self.model.generate(**kwargs)
            except Exception as e:
                print(f"!!! [CRITICAL] Crash generate: {e}")
                import traceback
                traceback.print_exc()

        thread = Thread(target=safe_generate, kwargs=generation_kwargs)
        thread.start()
        
        first_token = True
        generated_count = 0
        
        if pkv is not None:
            first_token_str = self.tokenizer.decode([first_token_id.item()], skip_special_tokens=True)
            yield first_token_str
            first_token = False
            generated_count += 1
            
        for new_text in streamer:
            if first_token:
                print(f"--- [LLM] Premier jeton genere en {time.time() - start_gen:.2f}s ---")
                first_token = False
            generated_count += 1
            yield new_text
            
        if generated_count == 0:
            print("!!! [LLM] Aucun jeton genere.")
            yield "Désolé, je n'ai pas pu générer de réponse."

# [WFGY] Zone: SAFE | λ: 0.1 | Action: Remove global instantiation to prevent load-twice bug
# engine = RAGEngine()

if __name__ == "__main__":
    print("--- [RAG] Test du moteur unifie ---")
    engine = RAGEngine()

