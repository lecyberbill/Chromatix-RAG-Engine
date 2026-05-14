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

# Chromatix CPS
from chromatix_cps.core import CPSPacket

# PDF Processing
from pypdf import PdfReader, PdfWriter

# [CRITICAL] Docling pour l'ingestion unifiee
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_chroma import Chroma

print("--- [DEBUG] Import Chromatix... ---")
# Add chromatix to path
sys.path.append(r"d:\image_to_text\chromatix")
from chromatix_cps.core import CPSPacket
from chromatix_cps.integrations import TransformersHook

class RAGEngine:
    def __init__(self, model_id="google/gemma-4-E2B-it"):
        # 1. Configuration
        self.model_id = model_id
        self.db_path = "data/db"
        self.pdf_path = "data/pdfs"
        
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
        
        # 2. Local LLM (Chromatix-Native) - Mode Ultra-Light 4-bit + SDPA
        print(f"--- [LLM] Chargement de {self.model_id} (Mode Eco-VRAM)... ---")
        from transformers import AutoProcessor
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        import time
        start_time = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True,
            attn_implementation="sdpa"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        print(f"--- [LLM] Modele et Processor charges en {time.time() - start_time:.2f}s ---")
        self.tokenizer = self.processor.tokenizer
        print("--- [LLM] Tokenizer et Template prets ! ---")
        
        # 3. Chromatix Hook & CPS
        print("--- [CPS] Initialisation du protocole Pixel Standard (No-Thinking Mode)... ---")
        self.hook = TransformersHook(self.model, "RAG-SECURE-STREAM", processor=self.processor)
        self.cps = CPSPacket(master_key="CHROMATIX-NEURAL-RAG-2026")
        
        # 4. Vector DB
        self.vector_db = None
        self.cps = CPSPacket(master_key="RAG-SECURE-STREAM", recovery_key="RAG-RECOVERY-TETHER")
        print("--- [DB] Initialisation de la base vectorielle... ---")
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            os.makedirs(self.db_path, exist_ok=True)

    def ingest_high_fidelity(self, file_path, progress_callback=None):
        """
        High-fidelity ingestion using Docling (Sequential Page-by-Page).
        """
        print(f"--- [INGEST] Strategie Sequentielle Granite + Brain Vault : {os.path.basename(file_path)} ---")
        vault_path = "data/brain_vault"
        os.makedirs(vault_path, exist_ok=True)
        
        try:
            from pypdf import PdfReader, PdfWriter
            import io
            import gc
            import time
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            all_texts = []
            all_metadatas = []
            
            # Initialisation Standard Ultra-Rapide (Zéro-OCR pour PDF Natifs)
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat, DocumentStream
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False
            pipeline_options.do_table_structure = True
            
            print("--- [RAG] Initialisation Moteur Nitro (Standard Fast)... ---")
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            print("--- [RAG] Moteur Nitro PRET ! ---")
            from queue import Queue
            from threading import Thread
            
            # Queue de pages extraites (Buffer pour fluidifier le pipeline)
            extraction_queue = Queue(maxsize=10)
            
            def extraction_worker():
                """Thread producteur : Extrait le texte via Docling en avance sur le GPU."""
                try:
                    for p_num in range(1, total_pages + 1):
                        p_idx = p_num - 1
                        page = reader.pages[p_idx]
                        
                        # Isoler la page dans un buffer
                        writer = PdfWriter()
                        writer.add_page(page)
                        page_stream = io.BytesIO()
                        writer.write(page_stream)
                        page_stream.seek(0)
                        
                        # Docling Extraction
                        if progress_callback: progress_callback(p_num, total_pages, "extraction")
                        doc_stream = DocumentStream(name=f"p{p_num}.pdf", stream=page_stream)
                        result = converter.convert(doc_stream)
                        doc = result.document
                        
                        # Extraction hybride (Math/Texte)
                        seen_content = set()
                        page_blocks = []
                        for item, level in doc.iterate_items():
                            label = getattr(item, "label", "").lower()
                            if any(x in label for x in ["table", "formula", "equation"]):
                                txt = doc.export_to_markdown([item]).strip()
                            else:
                                txt = getattr(item, "text", "").strip()
                            if txt and txt not in seen_content:
                                page_blocks.append(txt)
                                seen_content.add(txt)
                        
                        page_text = "\n\n".join(page_blocks)
                        if len(page_text) > 20000: page_text = page_text[:20000]
                        
                        extraction_queue.put((p_num, page_text))
                    
                    # Signal de fin
                    extraction_queue.put(None)
                except Exception as e:
                    print(f"!!! [PIPELINE_ERROR] Producer: {e}")
                    extraction_queue.put(None)

            # Lancement du producteur
            print("   [NITRO] Lancement du pipeline CPU/GPU...")
            producer_thread = Thread(target=extraction_worker, daemon=True)
            producer_thread.start()

            # Boucle principale (Consommateur GPU)
            processed_count = 0
            while True:
                item = extraction_queue.get()
                if item is None: break
                
                current_page_num, page_text = item
                if not page_text.strip():
                    print(f"--- [RAG] Page {current_page_num}/{total_pages} (Vide) ---")
                    if progress_callback: progress_callback(current_page_num, total_pages)
                    continue
                
                print(f"--- [RAG] Page {current_page_num}/{total_pages} (Neural Scan) ---")
                
                # Capture Neurale Gemma 4 (GPU)
                if progress_callback: progress_callback(current_page_num, total_pages, "embedding")
                start_cap = time.time()
                stream = self.hook.extract_brain_state(page_text, epoch_id=2026)
                cap_duration = time.time() - start_cap
                
                # Stockage Direct NATIVE (Chromatix Binary Compressed)
                if progress_callback: progress_callback(current_page_num, total_pages, "storage")
                img_id = f"{uuid.uuid4()}.cts"
                img_path = os.path.join(vault_path, img_id)
                with open(img_path, "wb") as f:
                    f.write(stream.getvalue())
                
                all_texts.append(page_text)
                all_metadatas.append({
                    "source": file_path, 
                    "page": current_page_num,
                    "brain_state_img": img_path,
                    "index": f"{processed_count}"
                })
                processed_count += 1
                
                # NETTOYAGE AGRESSIF
                torch.cuda.empty_cache()
                gc.collect()
                
                if progress_callback:
                    progress_callback(current_page_num, total_pages, "idle")
                
                mem_used = torch.cuda.memory_allocated() / 1024**2
                print(f"   [OK] Capture terminee en {cap_duration:.2f}s | VRAM: {mem_used:.2f} MB")


            # Finalisation ChromaDB
            if all_texts:
                if not self.vector_db:
                    self.vector_db = Chroma.from_texts(texts=all_texts, embedding=self.embeddings, persist_directory=self.db_path, metadatas=all_metadatas)
                else:
                    self.vector_db.add_texts(texts=all_texts, metadatas=all_metadatas)
                print(f"--- [INGEST] Termine : {len(all_texts)} pages indexees ! ---")
                return len(all_texts)
                
        except Exception as e:
            print(f"!!! [INGEST] Erreur globale : {e}")
            import traceback
            traceback.print_exc()
            return 0
        return 0
        
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
                stream = self.hook.extract_brain_state(seg, epoch_id=2026)
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
        
        return 0

    def ask(self, query, history=[]):
        if not self.vector_db:
            yield "Base de données non initialisée."
            return

        # 0. Condense Question (pour eviter la pollution du Retrieval)
        search_query = query
        if history:
            # On demande au modele de reformuler pour la recherche
            condense_prompt = [
                {"role": "system", "content": "You are a research assistant. Rewrite the user's question to be a standalone search query based on the chat history. Output ONLY the standalone query."},
            ]
            # Ajouter l'historique recent
            for msg in history[-3:]:
                condense_prompt.append(msg)
            condense_prompt.append({"role": "user", "content": f"Rewrite this question for a PDF search: {query}"})
            
            inputs = self.processor.apply_chat_template(condense_prompt, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
            search_query = self.processor.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            print(f"--- [RAG] Query Condensee : {search_query} ---")

        # 1. Retrieval (utilise la query condensee)
        import time
        start_retrieval = time.time()
        print(f"--- [RAG] Recherche : {search_query} ---")
        docs = self.vector_db.similarity_search(search_query, k=1) 
        print(f"--- [RAG] Retrieval termine en {time.time() - start_retrieval:.2f}s ---")
        
        if not docs:
            yield "Aucun document pertinent trouvé."
            return

        doc = docs[0] # Top-1 for Nitro-Boost stability
        
        # Construction d'un objet metadata propre pour le frontend
        clean_metadata = {
            "source": doc.metadata.get("source", "Doc"),
            "page": doc.metadata.get("page", 1)
        }
        img_path = doc.metadata.get("brain_state_img", "")
        
        sources_info = [{
            "id": 1, 
            "file": os.path.basename(clean_metadata["source"]), 
            "page": clean_metadata["page"], 
            "content": doc.page_content[:200],
            "metadata": clean_metadata
        }]

        # 2. Preparation du Prompt (Standard Gemma 4)
        # On utilise le template officiel pour eviter les repetitions infinies
        messages = [
            {"role": "user", "content": f"Voici un extrait du document :\n\n{doc.page_content}\n\nQuestion : {query}\nRéponds de manière concise."}
        ]
        full_instruction = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        start_injection = time.time()
        pkv = None
        
        if img_path and os.path.exists(img_path):
            print(f"--- [CPS] Teleportation One-Shot ({os.path.basename(img_path)})... ---")
            try:
                with open(img_path, "rb") as f:
                    brain_stream = io.BytesIO(f.read())
                    # Injection du Doc uniquement dans le pkv
                    _, pkv = self.hook.inject_brain_state(brain_stream, pkv=pkv)
                
                # [NITRO] L'instruction est envoyée APRES l'injection du cache
                # Comme le cache contient le document brut, on envoie le template complet de la query
                tokens = self.processor.tokenizer(full_instruction, return_tensors="pt", add_special_tokens=False).to("cuda")
                input_ids = tokens.input_ids
            except Exception as e:
                print(f"!!! [CPS_FAILURE] Erreur lors de l'injection : {e}")
                import traceback
                traceback.print_exc()
                img_path = None # Force fallback standard

        if not img_path or not os.path.exists(img_path):
            # Fallback standard : On utilise le même prompt formaté
            print("--- [CPS] Fallback sur prefill standard (Mode Safe) ---")
            tokens = self.processor.tokenizer(full_instruction, return_tensors="pt", add_special_tokens=False).to("cuda")
            input_ids = tokens.input_ids
            pkv = None

        print(f"--- [CPS] Nitro-Boost termine en {time.time() - start_injection:.2f}s ---")
        yield {"sources": sources_info}

        # 3. Generation Stream
        print(f"--- [LLM] Lancement de la generation... ---")
        start_gen = time.time()
        
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # [NITRO] Masque d'attention explicite pour lier le cache CPS au prompt
        past_len = pkv.get_seq_length() if pkv is not None else 0
        current_len = input_ids.shape[1]
        attention_mask = torch.ones((1, past_len + current_len), device="cuda")
        
        print(f"      [DEBUG_GEN] Input: {current_len} | Cache: {past_len} | Total Mask: {attention_mask.shape}", flush=True)
        
        generation_kwargs = dict(
            input_ids=input_ids, 
            past_key_values=pkv,
            attention_mask=attention_mask,
            use_cache=True,
            streamer=streamer, 
            max_new_tokens=1024, 
            do_sample=True, 
            temperature=0.3
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
        for new_text in streamer:
            if first_token:
                print(f"--- [LLM] Premier jeton genere en {time.time() - start_gen:.2f}s ---")
                first_token = False
            generated_count += 1
            yield new_text
            
        if generated_count == 0:
            print("!!! [LLM] Aucun jeton genere.")
            yield "Désolé, je n'ai pas pu générer de réponse."

if __name__ == "__main__":
    print("--- [RAG] Test du moteur unifie ---")
    # Ce bloc est pour le test local uniquement
    # engine = RAGEngine()
    pass
