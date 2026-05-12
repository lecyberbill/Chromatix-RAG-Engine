# [WFGY] Zone: SAFE | λ: 0.2 | Action: Fix Syntax & Multi-format Logic
import os
import torch
import json
import gc
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores.utils import filter_complex_metadata
from pypdf import PdfReader, PdfWriter
import io
import tempfile
import sys
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add chromatix to path
sys.path.append(r"d:\image_to_text\chromatix")
from chromatix_cps.core import CPSPacket
from chromatix_cps.integrations import TransformersHook

class RAGEngine:
    def __init__(self):
        self.db_path = "data/db"
        self.pdf_path = "data/pdfs"
        self.model_id = "mistralai/Mistral-Nemo-Instruct-v1"
        
        # 1. Embedding Model
        model_name = "BAAI/bge-small-en-v1.5"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Local LLM (Chromatix-Native)
        print(f"--- [LLM] Chargement de {self.model_id} en 4-bit... ---")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # 3. Chromatix Hook
        self.hook = TransformersHook(self.model, "RAG-SECURE-STREAM")
        
        # 4. Vector DB
        self.vector_db = None
        self.cps = CPSPacket(master_key="RAG-SECURE-STREAM", recovery_key="RAG-RECOVERY-TETHER")
        self._init_db()

    def _init_db(self):
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        else:
            os.makedirs(self.db_path, exist_ok=True)

    def process_file(self, file_path):
        from langchain_docling.loader import DoclingLoader, ExportType
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat

        print(f"--- [DEBUT] Ingestion de : {file_path} ---")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False 
        pipeline_options.do_table_structure = False
        pipeline_options.images_scale = 0.05
        pipeline_options.generate_page_images = False
        pipeline_options.accelerator_options = AcceleratorOptions(device="cpu", num_threads=1)
        
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        ext = os.path.splitext(file_path)[1].lower()
        all_final_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

        # CAS 1 : PDF -> Ingestion Chirurgicale Page par Page
        if ext == ".pdf":
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for i in range(total_pages):
                current_page_num = i + 1
                print(f"Traitement Page {current_page_num}/{total_pages}...")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_path = tmp_file.name
                        writer = PdfWriter()
                        writer.add_page(reader.pages[i])
                        writer.write(tmp_path)
                    
                    try:
                        loader = DoclingLoader(file_path=tmp_path, export_type=ExportType.DOC_CHUNKS, converter=converter)
                        page_docs = loader.load()
                        for doc in page_docs:
                            doc.metadata['page'] = current_page_num
                            doc.metadata['source_name'] = os.path.basename(file_path)
                            doc.metadata['file_url'] = f"/files/{os.path.basename(file_path)}"
                        all_final_chunks.extend(splitter.split_documents(page_docs))
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                except Exception as e:
                    print(f"!!! Erreur Page {current_page_num}: {e}")
                
                if current_page_num % 5 == 0: gc.collect()
        
        # CAS 2 : DOCX / PPTX -> Ingestion Standard
        else:
            try:
                print(f"Traitement format natif {ext}...")
                loader = DoclingLoader(file_path=file_path, export_type=ExportType.DOC_CHUNKS, converter=converter)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['page'] = 1
                    doc.metadata['source_name'] = os.path.basename(file_path)
                    doc.metadata['file_url'] = f"/files/{os.path.basename(file_path)}"
                all_final_chunks.extend(splitter.split_documents(docs))
            except Exception as e:
                print(f"!!! Erreur Ingestion {ext}: {e}")

        # Sauvegarde d'une version Markdown pour l'aperçu (DOCX/PPTX UNIQUEMENT)
        if ext != ".pdf":
            try:
                print(f"Génération de l'aperçu Markdown pour {ext}...")
                res = converter.convert(file_path)
                md_content = res.document.export_to_markdown()
                md_path = file_path + ".md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
            except Exception as e:
                print(f"!!! Erreur export Markdown pour aperçu: {e}")

        # Indexation finale avec génération de Brain States (Phase 3)
        if all_final_chunks:
            print(f"--- [CPS] Génération des Brain States pour {len(all_final_chunks)} segments... ---")
            for chunk in all_final_chunks:
                try:
                    # Capturer le KV-cache pour ce chunk
                    # On utilise l'epoch 2026 pour le RAG
                    stream = self.hook.extract_brain_state(chunk.page_content, epoch_id=2026)
                    # Stockage en Hex dans les métadonnées (Teleportation Ready)
                    chunk.metadata['brain_state'] = stream.getvalue().hex()
                except Exception as e:
                    print(f"!!! Erreur Brain State pour un chunk: {e}")
                    chunk.metadata['brain_state'] = ""

            texts = filter_complex_metadata(all_final_chunks)
            if self.vector_db is None:
                self.vector_db = Chroma.from_documents(
                    documents=texts, 
                    embedding=self.embeddings, 
                    persist_directory=self.db_path
                )
            else:
                self.vector_db.add_documents(texts)
            print(f"--- [SUCCES] {len(texts)} segments avec Brain States indexés ! ---")
        
        gc.collect()
        return len(all_final_chunks)

    def ask(self, query, history=None):
        if self.vector_db is None:
            yield "Aucun document n'a été analysé."
            return
        
        search_query = query
        if history and len(history) > 0:
            llm_rewriter = ChatOllama(model="mistral-nemo", temperature=0)
            # Prompt plus directif pour la reformulation
            rewrite_prompt = f"""HISTORIQUE DES DERNIERS ÉCHANGES :
{json.dumps(history[-3:], ensure_ascii=False)}

NOUVELLE QUESTION : {query}

MISSION : Reformule la NOUVELLE QUESTION pour une recherche vectorielle. 
IMPORTANT : Ignore les contraintes des questions précédentes (ex: "naturel") si l'utilisateur semble élargir le sujet (ex: "quelles solutions ?"). Capture l'essence de la demande actuelle.
RÈGLE : Réponds UNIQUEMENT par la question reformulée. Pas de blabla."""
            
            res = llm_rewriter.invoke(rewrite_prompt)
            search_query = res.content.strip().replace('"', '')
            print(f"--- [DEBUG] Recherche reformulée : {search_query} ---")

        # 2. Recherche avec filtre de pertinence (seuil de distance)
        docs_with_scores = self.vector_db.similarity_search_with_score(search_query, k=10)
        
        # Filtrage : On ne garde que ce qui est en-dessous de 0.7 (plus permissif pour éviter les faux négatifs)
        THRESHOLD = 0.7
        valid_docs = []
        for doc, score in docs_with_scores:
            if score < THRESHOLD:
                valid_docs.append(doc)
            else:
                print(f"--- [INFO] Segment rejeté (Pertinence : {score:.2f} > {THRESHOLD}) ---")

        sources = []
        for d in valid_docs:
            sources.append({
                "page": d.metadata.get("page", 1),
                "file": d.metadata.get("source_name", "Inconnu"),
                "url": d.metadata.get("file_url", "")
            })
        
        if sources:
            yield f"SOURCES_JSON:{json.dumps(sources)}\n"
        
        context = "\n---\n".join([f"[Source: {d.metadata.get('source_name')} - Page {d.metadata.get('page')}]\n{d.page_content}" for d in valid_docs])
        
        # [CPS INTEGRATION] Generate visual matrix for the context
        if valid_docs:
            try:
                # On encode le contexte complet pour le transport sécurisé
                cps_stream = self.cps.encode_stream(context, epoch_id=2026)
                cps_base64 = cps_stream.getvalue().hex() # Transport simple via hex pour le POC
                yield f"CPS_MATRIX:{cps_base64}\n"
                print(f"--- [CPS] Matrice générée ({len(cps_base64)//2} octets) ---")
            except Exception as e:
                print(f"--- [CPS] Erreur encodage : {e} ---")

        if not valid_docs:
            context = "AUCUN DOCUMENT PERTINENT TROUVÉ DANS LA BASE DE DONNÉES."
        
        # [PHASE 3] Inférence via Injection de Brain State (Instant Prefill)
        if valid_docs:
            print("--- [CPS] Téléportation neuronale en cours... ---")
            # On prend le document le plus pertinent (ou on pourrait concaténer)
            best_doc = valid_docs[0]
            hex_state = best_doc.metadata.get("brain_state", "")
            
            if hex_state:
                try:
                    stream = io.BytesIO(bytes.fromhex(hex_state))
                    
                    # Construction du prompt final (Instruction)
                    instruction = f"\n\nQuestion: {query}\nExpert Scientifique: "
                    
                    # Injection & Génération
                    # Note: inject_brain_state retourne (outputs, tokenizer)
                    outputs, tokenizer = self.hook.inject_brain_state(stream, next_text=instruction)
                    
                    # Décodage de la réponse
                    # On ne veut que ce qui vient après l'instruction
                    generated_ids = outputs.logits.argmax(dim=-1)
                    # Note: La logique de streaming devra être adaptée ici pour un vrai SSE
                    # Pour le POC, on génère une réponse complète ou on utilise un streamer
                    
                    # Version simplifiée pour le POC (Non-streaming mais instantané)
                    # On pourrait utiliser un TextStreamer de HuggingFace
                    
                    full_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    # On nettoie pour ne garder que la réponse à la question
                    if "Expert Scientifique:" in full_response:
                        full_response = full_response.split("Expert Scientifique:")[-1].strip()
                    
                    yield full_response
                    
                except Exception as e:
                    print(f"!!! Erreur Injection : {e}")
                    yield f"❌ Erreur de téléportation : {e}. Fallback vers texte..."
                    # Fallback textuel si l'injection échoue
            else:
                yield "⚠️ Brain State non disponible pour ce segment. Utilisation du texte standard..."
        else:
            yield "Aucun document pertinent trouvé."

engine = RAGEngine()
