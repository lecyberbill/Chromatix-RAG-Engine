import os
import sys

# [CRITICAL] Charger Docling AVANT toute autre lib IA
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

import torch
import gc

# Add paths
sys.path.append(r"d:\image_to_text\RAG\backend")
sys.path.append(r"d:\image_to_text\chromatix")

from langchain_chroma import Chroma
from chromatix_cps.integrations import TransformersHook
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def run_ingestion(pdf_path):
    # Force absolute path for Docling compatibility
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        print(f"!!! [ERROR] Le fichier n'existe pas : {pdf_path}")
        return

    print(f"--- [INGESTOR] Démarrage Ingestion Haute-Fidélité (IBM Docling) : {os.path.basename(pdf_path)} ---")
    
    # 1. Extraction Docling
    print("--- [INGESTOR] Conversion Docling... ---")
    try:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        result = converter.convert(pdf_path)
        content = result.document.export_to_markdown()
        print("--- [INGESTOR] Conversion OK. ---")
    except Exception as e:
        print(f"!!! [INGESTOR] Erreur Docling : {e}")
        return

    # Découpage par paragraphes Markdown
    segments = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 150]
    print(f"--- [INGESTOR] {len(segments)} segments extraits.")

    # 2. Capture des Brain States (Nécessite de charger le modèle temporairement)
    print("--- [INGESTOR] Chargement temporaire du modèle pour capture neuronale... ---")
    model_id = "mistralai/Mistral-Nemo-Instruct-2407"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
    hook = TransformersHook(model, "RAG-SECURE-STREAM")
    
    # 3. Embeddings
    from transformers import AutoModel, AutoTokenizer
    class CustomEmbeddings:
        def __init__(self, model_id):
            print(f"--- [INGESTOR] Embeddings sur GPU... ---")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to("cuda").half()
            self.model.eval()
        def embed_documents(self, texts):
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()
    
    embeddings = CustomEmbeddings("BAAI/bge-small-en-v1.5")

    texts = []
    metadatas = []
    
    for i, seg in enumerate(segments[:20]): # On en prend 20 pour un vrai test
        print(f"Capture segment {i+1}/{len(segments[:20])}...")
        try:
            stream = hook.extract_brain_state(seg, epoch_id=2026)
            brain_state_hex = stream.getvalue().hex()
            texts.append(seg)
            metadatas.append({"source": pdf_path, "brain_state": brain_state_hex, "index": i})
        except: continue

    # 4. Stockage
    print("--- [INGESTOR] Indexation ChromaDB... ---")
    db_path = "data/db"
    Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=db_path, metadatas=metadatas)
    
    print("--- [INGESTOR] Ingestion terminée avec succès ! Libération des ressources... ---")
    del model
    del hook
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_ingestion(sys.argv[1])
    else:
        print("Usage: python ingest_docling.py <path_to_pdf>")
