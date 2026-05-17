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

def run_ingestion_geometric(pdf_path):
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.exists(pdf_path):
        print(f"!!! [ERROR] Le fichier n'existe pas : {pdf_path}")
        return

    print(f"--- [INGESTOR] Démarrage Ingestion GÉOMÉTRIQUE (CGC) : {os.path.basename(pdf_path)} ---")
    
    # 1. Extraction (PDF ou Markdown)
    if pdf_path.endswith(".pdf"):
        print("--- [INGESTOR] Conversion Docling (PDF)... ---")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        result = converter.convert(pdf_path)
        content = result.document.export_to_markdown()
    else:
        print("--- [INGESTOR] Lecture brute (Markdown/Texte)... ---")
        with open(pdf_path, "r", encoding="utf-8") as f:
            content = f.read()

    # Découpage par paragraphes
    segments = [s.strip() for s in content.split("\n\n") if len(s.strip()) > 100]
    print(f"--- [INGESTOR] {len(segments)} segments extraits.")
    
    if not segments:
        print("!!! [ERROR] Aucun segment extrait. Vérifiez le fichier.")
        return

    # 2. Chargement du modèle pour capture
    model_id = "google/gemma-4-E2B-it"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, trust_remote_code=True)
    hook = TransformersHook(model, master_key="RAG-SECURE-STREAM")
    
    # 3. Embeddings (Official LangChain HuggingFace)
    from langchain_huggingface import HuggingFaceEmbeddings
    print("--- [INGESTOR] Chargement des Embeddings (BGE-Small)... ---")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    texts = []
    metadatas = []
    
    for i, seg in enumerate(segments[:10]):
        print(f"--- [INGESTOR] Capture géométrique segment {i+1}/10 ({len(seg)} chars)... ---")
        try:
            # ON ACTIVE LE MODE GÉOMÉTRIQUE ICI
            stream = hook.extract_brain_state(seg, epoch_id=2026, geometric=True, step_dim=4)
            brain_state_hex = stream.getvalue().hex()
            texts.append(seg)
            metadatas.append({
                "source": pdf_path, 
                "brain_state": brain_state_hex, 
                "format": "CGC", # On marque le format
                "index": i
            })
            print(f"      [OK] Capture terminée. Cristal size: {len(brain_state_hex)//2} bytes.")
        except Exception as e:
            import traceback
            print(f"!!! [ERROR] Erreur capture segment {i}: {e}")
            traceback.print_exc()
            continue

    # 4. Stockage
    print(f"--- [INGESTOR] Indexation ChromaDB (Collection Géométrique) - {len(texts)} segments... ---")
    print(f"      [DEBUG] Texts count: {len(texts)}")
    if len(texts) > 0:
        print(f"      [DEBUG] First text length: {len(texts[0])}")
    
    db_path = "data/db_geometric"
    # Ensure directory exists
    if not os.path.exists("data"): os.makedirs("data")
    
    Chroma.from_texts(texts=texts, embedding=embeddings, persist_directory=db_path, metadatas=metadatas)
    
    print("--- [INGESTOR] Ingestion terminée ! ---")
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Test avec un document existant dans le workspace RAG
    test_pdf = r"d:\image_to_text\RAG\data\test_rag_doc.pdf" # Mock path, à vérifier
    if len(sys.argv) > 1:
        run_ingestion_geometric(sys.argv[1])
    else:
        # On cherche un pdf dans data/
        pdfs = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if pdfs:
            run_ingestion_geometric(os.path.join("data", pdfs[0]))
        else:
            print("Aucun PDF trouvé dans data/. Spécifiez un chemin.")
