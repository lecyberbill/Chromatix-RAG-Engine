import os
import time
from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

print("--- [TEST] Initialisation Granite 258M standalone... ---")
vlm_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
)
vlm_options.vlm_options.load_in_8bit = False
vlm_options.vlm_options.trust_remote_code = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_options,
        ),
    }
)

pdf_path = "data/pdfs/RCM_1_2020.pdf"
if not os.path.exists(pdf_path):
    print(f"ERROR: {pdf_path} not found.")
    exit(1)

print(f"--- [TEST] Conversion Page 1 de {pdf_path}... ---")
start = time.time()
try:
    # On limite a 1 page pour le test
    result = converter.convert(pdf_path, page_range=[1])
    print(f"--- [TEST] SUCCES en {time.time() - start:.2f}s ---")
    print("Contenu extrait :")
    print(result.document.export_to_markdown()[:500])
except Exception as e:
    print(f"--- [TEST] ERREUR : {e} ---")
    import traceback
    traceback.print_exc()
