import os
from langchain_docling.loader import DoclingLoader, ExportType
from docling.document_converter import DocumentConverter
import json

def inspect_metadata(file_path):
    loader = DoclingLoader(
        file_path=file_path,
        export_type=ExportType.DOC_CHUNKS,
    )
    
    docs = loader.load()
    print(f"Total chunks: {len(docs)}")
    
    if docs:
        first_doc = docs[0]
        print("\n--- Metadata of first chunk ---")
        # print(json.dumps(first_doc.metadata, indent=2))
        
        # Look for page info in different places
        doc_items = first_doc.metadata.get('doc_items', [])
        if doc_items:
            print(f"Number of doc_items: {len(doc_items)}")
            prov = doc_items[0].get('prov', [])
            if prov:
                print(f"Prov info: {prov}")
        
        # Check for dl_meta
        dl_meta = first_doc.metadata.get('dl_meta', {})
        if dl_meta:
            print(f"dl_meta: {dl_meta}")

if __name__ == "__main__":
    # Test with one of the PDFs in the upload dir
    pdf_dir = r"d:\image_to_text\RAG\backend\data\pdfs"
    files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if files:
        inspect_metadata(os.path.join(pdf_dir, files[0]))
    else:
        print("No PDF files found in data/pdfs")
