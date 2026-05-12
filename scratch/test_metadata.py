from langchain_docling.loader import DoclingLoader, ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False # Fast for test
pipeline_options.do_table_structure = False

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

loader = DoclingLoader(
    file_path="data/pdfs/RCM_1_2020.pdf",
    export_type=ExportType.DOC_CHUNKS,
    converter=converter
)

docs = loader.load()
if docs:
    print("Metadata keys:", docs[0].metadata.keys())
    print("Full metadata:", docs[0].metadata)
    
    # Try to extract page
    for doc in docs[:5]:
        page = None
        if 'doc_items' in doc.metadata and len(doc.metadata['doc_items']) > 0:
            doc_item = doc.metadata['doc_items'][0]
            if 'prov' in doc_item and len(doc_item['prov']) > 0:
                page = doc_item['prov'][0].get('page_no')
        print(f"Extracted page: {page}")
else:
    print("No docs loaded")
