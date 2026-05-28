# [WFGY] Zone: SAFE | λ: 0.1 | Action: Docling PDF text extractor (Isolated process)
import sys
import os
import json
import io
from pypdf import PdfReader, PdfWriter

# Ensure stdout uses UTF-8 to prevent encoding issues on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def extract_pdf(file_path):
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    
    # Import docling here to make startup fast
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat, DocumentStream
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    pages_data = []
    for p_num in range(1, total_pages + 1):
        p_idx = p_num - 1
        page = reader.pages[p_idx]
        writer = PdfWriter()
        writer.add_page(page)
        page_stream = io.BytesIO()
        writer.write(page_stream)
        page_stream.seek(0)
        
        doc_stream = DocumentStream(name=f"p{p_num}.pdf", stream=page_stream)
        result = converter.convert(doc_stream)
        doc = result.document
        
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
        
        # Fallback to PyPDF raw extraction if Docling filters out key text segments (like layout-aware columns)
        try:
            raw_txt = page.extract_text()
            if raw_txt:
                # Clean up multiple whitespaces/newlines for matching comparison
                import re
                raw_words = set(re.findall(r'\w+', raw_txt.lower()))
                doc_words = set(re.findall(r'\w+', page_text.lower()))
                
                # Check for significant text omission (more than 15% of unique words missing)
                missing_words = raw_words - doc_words
                ratio_missing = len(missing_words) / max(1, len(raw_words))
                
                # Also capture missed high-value tokens (URLs, emails, numbers/version identifiers)
                has_missed_high_value = False
                for word in missing_words:
                    if '.' in word or '@' in word or any(char.isdigit() for char in word):
                        # URL, email, or version/number identifier missing from Docling output
                        if len(word) > 3:
                            has_missed_high_value = True
                            break
                
                if ratio_missing > 0.15 or has_missed_high_value:
                    print(f"--- [INGEST] Omission detected on page {p_num} (Missing ratio: {ratio_missing:.2%}, High-value missed: {has_missed_high_value}). Appending raw text fallback. ---")
                    # Append the raw text but clean duplicates to keep embeddings clean
                    page_text += "\n\n=== Raw Text Fallback ===\n" + raw_txt
        except Exception as e:
            print(f"[!] PyPDF fallback failed on page {p_num}: {e}")
            
        if len(page_text) > 20000: page_text = page_text[:20000]
        pages_data.append({"page": p_num, "text": page_text})
        
    return pages_data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <pdf_path> <output_json_path>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        data = extract_pdf(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
