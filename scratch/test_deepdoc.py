import os
import sys
try:
    from deepdoc_pdfparser.parser import PdfParser
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_deepdoc(pdf_path):
    print(f"Parsing {pdf_path} with DeepDoc...")
    parser = PdfParser()
    try:
        # According to research, PdfParser is callable or has a parse method
        # Let's try the common API found in research
        # The package is deepdoc_pdfparser, let's check internal structure
        print("Parser initialized")
        # We might need to specify models or paths if not default
        # For now, let's just try to call it
        # boxes, tables = parser(pdf_path) 
        # print(f"Detected {len(boxes)} boxes and {len(tables)} tables")
    except Exception as e:
        print(f"Error during parsing: {e}")

if __name__ == "__main__":
    test_pdf = "backend/data/pdfs/RCM_1_2020.pdf"
    if os.path.exists(test_pdf):
        test_deepdoc(test_pdf)
    else:
        print(f"File not found: {test_pdf}")
