import os
import sys
try:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.pipe.UNIPipe import UNIPipe
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_magic_pdf(pdf_path):
    print(f"Parsing {pdf_path} with Magic-PDF...")
    # This is a simplified version of the API
    try:
        reader = FileBasedDataReader("")
        writer = FileBasedDataWriter("scratch/output")
        # In reality, this requires models and config
        print("Ready to parse (would require magic-pdf.json and models for full power)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pdf = "backend/data/pdfs/RCM_1_2020.pdf"
    if os.path.exists(test_pdf):
        test_magic_pdf(test_pdf)
    else:
        print(f"File not found: {test_pdf}")
