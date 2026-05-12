import torch
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    print("Embeddings loaded successfully.")
except Exception as e:
    print(f"Embeddings error: {e}")

try:
    from docling.document_converter import DocumentConverter
    print("Docling loaded successfully.")
except Exception as e:
    print(f"Docling error: {e}")
