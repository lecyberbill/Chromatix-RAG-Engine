from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

db_path = "backend/data/db"
model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}
)

db = Chroma(persist_directory=db_path, embedding_function=embeddings)

# Iterate over all documents and find 32.1
all_docs = db.get()
found = False
for i, text in enumerate(all_docs['documents']):
    if "32.1" in text:
        print(f"--- Found in Doc {i} ---")
        print(text)
        print("-" * 30)
        found = True

if not found:
    print("32.1 not found in any document.")
