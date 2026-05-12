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

all_docs = db.get()
doc_id = 1226
print(f"--- Doc {doc_id} Content ---")
print(all_docs['documents'][doc_id])
print("-" * 30)

# Search for 32.3
for i, text in enumerate(all_docs['documents']):
    if "32.3" in text:
        print(f"--- Found 32.3 in Doc {i} ---")
        print(text[:200] + "...")
