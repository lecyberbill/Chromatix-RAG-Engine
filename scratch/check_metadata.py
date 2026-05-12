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

# Get all metadata
all_data = db.get()
metadatas = all_data['metadatas']

if metadatas:
    print(f"Sample metadata: {metadatas[0]}")
    pages = [m.get('page') for m in metadatas if 'page' in m]
    if pages:
        print(f"Detected pages: {min(pages)} to {max(pages)}")
    else:
        print("No 'page' key found in metadata. Available keys: ", metadatas[0].keys())
else:
    print("No documents found in DB.")
