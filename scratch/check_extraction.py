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
print(f"Total docs: {db._collection.count()}")

# Search for the specific term
results = db.similarity_search("NF DTU 32.1", k=5)
print(f"Found {len(results)} matches for 'NF DTU 32.1'")
for i, res in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(res.page_content[:1000])
    print("-" * 30)

# Also try a broader search
results_alt = db.similarity_search("normes d'exécution", k=5)
print(f"Found {len(results_alt)} matches for 'normes d'exécution'")
for i, res in enumerate(results_alt):
    print(f"--- Alt Result {i+1} ---")
    print(res.page_content[:1000])
    print("-" * 30)
