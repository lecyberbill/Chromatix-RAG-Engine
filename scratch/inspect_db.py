from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

db_path = "data/db"
model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'}
)

db = Chroma(persist_directory=db_path, embedding_function=embeddings)
print(f"Number of documents in DB: {db._collection.count()}")

# Get some samples
results = db.similarity_search("acier", k=3)
for i, res in enumerate(results):
    print(f"--- Sample {i+1} ---")
    print(res.page_content[:500])
    print("-" * 20)
