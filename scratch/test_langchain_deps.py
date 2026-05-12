print("1. Test langchain_chroma...")
from langchain_chroma import Chroma
print("   OK")

print("2. Test langchain_huggingface...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("   OK")
except Exception as e:
    print(f"   ERREUR: {e}")

print("3. Test langchain_text_splitters...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("   OK")

print("4. Test langchain_community...")
from langchain_community.vectorstores.utils import filter_complex_metadata
print("   OK")

print("5. Test terminé.")
