# Chromatix RAG Engine 🧠⚡

**Chromatix RAG Engine** is a high-performance, local-first Retrieval-Augmented Generation system. It is specifically optimized for scientific documents and features **Neural Context Teleportation** (Phase 3) via the **Chromatix Pixel Standard (CPS)**.

## 🚀 Key Features

- **Instant Prefill (Phase 3)**: Bypasses traditional LLM prefill latency by injecting pre-calculated KV-cache (Brain States) directly into the model's memory.
- **Scientific OCR (Docling)**: High-fidelity extraction of LaTeX formulas, tables, and complex structures from architectural and scientific PDFs.
- **Mistral-Nemo Native**: Powered by `Mistral-Nemo-12B-Instruct-v1` loaded in 4-bit (NF4) for high-reasoning capabilities on consumer GPUs (RTX 4070 Ti+).
- **Agentic Security**: Context is transported and stored as encrypted "Pixel-Blobs" via the CPS protocol, ensuring bit-perfect fidelity and intrinsic security.
- **Zero-Cloud**: 100% local processing for maximum data privacy.

## 🛠 Tech Stack

- **Inference**: PyTorch / Transformers (with `bitsandbytes` 4-bit quantization).
- **Protocol**: [Chromatix Pixel Standard (CPS)](https://github.com/lecyberbill/Chromatix-Pixel-Standard).
- **Document Processing**: IBM Docling.
- **Vector Database**: ChromaDB.
- **Frontend**: Vite / React (Premium Compositor UI).

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/lecyberbill/Chromatix-RAG-Engine.git
   cd Chromatix-RAG-Engine
   ```

2. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Chromatix Link**:
   Ensure the `chromatix` core library is located at `d:\image_to_text\chromatix` or update the path in `backend/engine.py`.

## 📖 Usage

### 1. Ingestion
Run the bulk ingestion script to process your scientific PDFs and generate Brain States:
```bash
.\ingest.bat
```

### 2. Start the Engine
Launch the backend and frontend simultaneously:
```bash
.\start.bat
```

## 🧠 Phase 3: How it works
Unlike traditional RAG which sends raw text to the LLM, Chromatix RAG:
1. **Encodes** the document into a high-dimensional neural state (KV-cache).
2. **Stores** this state as a binary "Brain State" in the vector database.
3. **Teleports** (Injects) this state directly into the model during a query.
4. **Responds** instantly, as the model "inherits" the context without reading it.

---
**Author:** lecyberbill  
**Protocol:** Chromatix Pixel Standard v1.5
