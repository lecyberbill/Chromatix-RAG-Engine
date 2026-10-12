# Chromatix RAG Engine 🧠⚡

**Chromatix RAG Engine** is a high-performance, local-first Retrieval-Augmented Generation system. It is specifically optimized for scientific documents and features **Neural Context Teleportation** via the **Chromatix Pixel Standard (CPS)**.

This engine introduces two major scientific innovations in local RAG architectures: **Crystallization** (Neural caching storage) and **SEH (Stabilized Harmonic Equilibrium)** (Neural attractor decision router).

---

## 🚀 Core Innovations

### 1. 💎 Crystallization (Unified Crystal Storage)
Instead of feeding raw text into the LLM's attention mechanism on every query—saturating VRAM and slowing down inference:
- **Page-by-Page Encoding**: Each page is pre-parsed and compiled into its attention memory state (KV-cache).

- **Crystal Format (.cgc)**: This state is compressed into a 4-bit geometric format (NF4) and sequentially serialized as a binary crystal file (`.cgc`), reducing memory footprint by a factor of **650x** (~21 KB/page).

- **Neural Context Teleportation**: At query time, the attention cache is injected (teleported) directly into the GPU, enabling the model to absorb technical context instantly without re-reading the document.


### 2. 🌊 SEH Neural Arbitration (Stabilized Harmonic Equilibrium)
Prevents context pollution caused by near-match, irrelevant pages returned by standard vector search:
- **Winner-Take-All Attractor**: Candidates retrieved from ChromaDB compete within a dynamic neural field governed by the SEH equation (uniform decay, bistability, and global inhibition).

- **Intelligent Filtering**: Parasitic pages are actively suppressed via negative projections, while semantically stable, query-coherent pages emerge as winners and are teleported.

- **Native Rollback**: Includes a dynamic `use_seh = True/False` parameter, allowing immediate fallback to legacy retrieval at any time.


---

## 🚀 Key Features

- **Instant Prefill**: Eliminates prompt-reading latency entirely through direct KV-cache injection.

- **Scientific OCR (Docling)**: High-fidelity extraction of LaTeX equations, tables, and complex document layouts from scientific PDFs.

- **Dynamic Multi-Doc Support**: Dynamically teleports and aggregates attention states across multiple pages or documents (up to `max_references`) within the same session.

- **Zero-Cloud**: Fully local execution ensuring strict privacy and end-to-end data integrity for scientific literature.


---

## 🛠 Tech Stack

- **Inference**: PyTorch / Transformers (bitsandbytes NF4 quantization, custom SDPA monkeypatches).

- **Default LLM**: `google/gemma-4-E2B-it` (NF4).

- **Protocol**: [Chromatix Pixel Standard (CPS)](https://github.com/lecyberbill/Chromatix-Pixel-Standard).

- **Document Processing**: IBM Docling.

- **Vector Database**: ChromaDB.

- **Frontend**: Vite / React (Premium Compositor UI).


---

## 📦 Installation

1. **Set up Virtual Environment**:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Chromatix Link**:
Ensure the chromatix library is available at the path referenced by the backend (d:\image_to_text\chromatix).

---

## 📖 Usage

### 1. Ingestion
1. Ingestion

Run the ingestion pipeline to parse and crystallize PDF files into .cgc neural states:
```bash
.\ingest.bat
```

### 2. Démarrage du serveur et de l'interface
```bash
.\start.bat
```

---
**Author:** lecyberbill  
**Protocol:** Chromatix Pixel Standard v1.5 / SEH V1.3 Integration
