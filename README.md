# Chromatix RAG Engine 🧠⚡

**Chromatix RAG Engine** is a high-performance, local-first Retrieval-Augmented Generation system. It is specifically optimized for scientific documents and features **Neural Context Teleportation** via the **Chromatix Pixel Standard (CPS)**.

This engine introduces two major scientific innovations in local RAG architectures: **Crystallization** (Neural caching storage) and **SEH (Stabilized Harmonic Equilibrium)** (Neural attractor decision router).

---

## 🚀 Core Innovations

### 1. 💎 La Cristallisation (Unified Crystal Storage)
Plutôt que d'envoyer du texte brut à l'attention du LLM à chaque requête, ce qui sature la VRAM et ralentit l'inférence :
* **Page-by-Page Encoding** : Chaque page du document est pré-analysée et transformée en son état de mémoire attentionnel (KV-cache).
* **Format Crystal (.cgc)** : Cet état est compressé en format géométrique 4 bits (NF4) et enregistré de façon séquentielle sous forme de cristal binaire (`.cgc`), réduisant l'empreinte mémoire d'un facteur **650x** (~21 Ko/page).
* **Neural Context Teleportation** : Lors d'une requête, le cache d'attention est injecté (téléporté) directement dans le GPU, permettant au modèle d'assimiler instantanément le contexte technique sans avoir à relire le document.

### 2. 🌊 Arbitrage Neuronal SEH (Stabilized Harmonic Equilibrium)
Pour éviter la pollution du contexte par des pages similaires mais non pertinentes remontées par la recherche vectorielle :
* **Winner-Take-All Attractor** : Les candidats de ChromaDB s'affrontent au sein d'un champ neuronal dynamique régi par l'équation SEH (oubli uniforme, bistabilité et inhibition globale).
* **Filtrage Intelligent** : Les pages parasites sont activement inhibées (projections négatives), tandis que les pages sémantiquement stables et cohérentes avec la question émergent comme gagnantes et sont téléportées.
* **Rollback natif** : Le système inclut un paramètre dynamique `use_seh = True/False` permettant un débrayage complet et un retour vers l'ancien système classique à tout moment.

---

## 🚀 Key Features

- **Instant Prefill** : Plus aucun délai de lecture initiale du prompt grâce à l'injection directe de KV-cache.
- **Scientific OCR (Docling)** : Extraction haute fidélité des formules LaTeX, tableaux et structures complexes à partir des PDFs scientifiques.
- **Dynamic Multi-Doc Support** : Téléporte et accumule dynamiquement les états d'attention de plusieurs pages ou documents différents (jusqu'à `max_references`) dans la même session.
- **Zero-Cloud** : Traitement 100% local préservant l'intégrité et la confidentialité absolue de vos documents scientifiques.

---

## 🛠 Tech Stack

- **Inference** : PyTorch / Transformers (bitsandbytes NF4 quantization, SDPA custom monkeypatches).
- **LLM default** : `google/gemma-4-E2B-it` (NF4).
- **Protocol** : [Chromatix Pixel Standard (CPS)](https://github.com/lecyberbill/Chromatix-Pixel-Standard).
- **Document Processing** : IBM Docling.
- **Vector Database** : ChromaDB.
- **Frontend** : Vite / React (Premium Compositor UI).

---

## 📦 Installation

1. **Setup Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Chromatix Link**:
   Assurez-vous que la librairie `chromatix` est présente à l'emplacement indiqué dans le backend (`d:\image_to_text\chromatix`).

---

## 📖 Usage

### 1. Ingestion
Lancez le script d'ingestion pour numériser et cristalliser vos fichiers PDFs en états neuronaux `.cgc` :
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
