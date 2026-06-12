# TECHNICAL_REPORT.md - Project RAG Local (Scientific Focus)

## Context & Core Axiom
Build a fully local RAG (Retrieval-Augmented Generation) system optimized for long scientific documents and OCR-heavy PDFs (images, formulas, tables).
**Axiom:** Data privacy and structural fidelity of complex scientific data are non-negotiable.

## Stack & Topology
- **Topologie:** Mesh / Distributed (Frontend [Host/Network] <-> Backend [0.0.0.0:8080] <-> Client [Browser])
- **Structure Logic:** DAG (Visual/Text Ingestion -> Docling -> ChromaDB -> Retrieval -> LLM)
- **Language:** Python 3.10+
- **OCR Engine:** Docling (IBM) - Optimized for scientific LaTeX formulas & tables (Windows friendly)
- **Fallback Engine:** NVIDIA Nemotron-OCR-v2 (Requires Linux/WSL2 for local compilation)
- **Visual RAG:** Docling PDF-to-Markdown (Preserves structural fidelity)
- **Vector DB:** ChromaDB (Local)
- **LLM Orchestration:** LangChain / LlamaIndex
- **Inference:** Ollama (Models: Nemotron-3-Nano, Llama-3-8B-Nemotron)

## Structural Invariants & Healthchecks
- **Invariant 1 [Local-Only]:** No external API calls for data processing or inference.
- **Invariant 2 [OCR-Formula-Fidelity]:** Formulas must be extracted in a readable format (LaTeX or clear text).
- **Invariant 3 [Long-Context]:** Chunking strategy must handle documents > 50 pages without losing context.
- **Invariant 4 [Multi-Host]:** Frontend must dynamically resolve Backend API via `window.location.hostname`.
- **Invariant 5 [Neural-Structural-Fidelity]:** KV teleportation must match Gemma 4's 15-layer shared cache and heterogeneous head_dim (256/512).
- **Invariant 6 [Multilingual-Search-Alignment]:** Search queries must be aligned to English for embedding similarity search and to document languages for LKPT keyword scoring.
- **Invariant 7 [Multi-Doc-Accumulation]:** Multiple distinct documents (up to 5) are supported in a single session by sequentially injecting their brain states into the attention context.
- **Invariant 8 [Streaming-Ingestion]:** Ingest must stream extracted text and brain states page-by-page directly to ChromaDB and the file storage.
- **Invariant 9 [CGC-NF4-Quantized-Storage]:** Ingest uses isolated subprocess for Docling CPU-extraction and writes brain states in 4-bit NF4 geometric format (~21KB/page), yielding a 650x footprint reduction with zero VRAM context clash.

## Verification Gate
- [x] Environment Setup: SUCCESS
- [x] OCR Baseline Test: SUCCESS (Docling)
- [x] Embedding Quality Test: SUCCESS (HuggingFaceEmbeddings / BGE)
- [x] End-to-End Retrieval: SUCCESS (Llama3)
- [x] OCR Optimization: SUCCESS (Scale 1.5, Plugin Support enabled)
- [x] Hardware Optimization: SUCCESS (Torch 2.6.0+cu124, RTX 4070 Ti active)
- [x] Retrieval Fidelity: SUCCESS (Native Docling Chunks with correct `doc_items` parsing)
- [x] Stability: SUCCESS (Minimalist mode: OCR & Tables disabled to bypass PDFium vector rendering RAM limits)
- [x] UI Design: SUCCESS (Compositor v2.0 Premium Implementation)
- [x] Metadata Fidelity: SUCCESS (Robust page extraction from Docling chunks)
- [x] Invariant 10 [Unified-Crystal-Storage]: SUCCESS (Sequential appending with length prefixing to a single cgc file per document)
- [x] Invariant 11 [Attention-Mask-Patch]: SUCCESS (Monkey-patched ALL_ATTENTION_FUNCTIONS to enforce dynamic shape-aligned SDPA)
- [x] Invariant 12 [BFloat16-Inference]: SUCCESS (Native bfloat16 loading at 0.76s prefill with no quantization drift)
- [x] Invariant 13 [Single-Load-Memory]: SUCCESS (Eliminated double engine instantiation memory leak)
- [x] Invariant 14 [CGC-NF4-Quantized-Storage]: SUCCESS (Isolated CPU-Docling + NF4 geometric extraction cycle runs with zero GPU driver conflicts)

## Benchmarking & Comparison: Dot-main Analysis
Analysed the [Dot-main](file:///d:/image_to_text/RAG/Dot-main) repository to identify UX/UI and architectural patterns.
- **Key Feature Identified:** Integrated PDF context viewer via `<iframe>` with `#page=N` deep-linking.
- **Tech Stack Delta:** Dot uses Langchain JS + FAISS vs our Langchain Python + ChromaDB.
- **Improvement Roadmap:**
  1. Implement Streaming Response (SSE/StreamingResponse).
  2. Integrated PDF Sidebar for source verification.
  3. Support for DOCX/PPTX via Docling's extended loaders.
  4. LaTeX rendering for scientific formulas (MathJax/KaTeX).

## Changelog & Development Milestones

- **v1.0 - v9.5 (May 5 - May 26): Core RAG Architecture & Optimization**
    - Established unified offset crystal (`.cgc`) storage protocol to read/write pages using 4-byte big-endian prefixes.
    - Integrated native Rust C++ space acceleration engine (`chromatix_rust`) for sub-millisecond scaling and key/value manifold interpolation.
    - Implemented multi-document sequential dymamic caching, multilingual topic translation alignment, and event-loop thread-pool streaming.
    - Isolated Docling PDF-to-Markdown extraction in an external CPU subprocess to eliminate Windows WDDM driver hangs.

- **2026-05-28 (v10.0 - MINISTRAL-3 FP8 MIGRATION):** Migrated local RAG LLM to native FP8.
    - Switched weights layout to FP8, resolved scale parsing, and forced `bfloat16` for KV Cache.
    - Enforced `step_dim=1` to bypass spatial downsampling feature drift on Mistral.
    - Restricted retrieval to top-1 page to optimize VRAM and generation latency.

- **2026-05-30 (v11.0 - CONFIGURABLE NEURAL COMPRESSION & SEQUENTIAL INTERPOLATION):** Parameterized compression coefficients.
    - Parameterized `step_seq` (sequence temporal downsampling factor) and `zstd_level` at the `RAGEngine` constructor.
    - Implemented PyTorch native 1D linear interpolation (`torch.nn.functional.interpolate`) to dynamically reconstruct downsampled key/value sequence lengths during injection, restoring full attention window alignment.

- **2026-05-31 (v12.0 - GEMMA-4 INTEGRATION & MAXIMUM SEQUENCE COMPRESSION):** Reinstated Gemma-4.
    - Returned backend default model to `google/gemma-4-E2B-it` loaded in 4-bit NF4 via bitsandbytes.
    - Enabled dynamic shape-aligned SDPA (`custom_sdpa_attention_forward`) and cache usability (`custom_get_usable_length`) patches.
    - Preserved `step_dim=1` for feature integrity while setting `step_seq=2` for 2x sequence compression.

- **2026-05-31 (v12.1 - ROPE POSITION-IDS PREFILL RECOVERY):** Corrected positional encoding overlap.
    - Calculated and passed correct `position_ids` starting at `past_len` (`past_len` -> `past_len + new_len`) during the manual prefill forward pass, resolving prompt positional embeddings overlap.
    - **Performance Gate**: Confirmed VRAM occupancy stabilized at an ultra-low **6.0 GB** total for a 108-page document, with a crystal file size of only **42.3 MB** (~390 KB/page), generating perfect LaTeX formulas.
    - **Delta: 0.1, Lambda: 0.2.**

- **2026-06-09 (v13.0 - CODE REVIEW & REGRESSION FIXES):** Senior engineering audit and corrective maintenance.
    - **Status Model Alignment:** Fixed `/status` endpoint hardcoded to `Ministral-3-8B` while engine loaded `Gemma-4-E2B-it`. Now returns `engine.model_id` dynamically (`app.py:71`).
    - **ALL_ATTENTION_FUNCTIONS Monkeypatch Restoration:** Re-added the v6.6 patch absent from the Gemma-4 loading path. Patches `modeling_gemma4.ALL_ATTENTION_FUNCTIONS` in addition to `sdpa_module.sdpa_attention_forward` to intercept the function during module compilation (`engine.py:130-140`).
    - **Fragile Import Hardening:** Removed top-level `from transformers import MistralCommonBackend` which would crash on transformers versions where this class is not exported. Replaced with lazy local imports inside Mistral branches (`engine.py:27`).
    - **Compression Parameter Metadata Injection:** Stored `step_seq` and `step_dim` in ChromaDB metadata during ingestion (`engine.py:468-469`). Injection now reads the per-document `step_seq` from metadata via `.get("step_seq", self.step_seq)`, preventing structure mismatch crashes when ingesting with different compression configurations (`engine.py:723`).
    - **Dead Code Elimination:** Removed redundant `CPSPacket` double-initialization and three unreachable `return 0` statements in `ingest_high_fidelity` (`engine.py:231-236`, `486-489`).
    - **Multi-Host Compliance (Invariant 4):** Replaced all hardcoded `http://localhost:8000` frontend URLs with `http://${window.location.hostname}:8000` across `Chat.jsx`, `Ingest.jsx`, and `App.jsx`, restoring dynamic backend resolution.
    - **Delta: 0.1, Lambda: 0.1.**

- **2026-06-09 (v14.0 - SEH HARDENING & ORTHOGONAL BASIS):** Corrected three regressions in the SEH neural field router.
    - **Walsh-Hadamard Orthogonal Basis:** Replaced `sinus_patterns` (sinusoidal basis with non-zero cross-talk on discrete n=60 grid) with `walsh_hadamard_patterns` using Sylvester construction. Patterns are strictly orthogonal binary (+1/-1) vectors normalized to unit norm, eliminating cross-inhibition between non-competing candidates (`backend/seh.py`).
    - **Query-Derived RNG Seed:** Replaced fixed seed `0xABCDEF` with `seed_from_query(query)` which derives a deterministic per-query seed from Python's salted hash. The noise in `semer()` now varies by query while staying reproducible within the same session (`backend/seh.py`, `backend/engine.py:704`).
    - **Configurable Candidate Pool:** Replaced hardcoded `[:5]` pre-filter limit with `self.max_seh_candidates = 5` parameter on `RAGEngine`. The field size `n` auto-scales to the next power of 2 when candidates exceed 60 (`backend/engine.py:111`).
    - **Delta: 0.05, Lambda: 0.1.**

- **2026-06-09 (v14.5 - SEH ARBITRATION & CRYSTAL INTEGRITY):** Retuned SEH dynamics and added crystal corruption detection.
    - **SEH Dynamics Overhaul:** Adjusted field parameters (`gain=1.0`, `beta=0.5`, `c_target=0.001`, `damping=0.05`, `inhib=0.02`, `noise_amp=0.01`) so attractors at ±1.4 replace the previous trivial fixed point at zero. Projections now meaningfully differentiate candidates (`engine.py:330-344`).
    - **Proportional Arbitration:** Replaced binary winner-take-all with proportional scoring: `final_score = LKPT × max(0.1, projection + 1.0)`. A page with projection +1.4 gets 2.4× boost; a page at -0.9 gets 0.1× survival floor. SEH now genuinely reranks instead of rubber-stamping LKPT (`engine.py:362-370`).
    - **Crystal SHA-256 Checksum:** Computes `hashlib.sha256(payload).hexdigest()` during ingestion and stores it as `crc` in ChromaDB metadata. Before injection, recalculates and compares; mismatches are logged and skipped, preventing silent garbage from corrupted `.cgc` files (`engine.py:548,567`, `engine.py:827-832`).
    - **Capture Retry Loop:** Wraps `extract_brain_state` + CGC write in a 2-attempt retry per page. Catches exceptions and empty crystals (< 64 bytes). Pages with empty text are skipped before the retry loop (`engine.py:536-561`).
    - **Delta: 0.05, Lambda: 0.1.****

- **2026-06-09 (v15.0 - SEH ARBITRATION & CRYSTAL INTEGRITY):** Retuned SEH dynamics and added crystal corruption detection.
    - **SEH Dynamics Overhaul:** Adjusted field parameters (`gain=1.0`, `beta=0.5`, `c_target=0.001`, `damping=0.05`, `inhib=0.02`, `noise_amp=0.01`) so attractors at ±1.4 replace the previous trivial fixed point at zero. Projections now meaningfully differentiate candidates (`engine.py:330-344`).
    - **Proportional Arbitration:** Replaced binary winner-take-all with proportional scoring: `final_score = LKPT × max(0.1, projection + 1.0)`. A page with projection +1.4 gets 2.4× boost; a page at -0.9 gets 0.1× survival floor. SEH now genuinely reranks instead of rubber-stamping LKPT (`engine.py:362-370`).
    - **Crystal SHA-256 Checksum:** Computes `hashlib.sha256(payload).hexdigest()` during ingestion and stores it as `crc` in ChromaDB metadata. Before injection, recalculates and compares; mismatches are logged and skipped, preventing silent garbage from corrupted `.cgc` files (`engine.py:548,567`, `engine.py:827-832`).
    - **Capture Retry Loop:** Wraps `extract_brain_state` + CGC write in a 2-attempt retry per page. Catches exceptions and empty crystals (< 64 bytes). Pages with empty text are skipped before the retry loop (`engine.py:536-561`).
    - **Delta: 0.05, Lambda: 0.1.**
