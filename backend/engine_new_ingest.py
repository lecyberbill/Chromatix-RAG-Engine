    def ingest_high_fidelity(self, file_path):
        """
        High-fidelity ingestion using Docling with Batch Processing (80 pages).
        """
        print(f"--- [INGEST] Strategie Batch Nitro + Brain Vault : {os.path.basename(file_path)} ---")
        vault_path = "data/brain_vault"
        os.makedirs(vault_path, exist_ok=True)
        
        try:
            from pypdf import PdfReader, PdfWriter
            import io
            import gc
            import time
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            # Parametres de batching pour la stabilite (IBM Docling Limit ~80 pages)
            BATCH_SIZE = 80
            all_texts = []
            all_metadatas = []
            
            # Initialisation Docling
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat, DocumentStream
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False 
            pipeline_options.do_table_structure = True
            converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})

            for start_page_idx in range(0, total_pages, BATCH_SIZE):
                end_page_idx = min(start_page_idx + BATCH_SIZE, total_pages)
                print(f"--- [BATCH] Traitement tranches pages {start_page_idx+1} a {end_page_idx} ---")
                
                # Creation d'un mini-PDF pour la tranche
                writer = PdfWriter()
                for i in range(start_page_idx, end_page_idx):
                    writer.add_page(reader.pages[i])
                
                batch_stream = io.BytesIO()
                writer.write(batch_stream)
                batch_stream.seek(0)
                
                # 1. Conversion de la tranche
                start_batch = time.time()
                doc_stream = DocumentStream(name=f"batch_{start_page_idx}.pdf", stream=batch_stream)
                result = converter.convert(doc_stream)
                doc = result.document
                print(f"--- [DOCLING] Tranche convertie en {time.time() - start_batch:.2f}s ---")
                
                # 2. Capture des pages de la tranche
                for local_page_idx in range(doc.page_count):
                    current_page_num = local_page_idx + 1
                    real_page_num = start_page_idx + current_page_num
                    
                    # Extraction du texte (Filtre par page locale)
                    page_text = ""
                    for item, level in doc.iterate_items():
                        if hasattr(item, "prov") and item.prov:
                            if any(p.page_no == current_page_num for p in item.prov):
                                page_text += doc.export_to_markdown([item]) + "\n"
                    
                    if not page_text.strip(): continue
                    
                    try:
                        start_page = time.time()
                        print(f"--- [RAG] Page {real_page_num}/{total_pages} : Capture integrale ({len(page_text)} chars)... ---")
                        
                        # Capture neurale
                        stream = self.hook.extract_brain_state(page_text, epoch_id=2026)
                        brain_data = stream.getvalue()
                        
                        # ENCODAGE CPS
                        img = self.cps.encode_raw_bytes(brain_data)
                        
                        # STOCKAGE IMAGE
                        img_id = f"{uuid.uuid4()}.webp"
                        img_path = os.path.join(vault_path, img_id)
                        img.save(img_path, "WEBP", lossless=True, method=0)
                        
                        all_texts.append(page_text)
                        all_metadatas.append({
                            "source": file_path, 
                            "page": real_page_num,
                            "brain_state_img": img_path,
                            "index": f"{real_page_num}"
                        })
                        
                        print(f"   [OK] Page {real_page_num} capturee en {time.time() - start_page:.2f}s")
                        
                    except Exception as e:
                        print(f"   [ERROR] Echec capture Page {real_page_num} : {str(e)}")
                        continue
                    
                    if real_page_num % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            # 3. Finalisation ChromaDB
            if all_texts:
                if not self.vector_db:
                    self.vector_db = Chroma.from_texts(texts=all_texts, embedding=self.embeddings, persist_directory=self.db_path, metadatas=all_metadatas)
                else:
                    self.vector_db.add_texts(texts=all_texts, metadatas=all_metadatas)
                
                print(f"--- [INGEST] Ingestion terminee. {len(all_texts)} pages indexees ! ---")
                return len(all_texts)
                
        except Exception as e:
            print(f"!!! [INGEST] Erreur globale : {e}")
            import traceback
            traceback.print_exc()
            return 0
        return 0
